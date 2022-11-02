import pytorch_lightning as pl
import argparse
import ast
from functools import partial
import numpy as np
import torch
import monai
from training_setup.callbacks import (
    optimize_model, resume_or_restart_training, validate_model,resume_or_restart_training_tracking)
from training_setup.compute_spec import \
    compute_spec_for_run
from training_setup.data_generator import prepare_datagens
from training_setup.default_hyperparam import \
    get_default_hyperparams
from training_setup.loss_functions.focal import FocalLoss
from training_setup.neural_network_selector import \
    neural_network_for_run
# from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import torch
from picai_eval import evaluate
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
from os.path import basename, dirname, exists, isdir, join, split
import tempfile
import multiprocessing as mp
from torchmetrics.classification import BinaryF1Score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.intrinsic.qat import ConvBnReLU3d



class UNetToRegresion(nn.Module):
    def __init__(self,
        in_channels,
        regression_channels
        ,segmModel
    ) -> None:
        super().__init__()
        print(" in UNetToRegresion {in_channels}")
        self.segmModel=segmModel
        self.model = nn.Sequential(
            ConvBnReLU3d(in_channels=in_channels, out_channels=regression_channels[0], kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[0], out_channels=regression_channels[1], kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[1], out_channels=regression_channels[2], kernel_size=3, stride=1,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[2], out_channels=1, kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            nn.AdaptiveMaxPool3d((8,8,2)),#ensuring such dimension 
            nn.Flatten(),
            #nn.BatchNorm3d(8*8*4),
            nn.Linear(in_features=8*8*2, out_features=100),
            #nn.BatchNorm3d(100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=1)
            # ,torch.nn.Sigmoid()
        )
    def forward(self, x):
        segmMap=self.segmModel(x)
        #print(f"segmMap  {segmMap}")
        return (segmMap,self.model(segmMap))





# def getSwinUNETRa(dropout,input_image_size,in_channels,out_channels):
#     return monai.networks.nets.SwinUNETR(
#         spatial_dims=3,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         img_size=input_image_size,
#         #depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24)
#         #depths=(4, 4, 4, 4), num_heads=(6, 12, 24, 48)
#     )

# def getSwinUNETRb(dropout,input_image_size,in_channels,out_channels):
#     return monai.networks.nets.SwinUNETR(
#         spatial_dims=3,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         img_size=input_image_size,
#         #depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24)
#         depths=(4, 4, 4, 4), num_heads=(6, 12, 24, 48)
#     )

def getSegResNeta(dropout,input_image_size,in_channels,out_channels):
    input_image_size=(3,32,256,256)
    return (monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
        # blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
        blocks_down=(8, 16,16, 32), blocks_up=(8, 8, 8)
    ),input_image_size,8)

def getSegResNetb(dropout,input_image_size,in_channels,out_channels):
    input_image_size=(3,32,256,256)
    return (monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
        # blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
        blocks_down=(4, 8, 8, 16), blocks_up=(4, 4, 4)
    ),input_image_size,14)
 
def getVneta(dropout,input_image_size,in_channels,out_channels):
    input_image_size=(3,20,256,256)
    return (monai.networks.nets.ViT(in_channels=in_channels, num_classes=out_channels
    ,img_size=input_image_size, pos_embed='conv', classification=True, spatial_dims=3
    , num_layers=12, num_heads=12 , dropout_rate=dropout,patch_size=(16,16,16) ) 
    ,input_image_size,32)


def getVnetb(dropout,input_image_size,in_channels,out_channels):
    input_image_size=(3,20,256,256)
    return (monai.networks.nets.ViT(in_channels=in_channels, num_classes=out_channels
    ,img_size=input_image_size, pos_embed='conv', classification=True, spatial_dims=3
    , num_layers=24, num_heads=24 , dropout_rate=dropout,patch_size=(16,16,16)) 
    ,input_image_size,30)
    

def getVnetc(dropout,input_image_size,in_channels,out_channels):
    input_image_size=(3,20,256,256)
    return (monai.networks.nets.ViT(in_channels=in_channels, num_classes=out_channels
    ,img_size=input_image_size, pos_embed='conv', classification=True, spatial_dims=3
    , num_layers=48, num_heads=24 , dropout_rate=dropout,patch_size=(16,16,16)) 
    ,input_image_size,22)




def getUneta(args,devicee):
    return (neural_network_for_run(args=args, device=devicee),(3,20,256,256),32)

# def getUnetb(args,devicee):
#     args.model_features = [ 64, 128, 256, 512, 1024,2048]
#     return (neural_network_for_run(args=args, device=devicee),(3,20,256,256),32)

# def getVNet(dropout,input_image_size,in_channels,out_channels):
#     return (monai.networks.nets.VNet(
#         spatial_dims=3,
#         in_channels=4,
#         out_channels=out_channels,
#         dropout_prob=dropout
#     ),(4,32,256,256),6)


class UnetWithTransformerA(nn.Module):
    def __init__(self,
        dropout
        ,input_image_size
        ,in_channels
        ,out_channels
        ,args
        ,devicee
    ) -> None:
        super().__init__()
        self.unet = getUneta(args,devicee)
        self.tranformer = getVnetb(dropout,input_image_size,2,out_channels)
        
    def forward(self, x):
        return self.tranformer(self.unet(x))


class UnetWithTransformerB(nn.Module):
    def __init__(self,
        dropout
        ,input_image_size
        ,in_channels
        ,out_channels
        ,args
        ,devicee
    ) -> None:
        super().__init__()
        self.unet = getUneta(args,devicee)
        self.tranformer = getVnetb(dropout,input_image_size,3,3)
        
    def forward(self, x):
        return self.unet(self.tranformer(x))        


def getUnetWithTransformerA(dropout,input_image_size,in_channels,out_channels,args,devicee):
    input_image_size=(3,20,256,256)
    return (UnetWithTransformerA(dropout,input_image_size,in_channels,out_channels,args,devicee),
    input_image_size,22)

def getUnetWithTransformerB(dropout,input_image_size,in_channels,out_channels,args,devicee):
    input_image_size=(3,20,256,256)
    return (UnetWithTransformerB(dropout,input_image_size,in_channels,out_channels,args,devicee),
    input_image_size,22)    