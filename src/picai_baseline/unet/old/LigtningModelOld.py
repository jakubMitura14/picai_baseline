### Define Data Handling

import concurrent.futures
import functools
import glob
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import operator
import os
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from distutils.log import error
from functools import partial
from glob import glob
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from typing import (Callable, Dict, Hashable, Iterable, List, Optional,
                    Sequence, Sized, Tuple, Union)

import gdown
import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio
import torchio as tio
import torchmetrics
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import (CacheDataset, Dataset, PersistentDataset,
                        decollate_batch, list_data_collate)
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
# lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
from monai.metrics import (ConfusionMatrixMetric, DiceMetric,
                           HausdorffDistanceMetric, SurfaceDistanceMetric,
                           compute_confusion_matrix_metric,
                           do_metric_reduction, get_confusion_matrix)
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers import Norm
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.nets import UNet
from monai.transforms import (AddChanneld, AsDiscrete, AsDiscreted, Compose,
                              ConcatItemsd, CropForegroundd, DivisiblePadd,
                              EnsureChannelFirst, EnsureChannelFirstd,
                              EnsureType, EnsureTyped, Invertd, LoadImaged,
                              MapTransform, Orientationd, RandAdjustContrastd,
                              RandAffined, RandCoarseDropoutd,
                              RandCropByPosNegLabeld, RandFlipd,
                              RandGaussianNoised, RandGaussianSmoothd,
                              RandRicianNoised, RandSpatialCropd, Resize,
                              Resized, ResizeWithPadOrCropd, SaveImage,
                              SaveImaged, ScaleIntensityRanged, SelectItemsd,
                              Spacingd, SpatialPadd)
from monai.utils import alias, deprecated_arg, export, set_determinism
from picai_eval import evaluate
from report_guided_annotation import extract_lesion_candidates
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from torch.nn.intrinsic.qat import ConvBnReLU3d
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader, random_split
from torchmetrics import Precision
from torchmetrics.classification import BinaryF1Score
from torchmetrics.functional import precision_recall
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

import random

# import modelUtlils
import matplotlib.pyplot as plt
import sklearn
from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.eval import evaluate_case
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics
from sklearn.metrics import f1_score

from model import transformsForMain as transformsForMain


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

def getNext(i,results,TIMEOUT):
    try:
        # return it.next(timeout=TIMEOUT)
        return results[i].get(TIMEOUT)

    except Exception as e:
        print(f"timed outt {e} ")
        return None    


def monaiSaveFile(directory,name,arr):
    #Compose(EnsureChannelFirst(),SaveImage(output_dir=directory,separate_folder=False,output_postfix =name) )(arr)
    SaveImage(output_dir=directory,separate_folder=False,output_postfix =name,writer="ITKWriter")(arr)


def getArrayFromPath(path):
    image1=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image1)


def save_heatmap(arr,dir,name,numLesions,cmapp='gray'):
    path = join(dir,name+'.png')
    arr = np.flip(np.transpose(arr),0)
    plt.imshow(arr , interpolation = 'nearest' , cmap= cmapp)
    plt.title( name+'__'+str(numLesions))
    plt.savefig(path)
    return path


def processDecolated(i,gold_arr,y_hat_arr, directory, studyId,imageArr, postProcess,epoch,regr,threshold):
    regr_now = regr[i]
    # if(regr_now==0):
    #     return np.zeros_like(y_hat_arr[i][1,:,:,:])        
    curr_studyId=studyId[i]
    print(f"extracting {curr_studyId}")
    extracted=np.array(extract_lesion_candidates(y_hat_arr[i][1,:,:,:].cpu().detach().numpy(),threshold=threshold)[0]) # dynamic-fast  dynamic
    print(f"extracted {curr_studyId}")
    return extracted

def iterOverAndCheckType(itemm):
    if(type(itemm) is tuple):
        return list(map(lambda en: en.cpu().detach().numpy(),itemm )) 
    if(torch.is_tensor(itemm)):
        return itemm.cpu().detach().numpy()
    return itemm 

def log_images(i,experiment,golds,extracteds ,t2ws, directory,patIds,epoch,numLesions):
    goldChannel=1
    gold_arr_loc=golds[i]
    maxSlice = max(list(range(0,gold_arr_loc.size(dim=3))),key=lambda ind : torch.sum(gold_arr_loc[goldChannel,:,:,ind]).item() )
    
    t2w = t2ws[i][0,:,:,maxSlice].cpu().detach().numpy()
    t2wMax= np.max(t2w.flatten())

    curr_studyId=patIds[i]
    gold=golds[i][goldChannel,:,:,maxSlice].cpu().detach().numpy()
    extracted=extracteds[i]
    #logging only if it is non zero case
    if np.sum(gold)>0:
        experiment.log_image( save_heatmap(np.add(t2w.astype('float'),(gold*(t2wMax)).astype('float')),directory,f"gold_plus_t2w_{curr_studyId}_{epoch}",numLesions[i]))
        experiment.log_image( save_heatmap(np.add(gold*3,((extracted[:,:,maxSlice]>0).astype('int8'))),directory,f"gold_plus_extracted_{curr_studyId}_{epoch}",numLesions[i],'plasma'))
        # experiment.log_image( save_heatmap(gold,directory,f"gold_{curr_studyId}_{epoch}",numLesions[i]))

def getMonaiSubjectDataFromDataFrame(row,label_name,label_name_val,t2wColName
,adcColName,hbvColName ):
        """
        given row from data frame prepares Subject object from it
        """
        subject= {#"chan3_col_name": str(row[chan3_col_name])
        "t2w": str(row[t2wColName]),        
        "t2wb": str(row[t2wColName])        
        ,"hbv": str(row[adcColName])        
        ,"adc": str(row[hbvColName]) 
        ,"labelB"    :str(row[label_name])    
        
       , "isAnythingInAnnotated":int(row['isAnythingInAnnotated'])
        , "study_id":str(row['study_id'])
        , "patient_id":str(row['patient_id'])
        , "num_lesions_to_retain":int(row['num_lesions_to_retain_bin'])
        # , "study_id":row['study_id']
        # , "patient_age":row['patient_age']
        # , "psa":row['psa']
        # , "psad":row['psad']
        # , "prostate_volume":row['prostate_volume']
        # , "histopath_type":row['histopath_type']
        # , "lesion_GS":row['lesion_GS']
        , "label":str(row[label_name])
        , "label_name_val":str(row[label_name_val])
        
        
        }

        return subject
class Model(pl.LightningModule):
    def __init__(self
    , net
    , criterion
    , learning_rate
    , optimizer_class
    ,picaiLossArr_auroc_final
    ,picaiLossArr_AP_final
    ,picaiLossArr_score_final
    ,regression_channels
    ,trial
    ,dice_final
    ,trainSizePercent,batch_size,num_workers
    ,drop_last,df,chan3_col_name,chan3_col_name_val
    ,label_name,label_name_val,
    t2wColName,adcColName,hbvColName,
    RandAdjustContrastd_prob
    ,RandGaussianSmoothd_prob
    ,RandRicianNoised_prob
    ,RandFlipd_prob
    ,RandAffined_prob
    ,RandomElasticDeformation_prob
    ,RandomAnisotropy_prob
    ,RandomMotion_prob
    ,RandomGhosting_prob
    ,RandomSpike_prob
    ,RandomBiasField_prob
    ,persistent_cache
    ,spacing_keyword,netIndex, regr_chan_index,isVnet
    ,train_transforms,train_transforms_noLabel,val_transforms
    ,threshold='dynamic-fast'
    ,toWaitForPostProcess=15
    ,toLogHyperParam=True
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.net=net
        self.modelRegression = UNetToRegresion(2,regression_channels,net)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = monai.metrics.GeneralizedDiceScore()
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]
        self.dices=[]
        self.surfDists=[]
        self.dice_final=dice_final        
        self.picaiLossArr_auroc_final=picaiLossArr_auroc_final
        self.picaiLossArr_AP_final=picaiLossArr_AP_final
        self.picaiLossArr_score_final=picaiLossArr_score_final
        self.temp_val_dir= '/home/sliceruser/locTemp/tempH' #tempfile.mkdtemp()
        self.list_gold_val=[]
        self.list_yHat_val=[]
        self.ldiceLocst_back_yHat_val=[]
        self.isAnyNan=False
        self.postProcessA=monai.transforms.Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postProcess=monai.transforms.Compose([EnsureType(),EnsureChannelFirst(), AsDiscrete(argmax=True, to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postTrue = Compose([EnsureType()])
        self.regLoss = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.df = df
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.train_set = None
        self.val_set = None
        self.test_set = None  
        self.trainSizePercent =trainSizePercent
        self.train_files = None
        self.val_files= None
        self.test_files= None
        self.train_ds = None
        self.val_ds= None
        self.test_ds= None        
        self.subjects= None
        self.chan3_col_name=chan3_col_name
        self.chan3_col_name_val=chan3_col_name_val
        self.label_name=label_name
        self.label_name_val=label_name_val
        self.t2wColName=t2wColName
        self.adcColName=adcColName
        self.hbvColName=hbvColName
        self.RandAdjustContrastd_prob=RandAdjustContrastd_prob
        self.RandGaussianSmoothd_prob=RandGaussianSmoothd_prob
        self.RandRicianNoised_prob=RandRicianNoised_prob
        self.RandFlipd_prob=RandFlipd_prob
        self.RandAffined_prob=RandAffined_prob
        self.RandomElasticDeformation_prob=RandomElasticDeformation_prob
        self.RandomAnisotropy_prob=RandomAnisotropy_prob
        self.RandomMotion_prob=RandomMotion_prob
        self.RandomGhosting_prob=RandomGhosting_prob
        self.RandomSpike_prob=RandomSpike_prob
        self.RandomBiasField_prob=RandomBiasField_prob
        self.persistent_cache=persistent_cache
        self.spacing_keyword=spacing_keyword
        self.netIndex=netIndex
        self.regr_chan_index=regr_chan_index
        self.trial=trial
        self.regressionMetric=BinaryF1Score()
        self.isVnet=isVnet
        os.makedirs(self.temp_val_dir,  exist_ok = True)             
        shutil.rmtree(self.temp_val_dir) 
        os.makedirs(self.temp_val_dir,  exist_ok = True)  
        self.threshold=threshold
        self.toWaitForPostProcess=toWaitForPostProcess
        self.train_transforms =train_transforms
        self.train_transforms_noLabel= train_transforms_noLabel
        self.val_transforms=val_transforms 
        self.toLogHyperParam=toLogHyperParam




    """
    splitting for test and validation and separately in case of examples with some label inside 
        and ecxamples without such constraint
    """
    def getSubjects(self):
        self.df=self.df.loc[self.df['study_id'] !=1000110]# becouse there is error in this label
        self.df=self.df.loc[self.df['study_id'] !=1001489]# becouse there is error in this label
        #onlyPositve = self.df.loc[self.df['isAnyMissing'] ==False]
        onlyPositve = self.df.loc[self.df['isAnythingInAnnotated']>0 ]

        allSubj=list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1]
        ,label_name=self.label_name,label_name_val=self.label_name
        ,t2wColName=self.t2wColName
        ,adcColName=self.adcColName,hbvColName=self.hbvColName )   , list(self.df.iterrows())))
        
        onlyPositiveSubj=list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1]
        ,label_name=self.label_name,label_name_val=self.label_name
        ,t2wColName=self.t2wColName
        ,adcColName=self.adcColName,hbvColName=self.hbvColName )  , list(onlyPositve.iterrows())))
        
        return allSubj,onlyPositiveSubj

    #TODO replace with https://docs.monai.io/en/stable/data.html
    def splitDataSet(self,patList, trainSizePercent,noTestSet):
        """
        test train validation split
        TODO(balance sets)
        """
        totalLen=len(patList)
        train_test_split( patList  )
        numTrain= math.ceil(trainSizePercent*totalLen)
        numTestAndVal=totalLen-numTrain
        numTest=math.ceil(numTestAndVal*0.5)
        numVal= numTestAndVal-numTest

        # valid_set,test_set = torch.utils.data.random_split(test_and_val_set, [math. ceil(0.5), 0.5])
        print('Train data set:', numTrain)
        print('Test data set:',numTest)
        print('Valid data set:', numVal)
        if(noTestSet):
            return torch.utils.data.random_split(patList, [numTrain,numTestAndVal,0])
        else:    
            return torch.utils.data.random_split(patList, [numTrain,numVal,numTest])



    def setup(self, stage=None):
        set_determinism(seed=0)
        # self.subjects = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1]
        # ,self.label_name,self.label_name_val
        #     ,self.t2wColName, self.adcColName,self.hbvColName )   , list(self.df.iterrows())))
        # train_set, valid_set,test_set = self.splitDataSet(self.subjects , self.trainSizePercent,True)
        
        #train_subjects=self.subjects[0:179]
        #val_subjects=self.subjects[180:200]
        # train_subjects = train_set
        # val_subjects = valid_set+test_set
        # self.test_subjects = test_set

        allSubj,onlyPositve=  self.getSubjects()
        allSubjects= allSubj
        onlyPositiveSubjects= onlyPositve        
        random.shuffle(allSubjects)
        random.shuffle(onlyPositiveSubjects)


        self.allSubjects= allSubjects
        self.onlyPositiveSubjects=onlyPositiveSubjects
        onlyNegative=list(filter(lambda subj :  subj['num_lesions_to_retain']==0  ,allSubjects))        
        noLabels=list(filter(lambda subj :  subj['isAnythingInAnnotated']==0 and subj['num_lesions_to_retain']==1 ,allSubjects))        
        print(f" onlyPositiveSubjects {len(onlyPositiveSubjects)} onlyNegative {len(onlyNegative)} noLabels but positive {len(noLabels)}  ")

        if(self.train_transforms==" "):
            self.train_transforms=transformsForMain.get_train_transforms(
                self.RandAdjustContrastd_prob
                ,self.RandGaussianSmoothd_prob
                ,self.RandRicianNoised_prob
                ,self.RandFlipd_prob
                ,self.RandAffined_prob
                ,self.RandomElasticDeformation_prob
                ,self.RandomAnisotropy_prob
                ,self.RandomMotion_prob
                ,self.RandomGhosting_prob
                ,self.RandomSpike_prob
                ,self.RandomBiasField_prob
                ,self.isVnet           
                )
            self.train_transforms_noLabel=transformsForMain.get_train_transforms_noLabel(
                self.RandAdjustContrastd_prob
                ,self.RandGaussianSmoothd_prob
                ,self.RandRicianNoised_prob
                ,self.RandFlipd_prob
                ,self.RandAffined_prob
                ,self.RandomElasticDeformation_prob
                ,self.RandomAnisotropy_prob
                ,self.RandomMotion_prob
                ,self.RandomGhosting_prob
                ,self.RandomSpike_prob
                ,self.RandomBiasField_prob
                ,self.isVnet          
                )


            self.val_transforms= transformsForMain.get_val_transforms(self.isVnet )


        # self.val_ds=     Dataset(data=onlyPositiveSubjects[0:25]+onlyNegative[0:10], transform=val_transforms)
        # self.train_ds_labels = Dataset(data=onlyPositiveSubjects[25:]+onlyNegative[10:], transform=train_transforms)

        # self.val_ds=     LMDBDataset(data=onlyPositiveSubjects[0:25]+onlyNegative[0:10], transform=val_transforms ,cache_dir=self.persistent_cache)
        # self.train_ds_labels = LMDBDataset(data=onlyPositiveSubjects[25:]+onlyNegative[10:], transform=train_transforms  ,cache_dir=self.persistent_cache)
                #self.train_ds_no_labels = SmartCacheDataset(data=noLabels, transform=train_transforms  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())
        # self.val_ds=     SmartCacheDataset(data=onlyPositiveSubjects[0:25]+onlyNegative[0:10], transform=val_transforms  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())
        # self.train_ds_labels = SmartCacheDataset(data=onlyPositiveSubjects[25:]+onlyNegative[10:], transform=train_transforms  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())
        # self.train_ds_no_labels = SmartCacheDataset(data=noLabels, transform=train_transforms_noLabel  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())

        # self.train_ds_all =  LMDBDataset(data=train_set_all, transform=train_transforms,cache_dir=self.persistent_cache)
        onlyPosTreshold=24
        onlyNegativeThreshold=12
        onlyNegativeThresholdB=800
        # self.val_ds=  LMDBDataset(data=onlyPositiveSubjects[0:onlyPosTreshold]+onlyNegative[0:onlyNegativeThreshold], transform=val_transforms ,cache_dir=self.persistent_cache)
        # self.train_ds_labels = LMDBDataset(data=onlyPositiveSubjects[onlyPosTreshold:]+onlyNegative[onlyNegativeThreshold:], transform=train_transforms,cache_dir=self.persistent_cache )
        # self.train_ds_no_labels = LMDBDataset(data=noLabels, transform=train_transforms_noLabel,cache_dir=self.persistent_cache)
        self.val_ds=  Dataset(data=onlyPositiveSubjects[0:onlyPosTreshold]+onlyNegative[0:onlyNegativeThreshold], transform=self.val_transforms )
        self.train_ds_labels = Dataset(data=onlyPositiveSubjects[onlyPosTreshold:]+onlyNegative[onlyNegativeThreshold:onlyNegativeThresholdB], transform=self.train_transforms )
        self.train_ds_no_labels = Dataset(data=noLabels+onlyNegative[onlyNegativeThresholdB:], transform=self.train_transforms_noLabel)



    def train_dataloader(self):
        if(self.current_epoch%2):
            return {'train_ds_labels': DataLoader(self.train_ds_labels, batch_size=self.batch_size, drop_last=self.drop_last
                          ,num_workers=self.num_workers,collate_fn=list_data_collate, shuffle=True )}
        else:
            return {'train_ds_no_labels' : DataLoader(self.train_ds_no_labels, batch_size=self.batch_size, drop_last=self.drop_last
                          ,num_workers=self.num_workers,collate_fn=list_data_collate, shuffle=False)           
                          }                              
        # return {'train_ds_labels': DataLoader(self.train_ds_labels, batch_size=self.batch_size, drop_last=self.drop_last
        #                   ,num_workers=self.num_workers, shuffle=False ) }
        # return {'train_ds_labels': DataLoader(self.train_ds_labels, batch_size=self.batch_size, drop_last=self.drop_last
        #                   ,num_workers=self.num_workers,collate_fn=list_data_collate, shuffle=False ),
        #         'train_ds_no_labels' : DataLoader(self.train_ds_no_labels, batch_size=self.batch_size, drop_last=self.drop_last
        #                   ,num_workers=self.num_workers,collate_fn=list_data_collate, shuffle=False)           
        #                   }# ,collate_fn=list_data_collate ,collate_fn=list_data_collate , shuffle=True ,collate_fn=list_data_collate

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size
        , drop_last=self.drop_last,num_workers=self.num_workers, shuffle=False)#,collate_fn=list_data_collate,collate_fn=pad_list_data_collate




    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
        return [optimizer], [lr_scheduler]


    def infer_train_ds_labels(self, batch):
        x, y, numLesions = batch["train_ds_labels"]['chan3_col_name'] , batch["train_ds_labels"]['label'], batch["train_ds_labels"]['num_lesions_to_retain']
        segmMap,regr = self.modelRegression(x)
        return segmMap,regr, y, numLesions


    def infer_train_ds_no_labels(self, batch):
        x, numLesions =batch["train_ds_no_labels"]['chan3_col_name'],batch["train_ds_no_labels"]['num_lesions_to_retain']
        segmMap,regr = self.modelRegression(x)
        return regr, numLesions


    def training_step(self, batch, batch_idx):
        if(self.current_epoch%2):
            seg_hat,reg_hat, y_true, numLesions=self.infer_train_ds_labels( batch)
            return torch.add(self.criterion(seg_hat,y_true)
                            ,self.regLoss(reg_hat.flatten().float(),torch.Tensor(numLesions).to(self.device).flatten().float() ) 
                            # ,self.regLoss(regr_no_lab.flatten(),torch.Tensor(numLesions_no_lab).to(self.device).flatten() ) 
                                )
        else:
            regr_no_lab, numLesions_no_lab= self.infer_train_ds_no_labels( batch) 
            return self.regLoss(regr_no_lab.flatten().float(),torch.Tensor(numLesions_no_lab).to(self.device).flatten().float() ) 

        # return torch.sum(torch.stack([self.criterion(seg_hat,y_true)
        #                             ,self.regLoss(reg_hat.flatten().float(),torch.Tensor(numLesions).to(self.device).flatten().float() ) 
        #                             ,self.regLoss(regr_no_lab.flatten(),torch.Tensor(numLesions_no_lab).to(self.device).flatten() ) 
        #                                 ]))

    def logHyperparameters(self,experiment):
            
            experiment.log_parameter('spacing_keyword', self.spacing_keyword)
            experiment.log_parameter('netIndex', self.netIndex)
            experiment.log_parameter('regr_chan_index', self.regr_chan_index)

            experiment.log_parameter('RandAdjustContrastd_prob', self.RandAdjustContrastd_prob)
            experiment.log_parameter('RandGaussianSmoothd_prob', self.RandGaussianSmoothd_prob)
            experiment.log_parameter('RandRicianNoised_prob', self.RandRicianNoised_prob)
            experiment.log_parameter('RandFlipd_prob', self.RandFlipd_prob)
            experiment.log_parameter('RandAffined_prob', self.RandAffined_prob)
            experiment.log_parameter('RandomElasticDeformation_prob', self.RandomElasticDeformation_prob)
            experiment.log_parameter('RandomMotion_prob', self.RandomMotion_prob)
            experiment.log_parameter('RandomGhosting_prob', self.RandomGhosting_prob)
            experiment.log_parameter('RandomSpike_prob', self.RandomSpike_prob)
            experiment.log_parameter('RandomBiasField_prob', self.RandomBiasField_prob)



    def validation_step(self, batch, batch_idx):
        print("start validation")
        experiment=self.experiment=self.logger.experiment
        #log hyperparameters if it is epoch 1
        if(self.toLogHyperParam):
            self.logHyperparameters(experiment)

        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name_val'], batch['label_name_val'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        numBatches = y_true.size(dim=0)
        #seg_hat, reg_hat = self.modelRegression(x)        
        # seg_hat, reg_hat = self.modelRegression(x)        
        seg_hat,regr = self.modelRegression(x)
        print(f"regr raw {regr}")
        seg_hat = seg_hat.cpu().detach()
        regr=torch.sigmoid(regr)
        print(f"regr sigm  {regr}")
        # self.regressionMetric(regr.flatten().float(),torch.Tensor(numLesions).to(self.device).flatten().float())
        regrr=torch.round(regr.flatten()).int().cpu().detach().numpy()
        numL=torch.Tensor(numLesions).cpu().int().detach().numpy()
        print(f"regr{regrr} numL {numL} ")

        # f1_scoree = sklearn.metrics.accuracy_score(numL,regrr)
        conff=sklearn.metrics.confusion_matrix(numL,regrr).ravel()
       
        f1_scoree=0.0
        # f1_scoree=float(np.array_equal( numL,regrr )) #exactly the same
        # alt= (1-float(np.array_equal( numL,np.logical_not(numL))))
        if(np.array_equal( numL,regrr )):
            f1_scoree=1.0
        elif(np.array_equal( numL,np.logical_not(numL))):
            f1_scoree=0.0
        else:
            tn, fp, fn, tp = conff 
            f1_scoree=(tp+tn)/(tp+fp+fn+tn)


        
        #f1_scoree = sklearn.metrics.balanced_accuracy_score(numL,regrr)
        print(f"loc f1_score {f1_scoree}")
        self.regressionMetric(torch.round(regr.flatten().float()),torch.Tensor(numLesions).to(self.device).float())
        regr=regr.cpu().detach().numpy()
        # regr= list(map(lambda el : int(el>0.5) ,regr ))
        seg_hat=torch.sigmoid(seg_hat).cpu().detach()
        # diceLocRaw=monai.metrics.compute_generalized_dice( self.postProcessA(seg_hat) ,y_true.cpu())[1].cpu().detach().item()

        # t2wb=decollate_batch(batch['t2wb'])
        # labelB=decollate_batch(batch['labelB'])
        #loss= self.criterion(seg_hat,y_true)# self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)      
        y_det = decollate_batch(seg_hat.cpu().detach())
        # y_background = decollate_batch(seg_hat[:,0,:,:,:].cpu().detach())
        y_true = decollate_batch(y_true.cpu().detach())
        patIds = decollate_batch(batch['study_id'])
        numLesions = decollate_batch(batch['num_lesions_to_retain'])
        images = decollate_batch(x.cpu().detach()) 

        # print(f"val num batches {numBatches} t2wb {t2wb} patIds {patIds} labelB {labelB}")
        print(f"val num batches {numBatches} ")
        lenn=numBatches
        processedCases=[]
        my_task=partial(processDecolated,gold_arr=y_true,y_hat_arr=y_det,directory= self.temp_val_dir,studyId= patIds
                    ,imageArr=images, postProcess=self.postProcess,epoch=self.current_epoch,regr=regr,threshold=self.threshold)
        with mp.Pool(processes = mp.cpu_count()) as pool:
            #it = pool.imap(my_task, range(lenn))
            results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
            time.sleep(60)
            processedCases=list(map(lambda ind :getNext(ind,results,self.toWaitForPostProcess) ,list(range(lenn)) ))

        isTaken= list(map(lambda it:type(it) != type(None),processedCases))
        extracteds=list(filter(lambda it:type(it) != type(None),processedCases))

        lenn=len(extracteds)
        print(f"lenn after extract {lenn}")
        # extracteds=list(filter(lambda it:it.numpy(),extracteds))


        # processedCases=list(map(partial(processDecolated,gold_arr=y_true,y_hat_arr=y_det,directory= self.temp_val_dir,studyId= patIds
        #             ,imageArr=images, experiment=self.logger.experiment,postProcess=self.postProcess,epoch=self.current_epoch)
        #             ,range(0,numBatches)))
        # y_detD=list(map(lambda entry : self.postProcess(entry) ,y_det  ))
        # y_detD= torch.stack(y_detD).cpu()
        # goldsFull = torch.stack(y_true).cpu()
        # diceLocRaw=0.0
        # diceLocRaw=monai.metrics.compute_generalized_dice( y_detD.cpu() ,goldsFull)[1].item()
                

        # try:
        #     diceLocRaw=monai.metrics.compute_generalized_dice( y_detD.cpu() ,goldsFull)[1].item()
        # except:
        #     pass  

        if(len(extracteds)>1):
            directory= self.temp_val_dir
            epoch=self.current_epoch
            list(map(partial(log_images
                ,experiment=experiment,golds=y_true,extracteds=extracteds 
                ,t2ws=images,directory=directory ,patIds=patIds,epoch=epoch,numLesions=numLesions),range(lenn)))
            # y_true= list(map(lambda el: el.numpy()  ,y_true))                                              
            meanPiecaiMetr_auroc=0.0
            meanPiecaiMetr_AP=0.0
            meanPiecaiMetr_score= 0.0
            try:
                valid_metrics = evaluate(y_det=extracteds,
                                        y_true=list(map(lambda el: el.numpy()[1,:,:,:]  ,y_true)),
                                        num_parallel_calls= os.cpu_count()
                                        ,verbose=1)
                meanPiecaiMetr_auroc=0.0 if math.isnan(valid_metrics.auroc) else valid_metrics.auroc
                meanPiecaiMetr_AP=0.0 if math.isnan(valid_metrics.AP) else valid_metrics.AP
                meanPiecaiMetr_score= 0.0 if math.isnan(valid_metrics.score) else  valid_metrics.score
            except:
                pass
            print("start dice")
            extracteds= list(map(lambda numpyEntry : self.postProcess(torch.from_numpy((numpyEntry>0).astype('int8'))) ,extracteds  ))
            extracteds= torch.stack(extracteds)
            



            # extracteds= self.postProcess(extracteds)#argmax=True,
            y_truefil=list(filter(lambda tupl:  isTaken[tupl[0]] , enumerate(y_true)))
            y_truefil=list(map(lambda tupl:  tupl[1] ,y_truefil))
            golds=torch.stack(y_truefil).cpu()

            # print(f"get dice  extrrr {extracteds.cpu()}  Y true  {y_true_prim.cpu()}   ")
            diceLoc=0.0
            # diceLoc=monai.metrics.compute_generalized_dice( extracteds.cpu() ,golds)[1].item()

            try:
                diceLoc=monai.metrics.compute_generalized_dice( extracteds.cpu() ,golds)[1].item()
            except:
                pass    
  


            # print(f"diceLoc {diceLoc} diceLocRaw {diceLocRaw}")

            # gold = list(map(lambda tupl: tupl[2] ,processedCases ))

            return {'dices': diceLoc, 'meanPiecaiMetr_auroc':meanPiecaiMetr_auroc
                    ,'meanPiecaiMetr_AP' :meanPiecaiMetr_AP,'meanPiecaiMetr_score': meanPiecaiMetr_score, 'f1_scoree':f1_scoree}

        return {'dices': 0.0, 'meanPiecaiMetr_auroc':0.0
                ,'meanPiecaiMetr_AP' :0.0,'meanPiecaiMetr_score': 0.0, 'f1_scoree':f1_scoree}




    def validation_epoch_end(self, outputs):
        print("validation_epoch_end")

        allDices = np.array(([x['dices'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'] for x in outputs])).flatten() 
        allaccuracy = np.array(([x['f1_scoree'] for x in outputs])).flatten() 
        
    
        # allDices = np.array(([x['dices'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'].cpu().detach().numpy() for x in outputs])).flatten() 
        regressionMetric=self.regressionMetric.compute()
        self.regressionMetric.reset()
        self.log('regr_F1', regressionMetric)
        
        
        if(len(allDices)>0):            
            meanPiecaiMetr_auroc=np.nanmean(allmeanPiecaiMetr_auroc)
            meanPiecaiMetr_AP=np.nanmean(allmeanPiecaiMetr_AP)
            meanPiecaiMetr_score= np.nanmean(allmeanPiecaiMetr_score)
            accuracy= np.nanmean(allaccuracy)
            meanPiecaiMetr_score_my= (meanPiecaiMetr_auroc+meanPiecaiMetr_AP+accuracy)/3 #np.nanmean(allmeanPiecaiMetr_score)

            self.log('dice', np.nanmean(allDices))

            print(f"accuracy {accuracy} meanPiecaiMetr_score_my {meanPiecaiMetr_score_my} meanDice {np.nanmean(allDices)} regr_F1 {regressionMetric}  meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )
            self.log('val_mean_auroc', meanPiecaiMetr_auroc)
            self.log('val_mean_AP', meanPiecaiMetr_AP)
            self.log('meanPiecaiMetr_score', meanPiecaiMetr_score)
            self.log('accuracy', accuracy)
            
            self.log('score_my', meanPiecaiMetr_score_my)

            self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
            self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
            self.picaiLossArr_score_final.append(meanPiecaiMetr_score)
            self.dice_final.append(np.nanmean(allDices))

 















# def evaluate_all_cases(listPerEval):
#     case_target: Dict[Hashable, int] = {}
#     case_weight: Dict[Hashable, float] = {}
#     case_pred: Dict[Hashable, float] = {}
#     lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
#     lesion_weight: Dict[Hashable, List[float]] = {}

#     meanPiecaiMetr_auroc=0.0
#     meanPiecaiMetr_AP=0.0
#     meanPiecaiMetr_score=0.0

#     idx=0
#     if(len(listPerEval)>0):
#         for pairr in listPerEval:
#             idx+=1
#             lesion_results_case, case_confidence = pairr

#             case_weight[idx] = 1.0
#             case_pred[idx] = case_confidence
#             if len(lesion_results_case):
#                 case_target[idx] = np.max([a[0] for a in lesion_results_case])
#             else:
#                 case_target[idx] = 0

#             # accumulate outputs
#             lesion_results[idx] = lesion_results_case
#             lesion_weight[idx] = [1.0] * len(lesion_results_case)

#         # collect results in a Metrics object
#         valid_metrics = Metrics(
#             lesion_results=lesion_results,
#             case_target=case_target,
#             case_pred=case_pred,
#             case_weight=case_weight,
#             lesion_weight=lesion_weight
#         )
#         # for i in range(0,numIters):
#         #     valid_metrics = evaluate(y_det=self.list_yHat_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
#         #                         y_true=self.list_gold_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
#         #                         num_parallel_calls= min(numPerIter,os.cpu_count())
#         #                         ,verbose=1
#         #                         #,y_true_postprocess_func=lambda pred: pred[1,:,:,:]
#         #                         #y_true=iter(y_true),
#         #                         ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
#         #                         #,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
#         #                         )
#         # meanPiecaiMetr_auroc_list.append(valid_metrics.auroc)
#         # meanPiecaiMetr_AP_list.append(valid_metrics.AP)
#         # meanPiecaiMetr_score_list.append((-1)*valid_metrics.score)
#         #print("finished evaluating")

#         meanPiecaiMetr_auroc=valid_metrics.auroc
#         meanPiecaiMetr_AP=valid_metrics.AP
#         meanPiecaiMetr_score=(-1)*valid_metrics.score
#     return (meanPiecaiMetr_auroc,meanPiecaiMetr_AP,meanPiecaiMetr_score )    





# def saveFilesInDir(gold_arr,y_hat_arr, directory, patId,imageArr, hatPostA):
#     """
#     saves arrays in given directory and return paths to them
#     """
#     adding='_e'
#     monaiSaveFile(directory,patId+ "_gold"+adding,gold_arr)
#     monaiSaveFile(directory,patId+ "_hat"+adding,y_hat_arr)
#     monaiSaveFile(directory,patId+ "image"+adding,imageArr)
#     monaiSaveFile(directory,patId+ "imageB"+adding,imageArr)
#     monaiSaveFile(directory,patId+ "hatPostA"+adding,hatPostA)

#     # gold_im_path = join(directory, patId+ "_gold.npy" )
#     # yHat_im_path = join(directory, patId+ "_hat.npy" )
#     # np.save(gold_im_path, gold_arr)
#     # np.save(yHat_im_path, y_hat_arr)
#     gold_im_path = join(directory, patId+ "_gold.nii.gz" )
#     yHat_im_path =join(directory, patId+ "_hat.nii.gz" )
#     image_path =join(directory, patId+ "image.nii.gz" )
#     imageB_path =join(directory, patId+ "imageB.nii.gz" )
#     hatPostA_path =join(directory, patId+ "hatPostA.nii.gz" )
#     # print(f"suuum image {torch.sum(imageArr)}    suum hat  {np.sum( y_hat_arr.numpy())} hatPostA {np.sum(hatPostA)} hatPostA uniqq {np.unique(hatPostA) } hatpostA shape {hatPostA.shape} y_hat_arr sh {y_hat_arr.shape} gold_arr shape {gold_arr.shape} ")
#     print(f" suum hat  {np.sum( y_hat_arr.numpy())} gold_arr chan 0 sum  {np.sum(gold_arr[0,:,:,:].numpy())} chan 1 sum {np.sum(gold_arr[1,:,:,:].numpy())} hatPostA chan 0 sum  {np.sum(hatPostA[0,:,:,:])} chan 1 sum {np.sum(hatPostA[1,:,:,:])}    ")
#     # gold_arr=np.swapaxes(gold_arr,0,2)
#     # y_hat_arr=np.swapaxes(y_hat_arr,0,2)
#     # print(f"uniq gold { gold_arr.shape  }   yhat { y_hat_arr.shape }   yhat maxes  {np.maximum(y_hat_arr)}  hyat min {np.minimum(y_hat_arr)} ")
#     gold_arr=gold_arr[1,:,:,:].numpy()
#     # gold_arr=np.flip(gold_arr,(1,0))
#     y_hat_arr=y_hat_arr[1,:,:,:].numpy()

#     gold_arr=np.swapaxes(gold_arr,0,2)
#     y_hat_arr=np.swapaxes(y_hat_arr,0,2)
    
#     image = sitk.GetImageFromArray(gold_arr)
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(gold_im_path)
#     writer.Execute(image)


#     image = sitk.GetImageFromArray(y_hat_arr)
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(yHat_im_path)
#     writer.Execute(image) 

#     image = sitk.GetImageFromArray(  np.swapaxes(imageArr[0,:,:,:].numpy(),0,2) ) 
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(image_path)
#     writer.Execute(image)

#     image = sitk.GetImageFromArray(  np.swapaxes(imageArr[1,:,:,:].numpy(),0,2) )
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(imageB_path)
#     writer.Execute(image)

#     image = sitk.GetImageFromArray(np.swapaxes(hatPostA[1,:,:,:],0,2))
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(hatPostA_path)
#     writer.Execute(image)




#     return(gold_im_path,yHat_im_path)






    #     # return {'dices': dices, 'extrCases0':extrCases0,'extrCases1':extrCases1, 'extrCases2':extrCases2 }
    # def processOutputs(self,outputs):
    #     listt = [x['from_case'] for x in outputs] 
    #     listt =[item for sublist in listt for item in sublist]
    #     print(f"listt b {listt}" )
    #     listt= list(map(iterOverAndCheckType, listt))
    #     print(f"listt c {listt}" )
    #     return listt









#         # print( f"rocAuc  {self.rocAuc.aggregate().item()}"  )
#         # #self.log('precision ', monai.metrics.compute_confusion_matrix_metric("precision", confusion_matrix) )
#         # self.rocAuc.reset()        


        
#         print(f" num to validate  { len(self.list_yHat_val)} ")
#         if(len(self.list_yHat_val)>0 ): #and (not self.isAnyNan)
#         # if(False):
#             # with mp.Pool(processes = mp.cpu_count()) as pool:
#             #     dices=pool.map(partial(calcDiceFromPaths,list_yHat_val=self.list_yHat_val,list_gold_val=self.list_gold_val   ),list(range(0,len(self.list_yHat_val))))
#             # dices=list(map(partial(calcDiceFromPaths,list_yHat_val=self.list_yHat_val,list_gold_val=self.list_gold_val   ),list(range(0,len(self.list_yHat_val)))))
#             #meanDice=torch.mean(torch.stack( dices)).item()
#             meanDice=np.mean( self.dices)
#             self.log('meanDice',np.mean( self.dices))
#             print(f"meanDice {meanDice} ")
#             # self.log('meanDice',torch.mean(torch.stack( self.dices)).item() )
#             # print('meanDice',np.mean( np.array(self.dices ).flatten()))
#             # self.log('mean_surface_distance',torch.mean(torch.stack( self.surfDists)).item())

#             lenn=len(self.list_yHat_val)
#             numPerIter=1
#             numIters=math.ceil(lenn/numPerIter)-1



#             meanPiecaiMetr_auroc_list=[]
#             meanPiecaiMetr_AP_list=[]
#             meanPiecaiMetr_score_list=[]
#             print(f" numIters {numIters} ")
            
#             pool = mp.Pool()
#             listPerEval=[None] * lenn

#             # #timeout based on https://stackoverflow.com/questions/66051638/set-a-time-limit-on-the-pool-map-operation-when-using-multiprocessing
#             my_task=partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val)
#             # def my_callback(t):
#             #     print(f"tttttt  {t}")
#             #     s, i = t
#             #     listPerEval[i] = s
#             # results=[pool.apply_async(my_task, args=(i,), callback=my_callback) for i in list(range(0,lenn))]
#             # TIMEOUT = 300# second timeout
#             # time.sleep(TIMEOUT)
#             # pool.terminate()
#             # #filtering out those that timed out
#             # listPerEval=list(filter(lambda it:it!=None,listPerEval))
#             # print(f" results timed out {lenn-len(listPerEval)} from all {lenn} ")

#             TIMEOUT = 50# second timeout


# # TIMEOUT = 2# second timeout
# # with mp.Pool(processes = mp.cpu_count()) as pool:
# #     results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
    
# #     for i in range(lenn):
# #         try:
# #             return_value = results[i].get(2) # wait for up to time_to_wait seconds
# #         except mp.TimeoutError:
# #             print('Timeout for v = ', i)
# #         else:
# #             squares[i]=return_value
# #             print(f'Return value for v = {i} is {return_value}')


# #     # it = pool.imap(my_task, range(lenn))
# #     # squares=list(map(lambda ind :getNext(it,TIMEOUT) ,list(range(lenn)) ))
# # print(squares)


#             with mp.Pool(processes = mp.cpu_count()) as pool:
#                 #it = pool.imap(my_task, range(lenn))
#                 results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
#                 time.sleep(TIMEOUT)
#                 listPerEval=list(map(lambda ind :getNext(ind,results,5) ,list(range(lenn)) ))
#             #filtering out those that timed out
#             listPerEval=list(filter(lambda it:it!=None,listPerEval))
#             print(f" results timed out {lenn-len(listPerEval)} from all {lenn} ")                
#                     # pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,reg_hat=reg_hat),list(range(0,len(y_true))))
#                 # listPerEval=pool.map( partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val) , list(range(0,lenn)))


#             # listPerEval=list(map( partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val) , list(range(0,lenn))))


#             # initialize placeholders

#             # meanPiecaiMetr_auroc=np.nanmean(meanPiecaiMetr_auroc_list)
#             # meanPiecaiMetr_AP=np.nanmean(meanPiecaiMetr_AP_list)
#             # meanPiecaiMetr_score=np.nanmean(meanPiecaiMetr_score_list)
        

      
#             print(f"meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )

#             self.log('val_mean_auroc', meanPiecaiMetr_auroc)
#             self.log('val_mean_AP', meanPiecaiMetr_AP)
#             self.log('mean_val_acc', meanPiecaiMetr_score)
#             # tensorss = [torch.as_tensor(x['loc_dice']) for x in outputs]
#             # if( len(tensorss)>0):
#             #     avg_dice = torch.mean(torch.stack(tensorss))

#             self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
#             self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
#             self.picaiLossArr_score_final.append(meanPiecaiMetr_score)

#             #resetting to 0 
#             self.picaiLossArr_auroc=[]
#             self.picaiLossArr_AP=[]
#             self.picaiLossArr_score=[]







#         #clearing and recreatin temporary directory
#         #shutil.rmtree(self.temp_val_dir)   
#         #self.temp_val_dir=tempfile.mkdtemp() 
#         self.temp_val_dir=pathOs.join('/home/sliceruser/data/tempH',str(self.trainer.current_epoch))
#         os.makedirs(self.temp_val_dir,  exist_ok = True)  


#         self.list_gold_val=[]
#         self.list_yHat_val=[]
#         self.list_back_yHat_val=[]

#         #in case we have Nan values training is unstable and we want to terminate it     
#         # if(self.isAnyNan):
#         #     self.log('val_mean_score', -0.2)
#         #     self.picaiLossArr_score_final=[-0.2]
#         #     self.picaiLossArr_AP_final=[-0.2]
#         #     self.picaiLossArr_auroc_final=[-0.2]
#         #     print(" naans in outputt  ")

#         #self.isAnyNan=False
#         #return {"mean_val_acc": self.log}


#         # # avg_loss = torch.mean(torch.stack([torch.as_tensor(x['val_loss']) for x in outputs]))
#         # # print(f"mean_val_loss { avg_loss}")
#         # # avg_acc = torch.mean(torch.stack([torch.as_tensor(x['val_acc']) for x in outputs]))
#         # #val_accs=list(map(lambda x : x['val_acc'],outputs))
#         # val_accs=list(map(lambda x : x['val_acc'].cpu().detach().numpy(),outputs))
#         # #print(f" a  val_accs {val_accs} ")
#         # val_accs=np.nanmean(np.array( val_accs).flatten())
#         # #print(f" b  val_accs {val_accs} mean {np.mean(val_accs)}")

#         # #avg_acc = np.mean(np.array(([x['val_acc'].cpu().detach().numpy() for x in outputs])).flatten() )

#         # # self.log("mean_val_loss", avg_loss)
#         # self.log("mean_val_acc", np.mean(val_accs))

#         # # self.log('ptl/val_loss', avg_loss)
#         # # self.log('ptl/val_accuracy', avg_acc)
#         # #return {'mean_val_loss': avg_loss, 'mean_val_acc':avg_acc}

# #self.postProcess

# #             image1=sitk.ReadImage(path)
# # #     data = sitk.GetArrayFromImage(image1)



# def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir,images,hatPostA):
# # def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir,reg_hat):
#     return saveFilesInDir(y_true[i],y_det[i], temp_val_dir, patIds[i],images[i],hatPostA[i])
    

# def evaluate_case_for_map(i,y_det,y_true):
#     pred=sitk.GetArrayFromImage(sitk.ReadImage(y_det[i]))
#     pred=extract_lesion_candidates(pred)[0]
#     image = sitk.GetImageFromArray(  np.swapaxes(pred,0,2) )
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(y_det[i].replace(".nii.gz", "_extracted.nii.gz   "))
#     writer.Execute(image)

#     print("evaluate_case_for_map") 
#     return evaluate_case(y_det=y_det[i] 
#                         ,y_true=y_true[i] 
#                         ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

# def getNext(i,results,TIMEOUT):
#     try:
#         # return it.next(timeout=TIMEOUT)
#         return results[i].get(TIMEOUT)

#     except:
#         print("timed outt ")
#         return None    


# def processDice(i,postProcess,y_det,y_true):
#     hatPost=postProcess(y_det[i])
#     # print( f" hatPost {hatPost.size()}  y_true {y_true[i].cpu().size()} " )
#     locDice=monai.metrics.compute_generalized_dice( hatPost ,y_true[i])[1].item()
#     print(f"locDice {locDice}")
#     return (locDice,hatPost.numpy())




        # pathssList=[]
        # dicesList=[]
        # hatPostA=[]
        # # with mp.Pool(processes = mp.cpu_count()) as pool:
        # #     # pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,reg_hat=reg_hat),list(range(0,len(y_true))))
        # #     dicesList=pool.map(partial(processDice,postProcess=self.postProcess,y_det=y_det, y_true=y_true ),list(range(0,len(y_true))))
        # dicesList=list(map(partial(processDice,postProcess=self.postProcess,y_det=y_det, y_true=y_true ),list(range(0,len(y_true)))))

        # hatPostA=list(map(lambda tupl: tupl[1],dicesList ))
        # dicees=list(map(lambda tupl: tupl[0],dicesList ))
        # # self.logger.experiment.

        # # with mp.Pool(processes = mp.cpu_count()) as pool:        
        # #     pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,images=images,hatPostA=hatPostA),list(range(0,len(y_true))))

        # pathssList=list(map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,images=images,hatPostA=hatPostA),list(range(0,len(y_true)))))

        # forGoldVal=list(map(lambda tupl :tupl[0] ,pathssList  ))
        # fory_hatVal=list(map(lambda tupl :tupl[1] ,pathssList  ))
        # # fory__bach_hatVal=list(map(lambda tupl :tupl[2] ,pathssList  ))

        # for i in range(0,len(y_true)):
            
        #     # tupl=saveFilesInDir(y_true[i],y_det[i], self.temp_val_dir, patIds[i])
        #     # print("saving entry   ")
        #     # self.list_gold_val.append(tupl[0])
        #     # self.list_yHat_val.append(tupl[1])
        #     self.list_gold_val.append(forGoldVal[i])
        #     self.list_yHat_val.append(fory_hatVal[i])
        #     self.dices.append(dicees[i])
            # self.list_back_yHat_val.append(fory__bach_hatVal[i])
# #         self.log('val_loss', loss )

#        # return {'loss' :loss,'loc_dice': diceVall }

        #TODO probably this [1,:,:,:] could break the evaluation ...
        # y_det=[x.cpu().detach().numpy()[1,:,:,:][0] for x in y_det]
        # y_true=[x.cpu().detach().numpy() for x in y_true]
        # y_det= list(map(self.postProcess  , y_det))
        # y_true= list(map(self.postTrue , y_det))


        # if(torch.sum(torch.isnan( y_det))>0):
        #     self.isAnyNan=True

        # regress_res2= torch.flatten(reg_hat) 
        # regress_res3=list(map(lambda el:round(el) ,torch.flatten(regress_res2).cpu().detach().numpy() ))

        # total_loss=precision_recall(torch.Tensor(regress_res3).int(), torch.Tensor(numLesions).cpu().int(), average='macro', num_classes=4)
        # total_loss1=torch.mean(torch.stack([total_loss[0],total_loss[1]] ))#self.F1Score
        
        # if(torch.sum(isAnythingInAnnotated)>0):
        #     dice = DiceMetric()
        #     for i in range(0,len( y_det)):
        #         if(isAnythingInAnnotated[i]>0):
        #             y_det_i=self.postProcess(y_det[i])[0,:,:,:].cpu()
        #             y_true_i=self.postTrue(y_true[i])[1,:,:,:].cpu()
        #             if(torch.sum(y_det_i).item()>0 and torch.sum(y_true_i).item()>0 ):
        #                 dice(y_det_i,y_true_i)

        #     self.log("dice", dice.aggregate())
        #     #print(f" total loss a {total_loss1} val_loss {val_losss}  dice.aggregate() {dice.aggregate()}")
        #     total_loss2= torch.add(total_loss1,dice.aggregate())
        #     print(f" total loss b {total_loss2}  total_loss,dice.aggregate() {dice.aggregate()}")
            
        #     self.picaiLossArr_score_final.append(total_loss2.item())
        #     return {'val_acc': total_loss2.item(), 'val_loss':val_losss}
        
        # #in case no positive segmentation information is available
        # self.picaiLossArr_score_final.append(total_loss1.item())
        # return {'val_acc': total_loss1.item(), 'val_loss':val_losss}


    #return {'dices': dices, 'extrCases':extrCases}