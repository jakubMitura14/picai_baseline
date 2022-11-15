
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from comet_ml import Optimizer
import multiprocessing as mp
from optuna.storages import RetryFailedTrialCallback

import time
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from monai.utils import set_determinism
import math
import torch
from torch.utils.data import random_split, DataLoader
import monai

import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from datetime import datetime
import os
import tempfile
from glob import glob
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from monai.networks.layers.factories import Act, Norm
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import torch_optimizer as optim
monai.utils.set_determinism()
import geomloss
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from ray import air, tune
# from ray.air import session
# from ray.tune import CLIReporter
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import importlib.util
import sys
import LightningModel
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import basename, dirname, exists, isdir, join, split
import json
from monai.transforms import Compose, EnsureType
from functools import partial
import training_setup.data_generator as data_generator
import training_setup.augmentations.nnUNet_DA as nnUNet_DA
import training_setup.adaptTransforms as adaptTransforms


def mainTrain(project_name,args,trial: optuna.trial.Trial,imageShape) -> float:
    swa_lrs=0.01#trial.suggest_float("swa_lrs", 1e-1,0.5) #trial.suggest_float("swa_lrs", 1e-6, 1e-4)
    base_lr_multi =1.0#trial.suggest_float("base_lr_multi", 0.1, 3.0)
    schedulerIndex=2#trial.suggest_int("scheduler_int", 0, 2)
    #modelIndex=2
    modelIndex=0#trial.suggest_int("modelIndex", 0, 4)
    #modelIndex=0
    normalizationIndex=0#trial.suggest_int("normalizationIndex", 0, 1)
    dropout=0.1#trial.suggest_float("dropout", 0.0,0.4)
    RicianNoiseTransformProb=0.05#trial.suggest_float("dropout", 0.0,0.35)
    LocalSmoothingTransformProb=0.9#trial.suggest_float("dropout", 0.0,0.35)
    RandomBiasField_prob=0.1#trial.suggest_float("dropout", 0.0,0.35)
    RandomAnisotropy_prob=0#trial.suggest_float("dropout", 0.0,0.35)
    Random_GaussNoiseProb=0.05#trial.suggest_float("dropout", 0.0,0.15)
    optimizerIndex=0#trial.suggest_int("optimizerIndex", 0, 2)

    regression_channels=[64,128,256]

    machine = os.environ['machine']
    expId=trial.number

    toMonitor="valid_ranking"
        # optuna_prune=PyTorchLightningPruningCallback(trial, monitor=toMonitor)     
    # if(swa_lrs>0.0):
    #     callbacks=[early_stopping,stochasticAveraging], #optuna_prune
    # check_eval_every_epoch=40
    check_eval_every_epoch=5
    # for each fold
    fInd=-1

    for f in args.folds :#range(0, len(args.folds)):#args.folds:
        fInd=fInd+1
        outputPAth = join('/home/sliceruser/data/',str(fInd) )
        os.makedirs(outputPAth ,exist_ok = True)
        if(fInd>-1):
            comet_logger = CometLogger(
                api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
                #workspace="OPI", # Optional
                project_name=project_name, # Optional
                experiment_name=f"{machine}_{fInd}_{str(expId)}" # Optional
                #experiment_name=experiment_name # Optional
            )
            stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs= 0.5 )
            early_stopping = pl.callbacks.early_stopping.EarlyStopping(
                monitor=toMonitor,
                patience=5,
                mode="max",
                #divergence_threshold=(-0.1)
            )
            # f=args.folds[fInd]
            checkPointPath=f"/home/sliceruser/locTemp/checkPoints/{project_name}/{expId}/{fInd}"
            checkPointPathFromOut=f"/home/sliceruser/locTemp/checkB/checkpoints/f{fInd}.ckpt"
            checkpoint_callback = ModelCheckpoint(dirpath= checkPointPath,mode='max', save_top_k=1, monitor=toMonitor)
            schedulerIndexToLog= schedulerIndex
            callbacks=[early_stopping,checkpoint_callback,stochasticAveraging]#stochasticAveraging
            device = 'cuda'

            logImageDir=tempfile.mkdtemp()
            
            learningRate = 0.0057# manually taken from learning rate finder
            model = LightningModel.Model.load_from_checkpoint(checkPointPathFromOut).model.to(device)
            
            
            
            
            csvPath='/home/sliceruser/labels/clinical_information/marksheet.csv'
            df = pd.read_csv(csvPath)
            with open(args.overviews_dir+'PI-CAI_train-fold-'+str(fInd)+'.json') as fp:
                train_json = json.load(fp)
        
            pretx = [EnsureType()]
            
            train_data = [np.array(train_json['image_paths']), np.array(train_json['label_paths'])]
            subjects_train = list(map(partial(data_generator.getPatientDict,image_files=train_data[0], seg_files=train_data[1]) , range(0,len(train_data[0])) ))

            batchh=1
            expectedShape=(3,32,256,256)
            transfVal=adaptTransforms.loadValTransform(Compose(pretx),Compose(pretx),normalizationIndex,{},expectedShape,df)
            transfVal=Compose(transfVal,monai.transforms.ToTensord(keys=["data","seg"])  )
            valid_ds=Dataset(data=subjects_train, transform=transfVal )
            valid_ldr=DataLoader(valid_ds,batch_size=batchh, num_workers=6,shuffle=False)

            for i, batch_data in enumerate(valid_ldr, 0):
                inputs = batch_data['data'].to(device)
                output = model(inputs)[0,1,:,:,:].detach().cpu().numpy()
                studyId= batch_data['study_id'][0]             
                print("*")

                outPathFile = join(outputPAth,f"{studyId}.mha")
                image = sitk.GetImageFromArray(output)
                sitk.WriteImage(image, str(outPathFile))  
            print(f"finished fold {fInd}")
            

            
            
            # model = LightningModel.Model(f,args,args.base_lr,base_lr_multi
            #         ,schedulerIndex,normalizationIndex,modelIndex,imageShape
            #         ,fInd,logImageDir,dropout,regression_channels
            #         ,RicianNoiseTransformProb
            #         ,LocalSmoothingTransformProb
            #         ,RandomBiasField_prob
            #         ,RandomAnisotropy_prob
            #         ,Random_GaussNoiseProb
            #         ,optimizerIndex)



            #model = LightningModel.Model(f,args)
            # trainer = pl.Trainer(
            #     #accelerator="cpu", #TODO(remove)
            #     max_epochs=3000,#args.num_epochs,
            #     #gpus=1,
            #     precision=16,#experiment.get_parameter("precision"), 
            #     callbacks=callbacks, #optuna_prune
            #     logger=comet_logger,
            #     accelerator='auto',
            #     devices='auto',       
            #     default_root_dir= "/home/sliceruser/locTemp/lightning_logs",
            #     # auto_scale_batch_size="binsearch",
            #     auto_lr_find=True,
            #     check_val_every_n_epoch=check_eval_every_epoch,
            #     accumulate_grad_batches= 1,
            #     #gradient_clip_val=  0.9 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
            #     log_every_n_steps=5
            #     ,reload_dataloaders_every_n_epochs=1
            #     #strategy='dp'
            # )

            # trainer.tune(model)

            # experiment=trainer.logger.experiment
            # experiment.log_parameter('machine',machine)
            # experiment.log_parameter('base_lr_multi',base_lr_multi)
            # experiment.log_parameter('swa_lrs',swa_lrs)
            # experiment.log_parameter('schedulerIndex',schedulerIndexToLog)
            # experiment.log_parameter('normalizationIndex',normalizationIndex)
            # experiment.log_parameter('modelIndex',modelIndex)

            # experiment.log_parameter('dropout',dropout)
            # experiment.log_parameter('RicianNoiseTransformProb',RicianNoiseTransformProb)
            # experiment.log_parameter('LocalSmoothingTransformProb',LocalSmoothingTransformProb)
            # experiment.log_parameter('RandomBiasField_prob',RandomBiasField_prob)
            # experiment.log_parameter('RandomAnisotropy_prob',RandomAnisotropy_prob)
            # experiment.log_parameter('Random_GaussNoiseProb',Random_GaussNoiseProb)
            # experiment.log_parameter('optimizerIndex',optimizerIndex)
            # experiment.log_parameter('fInd',fInd)





            # trainer.fit(model)
    print("fiiiiiiiiiiiiiiiiiinished")



    res = model.tracking_metrics['best_metric']
    shutil.rmtree(logImageDir) 
    print(f"best_metric {res}")
    return res