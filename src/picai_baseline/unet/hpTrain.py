
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


def mainTrain(project_name,args,trial: optuna.trial.Trial,imageShape) -> float:
    machine = os.environ['machine']
    expId=trial.number
    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name=project_name, # Optional
        experiment_name=f"{machine}_{str(expId)}" # Optional
        #experiment_name=experiment_name # Optional
    )
    toMonitor="valid_ranking"
        # optuna_prune=PyTorchLightningPruningCallback(trial, monitor=toMonitor)     
    
    swa_lrs=trial.suggest_float("swa_lrs", 1e-5,0.5) #trial.suggest_float("swa_lrs", 1e-6, 1e-4)
    base_lr_multi =trial.suggest_float("base_lr_multi", 0.0001, 1.0)
    schedulerIndex=1#trial.suggest_int("scheduler_int", 0, 2)
    # modelIndex=trial.suggest_int("modelIndex", 0, 3)
    modelIndex=3
    normalizationIndex=0#trial.suggest_int("normalizationIndex", 0, 1)


    stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs= swa_lrs )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor=toMonitor,
        patience=7,
        mode="max",
        #divergence_threshold=(-0.1)
    )
    # callbacks=[early_stopping], #optuna_prune
    # if(swa_lrs>0.0):
    #     callbacks=[early_stopping,stochasticAveraging], #optuna_prune
    # check_eval_every_epoch=40
    check_eval_every_epoch=1

    # for each fold
    # for f in args.folds:
    f=args.folds[0]
    fInd=0

    checkPointPath=f"/home/sliceruser/locTemp/checkPoints/{project_name}/{expId}/{fInd}"
    checkpoint_callback = ModelCheckpoint(dirpath= checkPointPath,mode='max', save_top_k=1, monitor=toMonitor)
    schedulerIndexToLog= schedulerIndex
    callbacks=[early_stopping,stochasticAveraging,checkpoint_callback]
    # if(schedulerIndex==2):
    #     schedulerIndex=1
    #     callbacks=[early_stopping,checkpoint_callback]

    model = LightningModel.Model(f,args,base_lr_multi,schedulerIndex,normalizationIndex,modelIndex,imageShape,fInd)
    trainer = pl.Trainer(
        #accelerator="cpu", #TODO(remove)
        max_epochs=1,#args.num_epochs,
        #gpus=1,
        #precision=16,#experiment.get_parameter("precision"), 
        callbacks=callbacks, #optuna_prune
        logger=comet_logger,
        accelerator='auto',
        devices='auto',       
        default_root_dir= "/home/sliceruser/locTemp/lightning_logs",
        # auto_scale_batch_size="binsearch",
        #auto_lr_find=True,
        check_val_every_n_epoch=check_eval_every_epoch,
        accumulate_grad_batches= 1,
        #gradient_clip_val=  0.9 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
        log_every_n_steps=5
        ,reload_dataloaders_every_n_epochs=1
        #strategy='dp'
    )

    
    experiment=trainer.logger.experiment
    experiment.log_parameter('machine',machine)
    experiment.log_parameter('base_lr_multi',base_lr_multi)
    experiment.log_parameter('swa_lrs',swa_lrs)
    experiment.log_parameter('schedulerIndex',schedulerIndexToLog)
    experiment.log_parameter('normalizationIndex',normalizationIndex)
    experiment.log_parameter('modelIndex',modelIndex)



    trainer.fit(model)




    res = model.tracking_metrics['best_metric']
    print(f"best_metric {res}")
    return res