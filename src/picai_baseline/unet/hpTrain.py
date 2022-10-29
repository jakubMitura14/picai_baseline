
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
import gdown
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


def mainTrain(project_name,experiment_name,args,trial: optuna.trial.Trial) -> float:

    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name=project_name, # Optional
        experiment_name=experiment_name # Optional
    )
    toMonitor="valid_ranking"
    # checkpoint_callback = ModelCheckpoint(dirpath= checkPointPath,mode='max', save_top_k=1, monitor=toMonitor)
    # stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=trial.suggest_float("swa_lrs", 1e-6, 1e-4))
    # optuna_prune=PyTorchLightningPruningCallback(trial, monitor=toMonitor)     
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor=toMonitor,
        patience=7,
        mode="max",
        #divergence_threshold=(-0.1)
    )
    # check_eval_every_epoch=40
    check_eval_every_epoch=1

    # for each fold
    # for f in args.folds:
    f=args.folds[0]

    model = LightningModel.Model(f,args)
    trainer = pl.Trainer(
        #accelerator="cpu", #TODO(remove)
        max_epochs=1,#args.num_epochs,
        #gpus=1,
        #precision=experiment.get_parameter("precision"), 
        callbacks=[early_stopping ], #optuna_prune
        logger=comet_logger,
        accelerator='auto',
        devices='auto',       
        default_root_dir= "/home/sliceruser/locTemp/lightning_logs",
        # auto_scale_batch_size="binsearch",
        auto_lr_find=True,
        check_val_every_n_epoch=check_eval_every_epoch,
        #accumulate_grad_batches= 1,
        #gradient_clip_val=  0.9 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
        log_every_n_steps=5
        ,reload_dataloaders_every_n_epochs=1
        #strategy='dp'
    )
    trainer.fit(model)
    res = model.tracking_metrics['best_metric']
    print(f"mmmmmmmmmmmmmmmm {res}")