#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import ast
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
from functools import partial
from glob import glob
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from typing import (Callable, Dict, Hashable, Iterable, List, Optional,
                    Sequence, Sized, Tuple, Union)


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
from monai.transforms import (AddChanneld, AsDiscrete, Compose,
                              CropForegroundd, EnsureType, EnsureTyped,
                              LoadImaged, Orientationd, RandCropByPosNegLabeld,
                              ScaleIntensityRanged, Spacingd)
from monai.utils import alias, deprecated_arg, export, set_determinism
from optuna.integration import PyTorchLightningPruningCallback
from picai_eval import evaluate
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from report_guided_annotation import extract_lesion_candidates
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from torch.nn.intrinsic.qat import ConvBnReLU3d
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader, random_split
from torchmetrics import Precision
from torchmetrics.functional import precision_recall
from tqdm import tqdm

import LightningModel
from training_setup.callbacks import (optimize_model,
                                      resume_or_restart_training,
                                      validate_model)
from training_setup.compute_spec import compute_spec_for_run
from training_setup.data_generator import prepare_datagens
from training_setup.default_hyperparam import get_default_hyperparams
from training_setup.loss_functions.focal import FocalLoss
from training_setup.neural_network_selector import neural_network_for_run

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

import functools
import glob
import math
import multiprocessing
import multiprocessing as mp
import os
import os.path
import shutil
import tempfile
import time
from datetime import datetime
from glob import glob
from pathlib import Path


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
import torchio as tio
from monai.data import (CacheDataset, Dataset, PersistentDataset,
                        decollate_batch, list_data_collate)
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import UNet
from monai.utils import set_determinism
from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.eval import evaluate_case
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split

monai.utils.set_determinism()
from functools import partial

# import preprocessing.transformsForMain
# import preprocessing.ManageMetadata
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
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


import hpTrain

def main():
    # command line arguments for hyperparameters and I/O paths
    parser = argparse.ArgumentParser(description='Command Line Arguments for Training Script')

    # data I/0 + experimental setup
    parser.add_argument('--max_threads', type=int, default=12,
                        help="Max threads/workers for data loaders")
    parser.add_argument('--validate_n_epochs', type=int, default=10,               
                        help="Trigger validation every N epochs")
    parser.add_argument('--validate_min_epoch', type=int, default=50,               
                        help="Trigger validation after minimum N epochs")
    parser.add_argument('--export_best_model', type=int, default=1,                
                        help="Export model checkpoints")
    parser.add_argument('--resume_training', type=str, default=1,                
                        help="Resume training model, if checkpoint exists")
    parser.add_argument('--weights_dir', type=str, required=True,            
                        help="Path to export model checkpoints")
    parser.add_argument('--overviews_dir', type=str, required=True,            
                        help="Base path to training/validation data sheets")
    parser.add_argument('--folds', type=int, nargs='+', required=True, 
                        help="Folds selected for training/validation run")

    # training hyperparameters
    parser.add_argument('--image_shape', type=int, nargs="+", default=[20, 256, 256],   
                        help="Input image shape (z, y, x)")
    parser.add_argument('--num_channels', type=int, default=3,                
                        help="Number of input channels/sequences")
    parser.add_argument('--num_classes', type=int, default=2,                
                        help="Number of classes at train-time")
    parser.add_argument('--num_epochs', type=int, default=100,              
                        help="Number of training epochs")
    parser.add_argument('--base_lr', type=float, default=0.001,            
                        help="Learning rate")
    parser.add_argument('--focal_loss_gamma', type=float, default=1.0,              
                        help="Focal Loss gamma value")
    parser.add_argument('--enable_da', type=int, default=1,                
                        help="Enable data augmentation")

    # neural network-specific hyperparameters
    parser.add_argument('--model_type', type=str, default='unet',                                                    
                        help="Neural network: architectures")
    parser.add_argument('--model_strides', type=str, default='[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]', 
                        help="Neural network: convolutional strides (as string representation)")
    parser.add_argument('--model_features', type=str, default='[32, 64, 128, 256, 512, 1024]',                           
                        help="Neural network: number of encoder channels (as string representation)")
    parser.add_argument('--batch_size', type=int, default=32,                                                         
                        help="Mini-batch size")
    parser.add_argument('--use_def_model_hp', type=int, default=1,                                                         
                        help="Use default set of model-specific hyperparameters")

    args = parser.parse_args()
    
    project_name= "pic_mono_9"
    #experiment_name="baseline_pl"
    # args.batch_size=32
    # imageShape=(256, 256,20)
    imageShape=(256, 256,20)
    args.weights_dir=join(args.weights_dir,project_name )
    args.model_strides = ast.literal_eval(args.model_strides)
    args.model_features = ast.literal_eval(args.model_features)
    
    # retrieve default set of hyperparam (architecture, batch size) for given neural network
    # if bool(args.use_def_model_hp):
    #     args = get_default_hyperparams(args)
    
    study = optuna.create_study(
            study_name=project_name
            ,sampler=optuna.samplers.NSGAIISampler()    
            ,pruner=optuna.pruners.HyperbandPruner()
            ,storage=f"mysql://root:jm@34.91.215.109:3306/{project_name}"
            ,load_if_exists=True
            ,direction="maximize"
            )

    def objective(trial: optuna.trial.Trial) -> float:
        return hpTrain.mainTrain(project_name,args,trial,imageShape)



    study.optimize(objective, n_trials=400)

if __name__ == '__main__':
    main()

