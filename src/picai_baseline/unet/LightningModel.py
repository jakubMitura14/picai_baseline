import pytorch_lightning as pl
import argparse
import ast

import numpy as np
import torch

from training_setup.callbacks import (
    optimize_model, resume_or_restart_training, validate_model)
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


class Model(pl.LightningModule):
    def __init__(self
    ,f
    ,args):
        super().__init__()
        self.f = f
        devicee, args = compute_spec_for_run(args=args)
        self.learning_rate=args.base_lr
        self.devicee=devicee
        self.args = args
        model = neural_network_for_run(args=args, device=devicee)
        self.train_gen = []
        self.valid_gen = []
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.base_lr, amsgrad=True)
        model, optimizer, tracking_metrics = resume_or_restart_training(
            model=model, optimizer=optimizer,
            device=devicee, args=args, fold_id=f
        )
        self.model=model
        self.optimizer=optimizer
        self.tracking_metrics=tracking_metrics

    def setup(self, stage=None):
        """
        setting up dataset
        """
        train_gen, valid_gen, class_weights = prepare_datagens(args=self.args, fold_id=self.f)
        self.loss_func = FocalLoss(alpha=class_weights[-1], gamma=self.args.focal_loss_gamma)     
        # integrate data augmentation pipeline from nnU-Net
        # train_gen = apply_augmentations(
        #     dataloader=train_gen,
        #     num_threads=self.args.num_threads,
        #     disable=(not bool(self.args.enable_da))
        # )
        
        # initialize multi-threaded augmenter in background
        # train_gen.restart()
        self.train_gen=train_gen
        self.valid_gen=valid_gen

    def train_dataloader(self):
        return self.train_gen
    
    def val_dataloader(self):
        return self.valid_gen

    def configure_optimizers(self):
        optimizer = self.optimizer
        # optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch_data, batch_idx):        
        epoch=self.current_epoch
        train_loss, step = 0,  0
        inputs = batch_data['data'][:,0,:,:,:,:]
        labels = batch_data['seg'][:,0,:,:,:,:]
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels[:, 0, ...].long())
        train_loss += loss.item()
        self.log('train_loss', loss.item())
        return loss
    def validation_step(self, valid_data, batch_idx):
        valid_images = valid_data['data'][:,0,:,:,:,:]
        valid_labels = valid_data['seg'][:,0,:,:,:,:]                
        valid_images = [valid_images, torch.flip(valid_images, [4]).to(self.device)]
        preds = [
            torch.sigmoid(self.model(x))[:, 1, ...].detach().cpu().numpy()
            for x in valid_images
        ]
        # revert horizontally flipped tta image
        preds[1] = np.flip(preds[1], [3])

        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        # all_valid_preds += [
        #     np.mean([
        #         gaussian_filter(x, sigma=1.5)
        #         for x in preds
        #     ], axis=0)]

        # all_valid_labels += [valid_labels.numpy()[:, 0, ...]]
        return {'valid_label': valid_labels[:, 0, ...], 'validPred' :np.mean([
                                                                                    gaussian_filter(x, sigma=1.5)
                                                                                    for x in preds
                                                                                ], axis=0)  }

    def validation_epoch_end(self, outputs): 
        epoch=self.current_epoch
        all_valid_labels=np.array(([x['valid_label'] for x in outputs]))
        all_valid_preds=np.array(([x['validPred'] for x in outputs]))
        # print(f"all_valid_labels {all_valid_labels}")

        valid_metrics = evaluate(y_det=iter(np.concatenate([x for x in np.array(all_valid_preds)], axis=0)),
                                y_true=iter(np.concatenate([x for x in np.array(all_valid_labels)], axis=0)),
                                y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

        num_pos = int(np.sum([np.max(y) for y in np.concatenate(
            [x for x in np.array(all_valid_labels)], axis=0)]))
        num_neg = int(len(np.concatenate([x for x in
                                        np.array(all_valid_labels)], axis=0)) - num_pos)

        self.tracking_metrics['all_epochs'].append(epoch+1)
        # self.tracking_metrics['all_train_loss'].append(self.tracking_metrics['train_loss'])
        self.tracking_metrics['all_valid_metrics_auroc'].append(valid_metrics.auroc)
        self.tracking_metrics['all_valid_metrics_ap'].append(valid_metrics.AP)
        self.tracking_metrics['all_valid_metrics_ranking'].append(valid_metrics.score)
        
        self.log('valid_metrics_auroc',valid_metrics.auroc  )
        self.log('valid_metrics_AP',valid_metrics.AP  )
        self.log('valid_metrics_ranking',valid_metrics.score  )
    
        # export train-time + validation metrics as .xlsx sheet
        metricsData = pd.DataFrame(list(zip(self.tracking_metrics['all_epochs'],
                                            # self.tracking_metrics['all_train_loss'],
                                            self.tracking_metrics['all_valid_metrics_ranking'],
                                            self.tracking_metrics['all_valid_metrics_auroc'],
                                            self.tracking_metrics['all_valid_metrics_ap'],
                                            self.tracking_metrics['all_valid_metrics_ranking'])),
                                columns=['epoch', 'train_loss', 'valid_auroc', 'valid_ap', 'valid_ranking'])

        # create target folder and save exports sheet
        os.makedirs(self.args.weights_dir, exist_ok=True)

        metricsData.to_excel(self.args.weights_dir + self.args.model_type + '_F' + str(self.f)
                            + '_metrics.xlsx', encoding='utf-8', index=False)
        
        # writer.add_scalar("valid_auroc",   valid_metrics.auroc, epoch+1)
        # writer.add_scalar("valid_ap",      valid_metrics.AP,    epoch+1)
        # writer.add_scalar("valid_ranking", valid_metrics.score, epoch+1)

        print(f"Valid. Performance [Benign or Indolent PCa (n={num_neg}) \
            vs. csPCa (n={num_pos})]:\nRanking Score = {valid_metrics.score:.3f},\
            AP = {valid_metrics.AP:.3f}, AUROC = {valid_metrics.auroc:.3f}", flush=True)

        # store model checkpoint if validation metric improves
        if valid_metrics.score > self.tracking_metrics['best_metric']:
            self.tracking_metrics['best_metric'] = valid_metrics.score
            self.tracking_metrics['best_metric_epoch'] = epoch + 1
            if bool(self.args.export_best_model):
                torch.save({'epoch':                epoch,
                            'model_state_dict':     self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                        self.args.weights_dir + self.args.model_type + '_F' + str(self.f) + ".pt")
                print("Validation Ranking Score Improved! Saving New Best Model", flush=True)


        # experiment=self.experiment=self.logger.experiment
        # if(epoch==0):
