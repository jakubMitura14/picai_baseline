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
import modelsToChoose 


def chooseModel(args,devicee,index, dropout, input_image_size,in_channels,out_channels  ):
    models=[modelsToChoose.getUneta(args,devicee),
            modelsToChoose.getSegResNeta(dropout, input_image_size,in_channels,out_channels),
            modelsToChoose.getSegResNetb(dropout, input_image_size,in_channels,out_channels),
            modelsToChoose.getVneta(dropout, input_image_size,in_channels,out_channels),
            modelsToChoose.getVnetb(dropout, input_image_size,in_channels,out_channels),
            #modelsToChoose.getVnetc(dropout, input_image_size,in_channels,out_channels),
            modelsToChoose.getUnetWithTransformerA(dropout,input_image_size,in_channels,out_channels,args,devicee),
            modelsToChoose.getUnetWithTransformerB(dropout,input_image_size,in_channels,out_channels,args,devicee)]
    
    return models[index]        

def chooseScheduler(optimizer, schedulerIndex):
    schedulers = [torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=300)
                  ,torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
                 ,torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )                  ]
    
    return schedulers[schedulerIndex]


def save_heatmap(arr,dir,name,cmapp='gray'):
    path = join(dir,name+'.png')
    #arr = np.flip(np.transpose(arr),0)
    plt.imshow(arr , interpolation = 'nearest' , cmap= cmapp)
    plt.title( name)
    plt.savefig(path)
    return path


def log_images(experiment,golds,extracteds ,labelNames, directory,epoch,dataloaderIdx):
    valTr='val'
    if(dataloaderIdx==1):
        valTr='train'
    for batchInd in range(0,golds.shape[0]):
        if(batchInd<10):
            gold_arr_loc=golds[batchInd,:,:,:]
            extracted=extract_lesion_candidates(extracteds[batchInd,:,:,:])[0]
            labelName=labelNames[batchInd]
            # print(f"gggg gold_arr_loc {gold_arr_loc.shape} {type(gold_arr_loc)} extracted {extracted.shape} {type(extracted)} t2w {t2ws[i].shape}  ")
            maxSlice = max(list(range(0,gold_arr_loc.shape[0])),key=lambda ind : np.sum(gold_arr_loc[ind,:,:]) )
            # t2w = t2ws[i][0,maxSlice,:,:]
            # t2wMax= np.max(t2w.flatten())
            # print(f"suuuum {np.sum(extracted)}")
            #logging only if it is non zero case
            if np.sum(gold_arr_loc)>0:
                # experiment.log_image( save_heatmap(np.add(gold_arr_loc[maxSlice,:,:].astype('float')*2,((extracted[maxSlice,:,:]).astype('float'))),directory,f"{valTr}_{labelName}_{epoch}",'plasma'))
                experiment.log_image( save_heatmap(np.add(gold_arr_loc[maxSlice,:,:]*2,((extracted[maxSlice,:,:]>0).astype('int8'))),directory,f"gold_plus_extracted_{labelName}_{epoch}",'plasma'))
                # experiment.log_image( save_heatmap(np.add(t2w.astype('float'),(gold_arr_loc[maxSlice,:,:]*(t2wMax)).astype('float')),directory,f"gold_plus_t2w_{labelName}_{epoch}"))


class Model(pl.LightningModule):
    def __init__(self
    ,f
    ,args
    ,learning_rate
    ,base_lr_multi
    ,schedulerIndex
    ,normalizationIndex
    ,modelIndex
    ,imageShape
    ,fInd
    ,logImageDir
    ,dropout
    ,regression_channels
    ,RicianNoiseTransformProb
    ,LocalSmoothingTransformProb
    ,RandomBiasField_prob
    ,RandomAnisotropy_prob
    ,Random_GaussNoiseProb
    ):
        super().__init__()
        self.save_hyperparameters()
        in_channels=3
        out_channels=2
        self.f = f
        devicee, args = compute_spec_for_run(args=args)
        self.learning_rate=args.base_lr
        self.normalizationIndex=normalizationIndex
        self.logImageDir=logImageDir
        self.devicee=devicee
        self.args = args
        #model = neural_network_for_run(args=args, device=devicee)
        self.train_gen = []
        self.valid_gen = []
        self.RicianNoiseTransformProb=RicianNoiseTransformProb
        self.LocalSmoothingTransformProb=LocalSmoothingTransformProb
        self.RandomBiasField_prob=RandomBiasField_prob
        self.RandomAnisotropy_prob=RandomAnisotropy_prob
        self.Random_GaussNoiseProb=Random_GaussNoiseProb



        # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.base_lr, amsgrad=True)
        # model, optimizer, tracking_metrics = resume_or_restart_training(
        #     model=model, optimizer=optimizer,
        #     device=devicee, args=args, fold_id=f
        # )
        tracking_metrics=resume_or_restart_training_tracking(args, fInd)
        model,expectedShape,newBatchSize=chooseModel(args,devicee,modelIndex, dropout, imageShape,in_channels,out_channels  )
        # args.batch_size= newBatchSize
        #self.expectedShape=expectedShape

        #self.modelRegression = UNetToRegresion(2,regression_channels,model)
        self.regressionMetric_val=BinaryF1Score()
        self.regressionMetric_train=BinaryF1Score()
        self.regLoss = nn.BCEWithLogitsLoss()
        #self.expectedShape=expectedShape= (3,20,256,256)
        # models=[getUneta(args,devicee),getUnetb(args,devicee)]
        #model,expectedShape,newBatchSize=getUneta(args,devicee) #models[0]
        self.expectedShape=expectedShape
        args.batch_size= newBatchSize
        self.model=model
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.base_lr*base_lr_multi, amsgrad=True)
        # self.lr_scheduler = chooseScheduler(optimizer,schedulerIndex )
        self.schedulerIndex=schedulerIndex
        self.optimizer=optimizer
        self.tracking_metrics=tracking_metrics

    def setup(self, stage=None):
        """
        setting up dataset
        """
        train_gen, valid_gen, test_gen, class_weights,df = prepare_datagens(args=self.args, fold_id=self.f,normalizationIndex=self.normalizationIndex
            ,expectedShape=self.expectedShape,RicianNoiseTransformProb=self.RicianNoiseTransformProb
            , LocalSmoothingTransformProb=self.LocalSmoothingTransformProb ,RandomBiasField_prob=self.RandomBiasField_prob
            ,RandomAnisotropy_prob=self.RandomAnisotropy_prob, Random_GaussNoiseProb=self.Random_GaussNoiseProb  )
        self.df = df
        # self.loss_func = FocalLoss(alpha=class_weights[-1], gamma=self.args.focal_loss_gamma)     
        self.loss_func = monai.losses.FocalLoss(include_background=False, to_onehot_y=True,gamma=self.args.focal_loss_gamma )
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
        self.test_gen=test_gen
    def train_dataloader(self):
        return self.train_gen
    
    def val_dataloader(self):
        return [self.valid_gen,self.test_gen]

    # def test_dataloader(self):
    #     return self.test_gen

    def configure_optimizers(self):
        optimizer = self.optimizer
        # optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_scheduler = chooseScheduler(optimizer,self.schedulerIndex )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
                "frequency": 1
            }}
        
        

    def training_step(self, batch_data, batch_idx):        
        epoch=self.current_epoch
        # train_loss, step = 0,  0
        inputs = batch_data['data'][:,0,:,:,:,:]
        labels = batch_data['seg'][:,0,:,:,:,:]
        isCa = batch_data['isCa']
        # print(f"uuuuu  inputs {type(inputs)} labels {type(labels)}  ")
        # outputs = self.modelRegression(inputs)
        segmMap = self.model(inputs)
        lossSegm = self.loss_func(segmMap, labels)
        self.log('train_loss', lossSegm.item())
        return lossSegm
        # if(epoch%2==0):
        #     lossSegm = self.loss_func(segmMap, labels)
        #     self.log('train_loss', lossSegm.item())
        #     return lossSegm

        # lossRegr=self.regLoss(reg_hat.flatten().float(),torch.Tensor(isCa).to(self.device).flatten().float() )
        # # train_loss += loss.item()
        # self.log('train_loss', lossRegr.item())
        # # print(f" sssssssssss loss {type(loss)}  ")

        # # return torch.Tensor([loss]).to(self.device)
        # return lossRegr

    def _shared_eval_step(self, valid_data, batch_idx,dataloader_idx):
        valid_images = valid_data['data'][:,0,:,:,:,:]
        valid_labels = valid_data['seg'][:,0,:,:,:,:]
        #segmMap = self.model(valid_images)                
        valid_images = [valid_images, torch.flip(valid_images, [4]).to(self.device)]
        isCa = valid_data['isCa']
        label_name = valid_data['seg_name']
        
        # if(dataloader_idx==0):
        #     self.regressionMetric_val(torch.round(reg_hat.flatten().float()),torch.Tensor(isCa).to(self.device).float())
        # if(dataloader_idx==1):
        #     self.regressionMetric_train(torch.round(reg_hat.flatten().float()),torch.Tensor(isCa).to(self.device).float())        
        # print(f"wwwwwwwwwwww {self.model(valid_images).shape}")
        preds = [
            torch.sigmoid(self.model(x))[:, 1, ...].detach().cpu().numpy()
            for x in valid_images
        ]
        preds[1] = np.flip(preds[1], [3])
        res= (valid_labels[:, 0, ...]
                , np.mean([ gaussian_filter(x, sigma=1.5)for x in preds], axis=0), )
        if(batch_idx<4):
            log_images(self.logger.experiment,res[0],res[1] ,label_name, self.logImageDir,self.current_epoch,dataloader_idx)
        
        return res


    # def test_step(self, batch, batch_idx):
    #     valid_labels, preds = self._shared_eval_step(batch, batch_idx)
    #     # revert horizontally flipped tta image
    #     return {'train_label': valid_labels[:, 0, ...], 'trainPred' :np.mean([
    #                                                     gaussian_filter(x, sigma=1.5)
    #                                                     for x in preds
    #                                                 ], axis=0)  }


    def validation_step(self, batch, batch_idx, dataloader_idx):
        valid_label, preds = self._shared_eval_step(batch, batch_idx,dataloader_idx)
        # print(f"in validation dataloader_idx {dataloader_idx} ")
        # revert horizontally flipped tta image
        return {'valid_label': valid_label, 'val_preds' : preds ,'dataloader_idx' :dataloader_idx}


    def _eval_epoch_end(self, outputs,labelKey,predsKey, dataloader_idxx):
        epoch=self.current_epoch
        # print(f"outputs {outputs}")
        outputs = outputs[dataloader_idxx]#list(filter( lambda entry : entry['dataloader_idx']==dataloader_idxx,outputs))
        all_valid_labels=np.array(([x[labelKey].cpu().detach().numpy() for x in outputs]))
        all_valid_preds=np.array(([x[predsKey] for x in outputs]))
        # all_valid_labels=np.array(([x['valid_label'].cpu().detach().numpy() for x in outputs]))
        # all_valid_preds=np.array(([x['val_preds']for x in outputs]))

        # all_train_labels=np.array(([x['train_label'].cpu().detach().numpy() for x in outputs]))
        # all_train_preds=np.array(([x['tain_preds']for x in outputs]))
        # print(f"all_valid_labels {all_valid_labels}")

        valid_metrics = evaluate(y_det=iter(np.concatenate([x for x in np.array(all_valid_preds)], axis=0)),
                                y_true=iter(np.concatenate([x for x in np.array(all_valid_labels)], axis=0)),
                                y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

        num_pos = int(np.sum([np.max(y) for y in np.concatenate(
            [x for x in np.array(all_valid_labels)], axis=0)]))
        num_neg = int(len(np.concatenate([x for x in
                                        np.array(all_valid_labels)], axis=0)) - num_pos)
        return (epoch,valid_metrics,num_pos,num_neg  )

    # def test_epoch_end(self, outputs): 


    

    def validation_epoch_end(self, outputs): 
        epoch,valid_metrics,num_pos,num_neg = self._eval_epoch_end( outputs,'valid_label','val_preds',0 )
        epoch,valid_metrics_train,num_pos_train,num_neg_train = self._eval_epoch_end( outputs,'valid_label','val_preds' ,1)

        self.tracking_metrics['all_epochs'].append(epoch+1)
        # self.tracking_metrics['all_train_loss'].append(self.tracking_metrics['train_loss'])
        self.tracking_metrics['all_valid_metrics_auroc'].append(valid_metrics.auroc)
        self.tracking_metrics['all_valid_metrics_ap'].append(valid_metrics.AP)
        self.tracking_metrics['all_valid_metrics_ranking'].append(valid_metrics.score)
        
        self.log('valid_auroc',valid_metrics.auroc  )
        self.log('valid_AP',valid_metrics.AP  )
        self.log('valid_ranking',valid_metrics.score  )

        self.log('train_auroc',valid_metrics_train.auroc  )
        self.log('train_AP',valid_metrics_train.AP  )
        self.log('train_ranking',valid_metrics_train.score  )    
        
        
        # regressionMetric_val=self.regressionMetric_val.compute()
        # self.regressionMetric_val.reset()
        # self.log('val_F1', regressionMetric_val)
        
        # regressionMetric_train=self.regressionMetric_train.compute()
        # self.regressionMetric_train.reset()
        # self.log('train_F1', regressionMetric_train)

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
                # torch.save({'epoch':                epoch,
                #             'model_state_dict':     self.model.state_dict(),
                #             'optimizer_state_dict': self.optimizer.state_dict()},
                #         self.args.weights_dir + self.args.model_type + '_F' + str(self.f) + ".pt")
                print("Validation Ranking Score Improved! Saving New Best Model", flush=True)


        experiment=self.experiment=self.logger.experiment
        if(epoch==0):
            experiment.log_parameter('machine',os.environ['machine'])
