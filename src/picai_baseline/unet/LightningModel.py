import pytorch_lightning as pl
import argparse
import ast

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
    return (monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
        # blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
        blocks_down=(2, 4, 4, 8), blocks_up=(2, 2, 2)
    ),(3,32,256,256),10)

def getSegResNetb(dropout,input_image_size,in_channels,out_channels):
    return (monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
        # blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
        #blocks_down=(2, 4, 4, 8), blocks_up=(2, 2, 2)
    ),(3,32,256,256),14)

def getUneta(args,devicee):
    return (neural_network_for_run(args=args, device=devicee),(3,20,256,256),32)

def getUnetb(args,devicee):
    args.model_features = [ 64, 128, 256, 512, 1024,2048]
    return (neural_network_for_run(args=args, device=devicee),(3,20,256,256),32)

def getVNet(dropout,input_image_size,in_channels,out_channels):
    return (monai.networks.nets.VNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=out_channels,
        dropout_prob=dropout
    ),(4,32,256,256),6)

def chooseModel(args,devicee,index, dropout, input_image_size,in_channels,out_channels  ):
    models=[#getSwinUNETRa(dropout,input_image_size,in_channels,out_channels),
            #getSwinUNETRb(dropout,input_image_size,in_channels,out_channels),
            getSegResNeta(dropout,input_image_size,in_channels,out_channels),
            getSegResNetb(dropout,input_image_size,in_channels,out_channels),
            getUneta(args,devicee),
            getUnetb(args,devicee),
            getVNet(dropout,input_image_size,in_channels,out_channels)
            ]
    return models[index]        

def chooseScheduler(optimizer, schedulerIndex):
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
                 ,torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )                  ]
    return schedulers[schedulerIndex]


def save_heatmap(arr,dir,name,cmapp='gray'):
    path = join(dir,name+'.png')
    arr = np.flip(np.transpose(arr),0)
    plt.imshow(arr , interpolation = 'nearest' , cmap= cmapp)
    plt.title( name)
    plt.savefig(path)
    return path

def log_images(i,experiment,golds,extracteds ,labelNames, directory,epoch):
    gold_arr_loc=golds[i]
    extracted=extracteds[i]
    labelName=labelNames[i]
    print(f"gggg gold_arr_loc {gold_arr_loc.shape} extracted {extracted.shape}")
    maxSlice = max(list(range(0,gold_arr_loc.size(dim=3))),key=lambda ind : torch.sum(gold_arr_loc[:,:,ind]).item() )

    #logging only if it is non zero case
    if np.sum(gold)>0:
        experiment.log_image( save_heatmap(np.add(gold*3,((extracted[:,:,maxSlice]>0).astype('int8'))),directory,f"gold_plus_extracted_{labelName}_{epoch}",'plasma'))
        # experiment.log_image( save_heatmap(gold,directory,f"gold_{curr_studyId}_{epoch}",numLesions[i]))


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
    ,logImageDir):
        super().__init__()
        in_channels=3
        out_channels=2
        devicee, args = compute_spec_for_run(args=args)
        self.devicee=devicee
        self.args = args
        model = neural_network_for_run(args=args, device=devicee)
        self.base_lr_multi = base_lr_multi
        self.learning_rate = learning_rate
        base_lr= learning_rate*base_lr_multi
        print(f"lr {self.learning_rate*self.base_lr_multi}")
        self.base_lr = base_lr
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate*self.base_lr_multi , amsgrad=True)
        self.optimizer =optimizer
        # optimizer = torch.optim.NAdam(params=model.parameters(),momentum_decay=0.004, lr=base_lr)
        self.scheduler = chooseScheduler(optimizer,schedulerIndex )    
        
        self.f = f
        self.learning_rate=learning_rate#args.base_lr
        self.modelIndex=modelIndex
        self.train_gen = []
        self.valid_gen = []
        self.normalizationIndex=normalizationIndex
        dropout=0.0
        #optimizer = torch.optim.Adam(params=model.parameters(), lr=args.base_lr, amsgrad=True)
        # model, optimizer, tracking_metrics = resume_or_restart_training(
        #     model=model, optimizer=optimizer,
        #     device=devicee, args=args, fold_id=f
        # )
        tracking_metrics=resume_or_restart_training_tracking(args, fInd)
        model,expectedShape,newBatchSize=chooseModel(args,devicee,modelIndex, dropout, imageShape,in_channels,out_channels  )
        args.batch_size= newBatchSize
        self.model=model
        self.expectedShape =expectedShape
        self.tracking_metrics=tracking_metrics
        print(f"argssssssss pl {args}")
        self.logImageDir=logImageDir#tempfile.mkdtemp()


    def setup(self, stage=None):
        """
        setting up dataset
        """
        train_gen, valid_gen, test_gen, class_weights = prepare_datagens(args=self.args, fold_id=self.f,normalizationIndex=self.normalizationIndex,expectedShape=self.expectedShape)
        self.loss_func = FocalLoss(alpha=class_weights[-1], gamma=self.args.focal_loss_gamma)     
        #self.loss_func = monai.losses.FocalLoss(include_background=False, to_onehot_y=True,gamma=self.args.focal_loss_gamma )
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
        experiment=self.logger.experiment
        experiment.log_parameter('lr',self.base_lr)

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
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "train_loss",
                "frequency": 1
            }}
        
        

    def training_step(self, batch_data, batch_idx):
        epoch=self.current_epoch
        # train_loss, step = 0,  0
        
        inputs = batch_data['data'][:,0,:,:,:,:]
        labels = batch_data['seg'][:,0,:,:,:,:]
        
        #inputs = batch_data['data']
        #labels = batch_data['seg']
        # print(f"uuuuu  inputs {type(inputs)} labels {type(labels)}  ")
        outputs = self.model(inputs)
        # print(f"ssshhh {batch_data['data'].shape} {type(batch_data['data'])} label {batch_data['seg'].shape} {type(batch_data['seg'])}  outputs {outputs.shape} {type(outputs)} ")

    
        # loss = self.loss_func(outputs, labels)
        loss = self.loss_func(outputs, labels.long())
        # train_loss += loss.item()
        self.log('train_loss', loss.item())
        # print(f" sssssssssss loss {type(loss)}  ")

        # return torch.Tensor([loss]).to(self.device)

        return loss

    def _shared_eval_step(self, valid_data, batch_idx):
        # print(f"ssshhh {valid_data['data'].shape}  label {valid_data['seg'].shape}")
        valid_images = valid_data['data'][0,:,:,:,:]
        valid_labels = valid_data['seg'][0,:,:,:,:]                
        label_name = valid_data['seg_name'][0]             
        valid_images = [valid_images, torch.flip(valid_images, [4]).to(self.device)]
        preds = [
            torch.sigmoid(self.model(x))[:, 1, ...].detach().cpu().numpy().astype(np.float32)
            for x in valid_images
        ]
        preds[1] = np.flip(preds[1], [3])

        return (valid_labels[:, 0, ...], np.mean([
                                                        gaussian_filter(x, sigma=1.5)
                                                        for x in preds
                                                    ], axis=0), label_name )



    def validation_step(self, batch, batch_idx, dataloader_idx):
        valid_label, preds,label_name = self._shared_eval_step(batch, batch_idx)
        # print(f"in validation dataloader_idx {dataloader_idx} ")
        # revert horizontally flipped tta image
        return {'valid_label': valid_label, 'val_preds' : preds ,'dataloader_idx' :dataloader_idx, 'label_name':label_name}


    def _eval_epoch_end(self, outputs,labelKey,predsKey, dataloader_idxx):
        epoch=self.current_epoch
        # print(f"outputs {outputs}")
        outputs = outputs[dataloader_idxx]#list(filter( lambda entry : entry['dataloader_idx']==dataloader_idxx,outputs))
        all_valid_labels=np.array(([x[labelKey].cpu().detach().numpy() for x in outputs]))
        all_valid_preds=np.array(([x[predsKey] for x in outputs]))
        all_label_name=np.array(([x['label_name'] for x in outputs]))
        for i in range(0,len(all_valid_preds)):
            log_images(i,self.logger.experiment,all_valid_labels,all_valid_preds ,all_label_name, self.logImageDir,self.current_epoch)


        #print(f" all_valid_labels {all_valid_labels[0].shape}  all_valid_preds {all_valid_preds[0].shape} ")
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



