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

import json
import numpy as np
import monai
import torch
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import DataLoader
from monai.transforms import Compose, EnsureType
from functools import partial
from .image_reader import SimpleITKDataset
from .augmentations import nnUNet_DA
from .adaptTransforms import loadTrainTransform
from .adaptTransforms import loadValTransform
from monai.data import (CacheDataset, SmartCacheDataset,Dataset, PersistentDataset,
                        decollate_batch, list_data_collate)
from torch.utils.data import DataLoader, random_split
from monai.data import (CacheDataset, Dataset, PersistentDataset,
                        decollate_batch, list_data_collate)
import os
from intensity_normalization.normalize.nyul import NyulNormalize

def default_collate(batch):
    """collate multiple samples into batches, if needed"""

    if isinstance(batch[0], np.ndarray):
        return np.vstack(batch)
    elif isinstance(batch[0], (int, np.int64)):
        return np.array(batch).astype(np.int32)
    elif isinstance(batch[0], (float, np.float32)):
        return np.array(batch).astype(np.float32)
    elif isinstance(batch[0], (np.float64,)):
        return np.array(batch).astype(np.float64)
    elif isinstance(batch[0], (dict, OrderedDict)):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], torch.Tensor):
        return torch.vstack(batch)
    else:
        raise TypeError('unknown type for batch:', type(batch))


class DataLoaderFromDataset(DataLoader):
    """Create dataloader from given dataset"""

    def __init__(self, data, batch_size, num_threads, seed_for_shuffle=1, collate_fn=default_collate,
                 return_incomplete=False, shuffle=True, infinite=False):
        super(DataLoaderFromDataset, self).__init__(data, batch_size, num_threads, seed_for_shuffle,
                                                    shuffle=shuffle,
                                                    infinite=infinite)
        self.collate_fn = collate_fn
        self.indices = np.arange(len(data))
        self.dataLen=len(data)

    def generate_train_batch(self):

        # randomly select N samples (N = batch size)
        indices = self.get_indices()

        # create dictionary per sample
        batch = [{'data': self._data[i][0].numpy(),
                  'seg': self._data[i][1].numpy()} for i in indices]

        return self.collate_fn(batch)

    def __len__(self):
        return self.dataLen

def getPatientDict(index, image_files,seg_files):
    subject= {#"chan3_col_name": str(row[chan3_col_name])
        "t2w": str(image_files[index][0])     
        ,"adc": str(image_files[index][1])        
        ,"hbv": str(image_files[index][2]) 
        
    #    , "isAnythingInAnnotated":int(row['isAnythingInAnnotated'])
    #     , "study_id":str(row['study_id'])
    #     , "patient_id":str(row['patient_id'])
    #     , "num_lesions_to_retain":int(row['num_lesions_to_retain_bin'])

        , "seg":str(seg_files[index])
        
        
        }
    return subject




def prepare_datagens(args, fold_id,normalizationIndex):
    """Load data sheets --> Create datasets --> Create data loaders"""

    # load datasheets
    with open(args.overviews_dir+'PI-CAI_train-fold-'+str(fold_id)+'.json') as fp:
        train_json = json.load(fp)
    with open(args.overviews_dir+'PI-CAI_val-fold-'+str(fold_id)+'.json') as fp:
        valid_json = json.load(fp)

    # load paths to images and labels
    train_data = [np.array(train_json['image_paths']), np.array(train_json['label_paths'])]
    valid_data = [np.array(valid_json['image_paths']), np.array(valid_json['label_paths'])]



    # use case-level class balance to deduce required train-time class weights
    class_ratio_t = [int(np.sum(train_json['case_label'])), int(len(train_data[0])-np.sum(train_json['case_label']))]
    class_ratio_v = [int(np.sum(valid_json['case_label'])), int(len(valid_data[0])-np.sum(valid_json['case_label']))]
    class_weights = (class_ratio_t / np.sum(class_ratio_t))

    # log dataset definition
    print('Dataset Definition:', "-"*80)
    print(f'Fold Number: {fold_id}')
    print('Data Classes:', list(np.unique(train_json['case_label'])))
    print(f'Train-Time Class Weights: {class_weights}')
    print(f'Training Samples [-:{class_ratio_t[1]};+:{class_ratio_t[0]}]: {len(train_data[1])}')
    print(f'Validation Samples [-:{class_ratio_v[1]};+:{class_ratio_v[0]}]: {len(valid_data[1])}')

    # dummy dataloader for sanity check
    pretx = [EnsureType()]
    # check_ds = SimpleITKDataset(image_files=train_data[0][:args.batch_size*2],
    #                             seg_files=train_data[1][:args.batch_size*2],
    #                             transform=Compose(pretx),
    #                             seg_transform=Compose(pretx))
    # check_loader = DataLoaderFromDataset(check_ds, batch_size=args.batch_size, num_threads=args.num_threads)
    # data_pair = monai.utils.misc.first(check_loader)
    # print('DataLoader - Image Shape: ', data_pair['data'].shape)
    # print('DataLoader - Label Shape: ', data_pair['seg'].shape)
    # print("-"*100)
    # assert args.image_shape == list(data_pair['data'].shape[2:])
    # assert args.num_channels == data_pair['data'].shape[1]
    # assert args.num_classes == len(np.unique(train_json['case_label']))



    normalizationsDir="/home/sliceruser/locTemp/picai_baseline/src/picai_baseline/standarizationModels"
    normalizerDict = {}
    for key in ["t2w","adc","hbv"]:
        pathNormalizer = os.path.join(normalizationsDir,key+".npy")
        nyul_normalizer = NyulNormalize()
        nyul_normalizer.load_standard_histogram(pathNormalizer)  
        normalizerDict[key]=nyul_normalizer


    subjects_train = list(map(partial(getPatientDict,image_files=train_data[0], seg_files=train_data[1]) , range(0,len(train_data[0])) ))
    subjects_val = list(map(partial(getPatientDict,image_files=valid_data[0], seg_files=valid_data[1]) , range(0,len(valid_data[0])) ))



    transfTrain=loadTrainTransform(Compose(pretx),Compose(pretx),nnUNet_DA.get_augmentations(),normalizationIndex,normalizerDict)
       
    transfVal=loadValTransform(Compose(pretx),Compose(pretx,normalizationIndex,normalizerDict))

    transfTrain=Compose(transfTrain,monai.transforms.ToTensord(keys=["data","seg"])  )
    transfVal=Compose(transfVal,monai.transforms.ToTensord(keys=["data","seg"])  )

    # print(f"train_data {train_data[0]}")
    # train_ds=Dataset(data=subjects_train, transform= transfTrain)
    # valid_ds=Dataset(data=subjects_val, transform= transfVal)
    # test_ds=Dataset(data=subjects_train[0:len(subjects_val)], transform= transfVal)

    train_ds=SmartCacheDataset(data=subjects_train, transform=transfTrain  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())
    valid_ds=Dataset(data=subjects_val, transform=transfVal )
    test_ds=Dataset(data=subjects_train[0:len(subjects_val)], transform=transfVal)
    batchh= args.batch_size

    print(f"aaaaaaaaaaaaaa batchh {batchh}")
    train_ldr=DataLoader(train_ds,batch_size=batchh, num_workers=args.num_threads, shuffle=True,collate_fn=list_data_collate )
    valid_ldr=DataLoader(valid_ds,batch_size=batchh, num_workers=args.num_threads,shuffle=False,collate_fn=list_data_collate)
    test_gen=DataLoader(test_ds,batch_size=batchh, num_workers=args.num_threads,shuffle=False,collate_fn=list_data_collate)





    return train_ldr, valid_ldr, test_gen, class_weights.astype(np.float32)


    # actual dataloaders used at train-time
    # train_ds = SimpleITKDataset(image_files=train_data[0], seg_files=train_data[1],
    #                             transform=Compose(pretx),  seg_transform=Compose(pretx))
    # valid_ds = SimpleITKDataset(image_files=valid_data[0], seg_files=valid_data[1],
    #                             transform=Compose(pretx),  seg_transform=Compose(pretx))
    # train_ldr = DataLoaderFromDataset(train_ds, 
    #     batch_size=args.batch_size, num_threads=args.num_threads, infinite=True, shuffle=True)
    # valid_ldr = DataLoaderFromDataset(valid_ds, 
    #     batch_size=args.batch_size, num_threads=1, infinite=False, shuffle=False)