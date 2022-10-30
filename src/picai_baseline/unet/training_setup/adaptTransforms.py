import monai
from monai.transforms import (
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    AddChanneld,
    Spacingd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Resize,
    Resized,
    RandSpatialCropd,
        AsDiscrete,
    AsDiscreted,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    SelectItemsd,
    Invertd,
    DivisiblePadd,
    SpatialPadd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandRicianNoised,
    RandFlipd,
    RandAffined,
    ConcatItemsd,
    RandCoarseDropoutd,
    AsDiscreted,
    MapTransform,
    ResizeWithPadOrCropd,
    RepeatChanneld,
    Rand3DElasticd,
    adaptor,
    ToNumpyd
    
)
from monai.transforms import Randomizable, apply_transform
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
import torchio
import torchio
import numpy as np
from typing import Any, Callable, Optional, Sequence
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.config import DtypeLike
from monai.transforms import Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed
from .preprocess_utils import z_score_norm
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass
from intensity_normalization.normalize.nyul import NyulNormalize
import os

def prepare_scan(path: str) -> "npt.NDArray[Any]":
    return np.expand_dims(
        sitk.GetArrayFromImage(
            sitk.ReadImage(path)
        ).astype(np.float32), axis=(0, 1)
    )



class loadImageMy(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        normalizationIndex,
        normalizerDict,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.normalizationIndex=normalizationIndex
        self.normalizerDict=normalizerDict

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            if(self.normalizationIndex==0):    
                d[key]=z_score_norm(prepare_scan(d[key]), 99.5)
            if(self.normalizationIndex==1):    
                nyul_normalizer=  self.normalizerDict[key]
                d[key]=nyul_normalizer(prepare_scan(d[key])).astype(np.float32)          
        return d

class concatImageMy(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        d = dict(data)
        img_t2w=d["t2w"]
        img_adc=d["adc"]
        img_hbv=d["hbv"]
        imgConc= np.concatenate([img_t2w, img_adc, img_hbv], axis=1)

        d["data"]=np.concatenate([img_t2w, img_adc, img_hbv], axis=1)

        return d


class printTransform(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        info,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.info=info

    def __call__(self, data):

        d = dict(data)
        print(self.info)
        return d

class loadlabelMy(MapTransform):

    def __init__(
    self,
    keys: KeysCollection,
    allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = sitk.GetArrayFromImage(sitk.ReadImage(d[key])).astype(np.int8)
            d[key] = np.expand_dims(d[key], axis=(0, 1))
        return d

class applyOrigTransforms(MapTransform): #RandomizableTransform
    def __init__(
    self,
    keys: KeysCollection,
    transform,
    allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.transform=transform

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] =  apply_transform(self.transform, d[key], map_items=False)
        return d
def loadTrainTransform(transform,seg_transform,batchTransforms,normalizationIndex,normalizerDict):
    return Compose([
            # printTransform(keys=["seg"],info=f"loadAndtransform "),
            loadImageMy(keys=["t2w","hbv","adc"],normalizationIndex=normalizationIndex,normalizerDict=normalizerDict),
            loadlabelMy(keys=["seg"]),
            #DivisiblePadd(keys=["t2w","hbv","adc","seg"],k=32),

            concatImageMy(keys=["t2w","hbv","adc"]),
            applyOrigTransforms(keys=["data"],transform=transform),
            applyOrigTransforms(keys=["seg"],transform=seg_transform),
            ToNumpyd(keys=["data","seg"]),
            adaptor(batchTransforms, {"data": "data"}),
            SelectItemsd(keys=["data","seg"]) ,
            monai.transforms.ToTensord(keys=["data","seg"], dtype=torch.float) 
             ]           )        
def loadValTransform(transform,seg_transform,normalizationIndex,normalizerDict):
    return Compose([
            # printTransform(keys=["seg"],info="loadAndtransform"),

            loadImageMy(keys=["t2w","hbv","adc"],normalizationIndex=normalizationIndex,normalizerDict=normalizerDict),
            loadlabelMy(keys=["seg"]),
            #DivisiblePadd(keys=["t2w","hbv","adc","seg"],k=32),
            concatImageMy(keys=["t2w","hbv","adc"]),
            applyOrigTransforms(keys=["data"],transform=transform),
            applyOrigTransforms(keys=["seg"],transform=seg_transform),
            SelectItemsd(keys=["data","seg"])  ,      
            monai.transforms.ToTensord(keys=["data","seg"], dtype=torch.float) 
            ])        

# def addBatchAugmentations(transforms,batchTransforms): 
#     return Compose(transforms
#                     ,printTransform(keys=["seg"],info="before adaptor")
#                     ,adaptor(batchTransforms, {"data": "data"}
#                     ,printTransform(keys=["seg"],info="after adaptor")

#                     ))           