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
import torchio
import nibabel as nib
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
from pathlib import Path
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img

def prepare_scan(path: str) -> "npt.NDArray[Any]":
    return np.expand_dims(
        sitk.GetArrayFromImage(
            sitk.ReadImage(path)
        ).astype(np.float32), axis=(0, 1)
    )


class wrapTorchio(MapTransform):
    def __init__(
        self,
        torchioObj,
        keys: KeysCollection = "data",
        # p: float=0.2,
        allow_missing_keys: bool = False,
        
    ):
        super().__init__(keys, allow_missing_keys)
        self.keys=keys
        self.torchioObj=torchioObj

    def __call__(self, data):
        return self.torchioObj(data)
        # d = dict(data)
        # for key in self.keys:
        #     d[key] = torchioObj()   (d[key] > 0.5).astype('int8')
        # return d


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
            stemm= Path(d[key]).stem
            d[key+'_name']=stemm
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

        # img_fulProst=d["fullProst"]
        img_fulProst=d["fullProst"]

        # imgConc= np.concatenate([img_t2w, img_adc, img_hbv], axis=1)
        d["data"]=np.concatenate([img_t2w, img_adc, img_hbv,img_fulProst], axis=1)
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

def tryLoadImageReturnZeros(path, labelArr):
    if os.path.exists(path):
        print(f"found path {path}")
        loadedArr= (sitk.GetArrayFromImage(sitk.ReadImage(path))>0)
        loadedArr= crop_or_pad(loadedArr,labelArr.shape )
        labelArrBool = (labelArr>0)
        return np.logical_and(np.logical_not(labelArrBool),loadedArr)
    print('path not found')    
    return np.zeros_like(labelArr,dtype='bool')    
class loadlabelMy(MapTransform):

    def __init__(
    self,
    keys: KeysCollection,
    df,
    allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.df= df
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            stemm= Path(d[key]).stem
            d[key+'_name']=stemm
            patient_id, study_id= stemm.replace('.nii','').split('_')
            patient_id=int(patient_id)
            study_id= int(study_id)
            #prostPath=f"/home/sliceruser/locTemp/prostateFull/prostateFold/prostate_{study_id}.nii.gz"
            prostPath=f"/home/sliceruser/labels/anatomical_delineations/whole_gland/AI/Bosma22b/{patient_id}_{study_id}.nii.gz"
            # print(f"ppppp {patient_id} study_id {study_id}   ")
            df=self.df
            locDf = df.loc[df['study_id'] == study_id]
            case_csPCa= (locDf['case_csPCa'].to_numpy())[0]
            isCa=(case_csPCa== 'YES')
            # print(f"list case_csPCa {case_csPCa} isCa {isCa}")
            d['isCa']=int(isCa)
            d[key] = sitk.GetArrayFromImage(sitk.ReadImage(d[key])).astype(np.int8)
            imageProst = sitk.ReadImage(prostPath)
            d['fullProst']= crop_or_pad(sitk.GetArrayFromImage(imageProst),d[key].shape )
            # print(f"prostPath {prostPath}")
            #dilatated=ndimage.binary_dilation(img_fulProst, iterations=4)
            nnunetPath = f"/home/sliceruser/locTemp/orig_nnunet_semi/orig_nnunet_semi/nnunetOut_{study_id}.nii.gz"
            orig_nnunet_semi=tryLoadImageReturnZeros(nnunetPath, d[key])
            orig_unet_semi_path = f"/home/sliceruser/locTemp/orig_unet_semi/orig_unet_semi/uun_semi_super_{study_id}.nii.gz"
            orig_unet_semi=tryLoadImageReturnZeros(orig_unet_semi_path, d[key])
            swinPath = f"/home/sliceruser/locTemp/outMultiSwin/outMultiSwin/ca_{study_id}.nii.gz"
            swin=tryLoadImageReturnZeros(swinPath, d[key])
            d['wrongLabel']= np.logical_or(np.logical_or(orig_nnunet_semi  ,orig_unet_semi ), swin   )

            #d[key] = np.expand_dims(d[key], axis=(0, 1))            
            d['fullProst']= np.expand_dims(d['fullProst'], axis=(0, 1))
            #d['wrongLabel']= np.expand_dims(d['fullProst'], axis=(0, 1))
            d[key]=np.stack(np.zeros_like(d[key]),d[key],d['wrongLabel']  )    



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

class getToShape(MapTransform): #RandomizableTransform
    def __init__(
    self,
    keys: KeysCollection,
    allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] =  d[key][0,:,:,:,:]
        return d


        
def loadTrainTransform(transform,seg_transform,batchTransforms,normalizationIndex
,normalizerDict,expectedShape,df,RandomBiasField_prob
    ,RandomAnisotropy_prob):
    # print(f"hhhh {expectedShape}")
    return Compose([
            # printTransform(keys=["seg"],info=f"loadAndtransform "),
            loadImageMy(keys=["t2w","hbv","adc"],normalizationIndex=normalizationIndex,normalizerDict=normalizerDict),
            loadlabelMy(keys=["seg"],df=df),
            #DivisiblePadd(keys=["t2w","hbv","adc","seg"],k=32),
            concatImageMy(keys=["t2w","hbv","adc","fullProst"]),
            ToNumpyd(keys=["data","seg"]),
            monai.transforms.SpatialPadd(keys=["data"],spatial_size=expectedShape),#(3,32,256,256)
            monai.transforms.SpatialPadd(keys=["seg"],spatial_size=(1,expectedShape[1],expectedShape[2],expectedShape[3])),
            applyOrigTransforms(keys=["data"],transform=transform),
            applyOrigTransforms(keys=["seg"],transform=seg_transform),
            ToNumpyd(keys=["data","seg"]),
            adaptor(batchTransforms, {"data": "data","seg": "seg"}),
            SelectItemsd(keys=["data","seg_name","seg","t2w_name","hbv_name","adc_name","isCa"])  ,      
            monai.transforms.ToTensord(keys=["data","seg"], dtype=torch.float),
            getToShape(keys=["data","seg"]),
            wrapTorchio(torchio.transforms.RandomAnisotropy(include=["data"],p=RandomAnisotropy_prob)),
            wrapTorchio(torchio.transforms.RandomBiasField(include=["data"],p=RandomBiasField_prob))
             ]           )        
def loadValTransform(transform,seg_transform,normalizationIndex,normalizerDict,expectedShape,df):
    # print(f"hhhh {expectedShape}")

    return Compose([
            # printTransform(keys=["seg"],info="loadAndtransform"),

            loadImageMy(keys=["t2w","hbv","adc"],normalizationIndex=normalizationIndex,normalizerDict=normalizerDict),
            loadlabelMy(keys=["seg"],df=df),
            #DivisiblePadd(keys=["t2w","hbv","adc","seg"],k=32),
            concatImageMy(keys=["t2w","hbv","adc","fullProst"]),
            ToNumpyd(keys=["data","seg"]),
            monai.transforms.SpatialPadd(keys=["data"],spatial_size=expectedShape),
            monai.transforms.SpatialPadd(keys=["seg"],spatial_size=(1,expectedShape[1],expectedShape[2],expectedShape[3])),

            applyOrigTransforms(keys=["data"],transform=transform),
            applyOrigTransforms(keys=["seg"],transform=seg_transform),
            SelectItemsd(keys=["data","seg_name","seg","t2w_name","hbv_name","adc_name","isCa"])  ,      
            monai.transforms.ToTensord(keys=["data","seg"], dtype=torch.float),
            getToShape(keys=["data","seg"])

            ])        

# def addBatchAugmentations(transforms,batchTransforms): 
#     return Compose(transforms
#                     ,printTransform(keys=["seg"],info="before adaptor")
#                     ,adaptor(batchTransforms, {"data": "data"}
#                     ,printTransform(keys=["seg"],info="after adaptor")

#                     ))           