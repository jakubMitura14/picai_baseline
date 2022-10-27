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
    adaptor
    
)
from monai.transforms import Randomizable, apply_transform

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


def randomize(self, data: Optional[Any] = None) -> None:
    self._seed = self.R.randint(MAX_SEED, dtype="uint32")

def prepare_scan(self, path: str) -> "npt.NDArray[Any]":
    return np.expand_dims(
        sitk.GetArrayFromImage(
            sitk.ReadImage(path)
        ).astype(np.float32), axis=(0, 1)
    )



class loadImageMy(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            d[key]=z_score_norm(prepare_scan(d[key]), 99.5)
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

class applyOrigTransforms(MapTransform):

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

def loadAndtransform(transform,seg_transform):
    return Compose([
            loadImageMy(keys=["t2w","hbv","adc"]),
            loadlabelMy(keys=["label"]),
            concatImageMy(keys=["t2w","hbv","adc"]),
            applyOrigTransforms(keys=["data"],transform=transform),
            applyOrigTransforms(keys=["seg"],transform=seg_transform),
            SelectItemsd(keys=["data","seg"])  ]      
            )        

def addBatchAugmentations(transforms,batchTransforms): 
    return Compose(transforms
                    ,adaptor(batchTransforms, {"data": "data"}))           