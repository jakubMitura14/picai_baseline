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
    Rand3DElasticd
    
)

from monai.config import KeysCollection
from monai.data import MetaTensor
import torchio
import torchio
import numpy as np

class standardizeLabels(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        #ref,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        #self.ref=ref

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            #print(f"in standdd {d[key].get_array().shape}")
            #print(f"in standd {d[key].meta} rrrrrrrrrrrrrrref {d[self.ref].meta}  ")
            #d[key].set_array(  np.flip((d[key].get_array() > 0.5).astype('int8'), (0, 1))   )
            d[key].set_array(  np.flip((d[key].get_array() > 0.5).astype('int8'), (1,2))   )
            #d[key].set_array(   (d[key].get_array() > 0.5).astype('int8')   )
            #d[key].meta['pixdim']=d[self.ref].meta['spacing']
            #update_meta(pixdim=d[self.ref].pixdim
            #print(f" {d['study_id']} label size {d[key].get_array().shape}  imSize  {d[self.ref].get_array().shape} labels_spac {d[key].pixdim} im_spac {d[self.ref].pixdim} ")
            # print(f" d[key].pixdim {d[key].pixdim} d[self.ref].pixdim {d[self.ref].pixdim} label size {d[key].get_array().shape}  imSize  {d[self.ref].get_array().shape} ")
        return d



# class standardizeLabels(MapTransform):

#     def __init__(
#         self,
#         keys: KeysCollection = "label",
#         allow_missing_keys: bool = False,
#     ):
#         super().__init__(keys, allow_missing_keys)

#     def __call__(self, data):

#         d = dict(data)
#         for key in self.keys:
#             d[key] = (d[key] > 0.5).astype('int8')
#         return d

class wrapTorchio(MapTransform):
    def __init__(
        self,
        torchioObj,
        keys: KeysCollection = "chan3_col_name",
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



def decide_if_whole_image_train(is_whole_to_train, chan3Name,labelName):
    """
    if true we will trian on whole images otherwise just on 64x64x32
    randomly cropped parts
    """
    fff=not is_whole_to_train
    print(f" in decide_if_whole_image_train {fff}")
    if(not is_whole_to_train):
        return [RandCropByPosNegLabeld(
                keys=[chan3Name,labelName],
                label_key=labelName,
                spatial_size=(96, 96, 32),
                pos=1,
                neg=1,
                num_samples=2,
                image_key=chan3Name,
                image_threshold=0
            )
             ]
    return []         

def concatInOne(isVnet):
    if(isVnet):
        return ["t2w","adc","hbv" ,"adc"]  
    return ["t2w","adc","hbv" ]
      


#https://github.com/DIAGNijmegen/picai_prep/blob/19a0ef2d095471648a60e45e30b218b7a81b2641/src/picai_prep/preprocessing.py


def get_train_transforms(
            RandAdjustContrastd_prob
            ,RandGaussianSmoothd_prob
            ,RandRicianNoised_prob
            ,RandFlipd_prob
            ,RandAffined_prob
            ,RandomElasticDeformation_prob
            ,RandomAnisotropy_prob
            ,RandomMotion_prob
            ,RandomGhosting_prob
            ,RandomSpike_prob
            ,RandomBiasField_prob  
            ,isVnet
     ):


     
    train_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label"],reader="ITKReader"),
            #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label"]),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label"]),
            standardizeLabels(keys=["label"]),

            AsDiscreted(keys=["label"],to_onehot=2),
            #ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label"], spatial_size=spatial_size,),
            ConcatItemsd(keys=concatInOne(isVnet),name="chan3_col_name"),

            #torchio.transforms.OneHot(include=["label"] ), #num_classes=3,
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            #Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),            
            #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
            # SelectItemsd(keys=["chan3_col_name","label"]),
            #DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,            
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label"],spatial_size=centerCropSize ),
          
            #*decide_if_whole_image_train(False,"chan3_col_name","label"),
            #SpatialPadd(keys=["chan3_col_name","label"]],spatial_size=maxSize) , 
            SelectItemsd(keys=["chan3_col_name","label","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),           
            RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
            #RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
            RandRicianNoised(keys=["chan3_col_name",], prob=RandRicianNoised_prob),
            RandFlipd(keys=["chan3_col_name","label"], prob=RandFlipd_prob),
            RandAffined(keys=["chan3_col_name","label"], prob=RandAffined_prob),
            Rand3DElasticd(keys=["chan3_col_name","label"],sigma_range= (1, 2), magnitude_range=(0.5, 0.5) ,prob=RandomElasticDeformation_prob),
            #wrapTorchio(torchio.transforms.RandomElasticDeformation(include=["chan3_col_name","label"],p=RandomElasticDeformation_prob)),
            #wrapTorchio(torchio.transforms.RandomAnisotropy(include=["chan3_col_name","label"],p=RandomAnisotropy_prob)),
            #wrapTorchio(torchio.transforms.RandomMotion(include=["chan3_col_name"],p=RandomMotion_prob)),
            #wrapTorchio(torchio.transforms.RandomGhosting(include=["chan3_col_name"],p=RandomGhosting_prob)),
            #wrapTorchio(torchio.transforms.RandomSpike(include=["chan3_col_name"],p=RandomSpike_prob)),
            #wrapTorchio(torchio.transforms.RandomBiasField(include=["chan3_col_name"],p=RandomBiasField_prob)),
            

        ]
    )
    return train_transforms


def get_train_transforms_noLabel(
            RandAdjustContrastd_prob
            ,RandGaussianSmoothd_prob
            ,RandRicianNoised_prob
            ,RandFlipd_prob
            ,RandAffined_prob
            ,RandomElasticDeformation_prob
            ,RandomAnisotropy_prob
            ,RandomMotion_prob
            ,RandomGhosting_prob
            ,RandomSpike_prob
            ,RandomBiasField_prob  
            ,isVnet
     ):


     
    train_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ],reader="ITKReader"),
            EnsureTyped(keys=["t2w","hbv","adc" ]),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ]),
            # ConcatItemsd(keys=["t2w","adc","hbv" ],name="chan3_col_name"),
            ConcatItemsd(keys=concatInOne(isVnet),name="chan3_col_name"),
            SelectItemsd(keys=["chan3_col_name","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),           
            RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
            #RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
            RandRicianNoised(keys=["chan3_col_name",], prob=RandRicianNoised_prob),
            RandFlipd(keys=["chan3_col_name"], prob=RandFlipd_prob),
            # RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
            # RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
            # RandRicianNoised(keys=["chan3_col_name",], prob=RandRicianNoised_prob),
            # RandFlipd(keys=["chan3_col_name"], prob=RandFlipd_prob),
            RandAffined(keys=["chan3_col_name"], prob=RandAffined_prob),
            Rand3DElasticd(keys=["chan3_col_name"],sigma_range= (1, 2), magnitude_range=(0.5, 0.5) ,prob=RandomElasticDeformation_prob),

            #wrapTorchio(torchio.transforms.RandomElasticDeformation(include=["chan3_col_name"],p=RandomElasticDeformation_prob)),
            #wrapTorchio(torchio.transforms.RandomAnisotropy(include=["chan3_col_name"],p=RandomAnisotropy_prob)),
            # wrapTorchio(torchio.transforms.RandomMotion(include=["chan3_col_name"],p=RandomMotion_prob)),
            # wrapTorchio(torchio.transforms.RandomGhosting(include=["chan3_col_name"],p=RandomGhosting_prob)),
            # wrapTorchio(torchio.transforms.RandomSpike(include=["chan3_col_name"],p=RandomSpike_prob)),
            # wrapTorchio(torchio.transforms.RandomBiasField(include=["chan3_col_name"],p=RandomBiasField_prob)),
            

        ]
    )
    return train_transforms


def get_val_transforms(isVnet):

    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label_name_val"],reader="ITKReader"),
            #Orientationd(keys=["t2w","adc", "hbv","label_name_val"], axcodes="RAS"),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label_name_val"]),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label_name_val"]),
            standardizeLabels(keys=["label_name_val"]),
            AsDiscreted(keys=["label_name_val"], to_onehot=2),
            #ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label_name_val"], spatial_size=spatial_size,),
            ConcatItemsd(concatInOne(isVnet), "chan3_col_name_val"),
            # ConcatItemsd(["t2w","adc","hbv"], "chan3_col_name_val"),

            #torchio.transforms.OneHot(include=["label"] ),#num_classes=3
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["chan3_col_name","label"]]),
            #Orientationd(keys=["chan3_col_name","label"]], axcodes="RAS"),
            # Spacingd(keys=["chan3_col_name","label"]], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
            #DivisiblePadd(keys=["chan3_col_name_val","label_name_val"],k=32) ,
            #ResizeWithPadOrCropd(keys=["chan3_col_name_val","label_name_val"],spatial_size=spatial_size ),
            SelectItemsd(keys=["labelB","t2wb","chan3_col_name_val","label_name_val","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),
            #*decide_if_whole_image_train(is_whole_to_train,"chan3_col_name_val","label_name_val"),
            #SelectItemsd(keys=["chan3_col_name","label"]),
            # ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name")
        ]
    )
    return val_transforms

def get_debug_transforms():
    spatial_size=(64,64,32)
    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label_name_val"]),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label_name_val"]),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label_name_val"]),
            Orientationd(keys=["t2w","adc", "hbv","label_name_val"], axcodes="RAS"),
            standardizeLabels(keys=["label_name_val"]),#,ref= "t2w"
            # Spacingd(keys=["t2w","adc","hbv"], pixdim=(
            #     1.0, 1.0, 1.0), mode="bilinear"),      
            # Spacingd(keys=["label_name_val"], pixdim=(
            #     1.0, 1.0, 1.0), mode="nearest"),  
            # Spacingd(keys=["t2w","adc","hbv","label_name_val"], pixdim=(
            #     1.0, 1.0, 1.0), mode=("bilinear","bilinear","bilinear","nearest") ),      #monai.utils.SplineMode.THREE
            # Spacingd(keys=["t2w","adc","hbv","label_name_val"], pixdim=(
            #     1.0, 1.0, 1.0), mode=("bilinear","bilinear","bilinear","nearest") ),      #monai.utils.SplineMode.THREE

            standardizeLabels(keys=["label_name_val"]),#,ref= "t2w"

            ConcatItemsd(["t2w","label_name_val","hbv","adc" ], "dummy"),


            ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label_name_val"], spatial_size=spatial_size,),
            ConcatItemsd(["t2w","adc","hbv","adc" ], "chan3_col_name_val"),
            AsDiscreted(keys=["label_name_val"], to_onehot=2),

            #torchio.transforms.OneHot(include=["label"] ),#num_classes=3
            #AsChannelFirstd(keys=["chan3_col_name","label"]]),
            # Orientationd(keys=["chan3_col_name","label_name_val"], axcodes="RAS"),
            # Spacingd(keys=["chan3_col_name","label_name_val"], pixdim=(
            #     1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
            #DivisiblePadd(keys=["chan3_col_name_val","label_name_val"],k=32) ,
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label_name_val"],spatial_size=centerCropSize ),

            #*decide_if_whole_image_train(is_whole_to_train,"chan3_col_name_val","label_name_val"),
            SelectItemsd(keys=["chan3_col_name_val","label_name_val","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),
            # ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name")
        ]
    )
    return val_transforms




def decide_if_whole_image_train(is_whole_to_train, chan3Name,labelName):
    """
    if true we will trian on whole images otherwise just on 64x64x32
    randomly cropped parts
    """
    fff=not is_whole_to_train
    print(f" in decide_if_whole_image_train {fff}")
    if(not is_whole_to_train):
        return [RandCropByPosNegLabeld(
                keys=[chan3Name,labelName],
                label_key=labelName,
                spatial_size=(96, 96, 32),
                pos=1,
                neg=1,
                num_samples=2,
                image_key=chan3Name,
                image_threshold=0
            )
             ]
    return []         


#https://github.com/DIAGNijmegen/picai_prep/blob/19a0ef2d095471648a60e45e30b218b7a81b2641/src/picai_prep/preprocessing.py


# def get_train_transforms(RandGaussianNoised_prob
#     ,RandAdjustContrastd_prob
#     ,RandGaussianSmoothd_prob
#     ,RandRicianNoised_prob
#     ,RandFlipd_prob
#     ,RandAffined_prob
#     ,RandCoarseDropoutd_prob
#     ,is_whole_to_train
#     ,spatial_size
#      ):


     
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","hbv","adc" ,"label"]),
#             EnsureTyped(keys=["t2w","hbv","adc" ,"label"]),
#             EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label"]),
#             standardizeLabels(keys=["label"]),#,ref= "t2w"
#             Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
#             Spacingd(keys=["t2w","adc","hbv","label"], pixdim=(
#                 1.0, 1.0, 1.0), mode=("bilinear","bilinear","bilinear","nearest") ),      #monai.utils.SplineMode.THREE
#             # Spacingd(keys=["label"], pixdim=(
#             #     1.0, 1.0, 1.0), mode="nearest"),     
#             # Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
#             #     1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#             ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label"], spatial_size=spatial_size,),
#             ConcatItemsd(keys=["t2w","adc","hbv","adc" ],name="chan3_col_name"),
#             SelectItemsd(keys=["chan3_col_name","label","num_lesions_to_retain","isAnythingInAnnotated"]),
#             AsDiscreted(keys=["label"],to_onehot=2),

#             # standardizeLabels(keys=["label"]),
#             # torchio.transforms.OneHot(include=["label"] ), #num_classes=3,
#             #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
#             #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
       
#             #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
#             # SelectItemsd(keys=["chan3_col_name","label"]),
#             #DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,            
#             #ResizeWithPadOrCropd(keys=["chan3_col_name","label"],spatial_size=centerCropSize ),
          
#             #*decide_if_whole_image_train(False,"chan3_col_name","label"),
#             #SpatialPadd(keys=["chan3_col_name","label"]],spatial_size=maxSize) ,            
#             RandGaussianNoised(keys=["chan3_col_name"], prob=RandGaussianNoised_prob),
#             RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
#             RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
#             RandRicianNoised(keys=["chan3_col_name",], prob=RandRicianNoised_prob),
#             RandFlipd(keys=["chan3_col_name","label"], prob=RandFlipd_prob),
#             RandAffined(keys=["chan3_col_name","label"], prob=RandAffined_prob),
#             RandCoarseDropoutd(keys=["chan3_col_name"], prob=RandCoarseDropoutd_prob,holes=6, spatial_size=5),
            
#         ]
#     )
#     return train_transforms
# def get_val_transforms(is_whole_to_train,spatial_size):
#     print(f"spatial_sizeeee {spatial_size}"  )

#     val_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","hbv","adc" ,"label_name_val"]),
#             EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label_name_val"]),
#             EnsureTyped(keys=["t2w","hbv","adc" ,"label_name_val"]),
#             Orientationd(keys=["t2w","adc", "hbv","label_name_val"], axcodes="RAS"),
           
#             standardizeLabels(keys=["label_name_val"]),#,ref= "t2w"
           
#             # Spacingd(keys=["t2w","adc","hbv"], pixdim=(
#             #     1.0, 1.0, 1.0), mode="bilinear"),      
#             # Spacingd(keys=["label_name_val"], pixdim=(
#             #     1.0, 1.0, 1.0), mode="nearest"),  
#             Spacingd(keys=["t2w","adc","hbv","label_name_val"], pixdim=(
#                 1.0, 1.0, 1.0), mode=("bilinear","bilinear","bilinear","nearest") ),      #monai.utils.SplineMode.THREE
#             # Spacingd(keys=["t2w","adc","hbv","label_name_val"], pixdim=(
#             #     1.0, 1.0, 1.0), mode=("bilinear","bilinear","bilinear","nearest") ),      #monai.utils.SplineMode.THREE

#             ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label_name_val"], spatial_size=spatial_size,),
#             ConcatItemsd(["t2w","adc","hbv","adc" ], "chan3_col_name_val"),
#             AsDiscreted(keys=["label_name_val"], to_onehot=2),

#             #torchio.transforms.OneHot(include=["label"] ),#num_classes=3
#             #AsChannelFirstd(keys=["chan3_col_name","label"]]),
#             # Orientationd(keys=["chan3_col_name","label_name_val"], axcodes="RAS"),
#             # Spacingd(keys=["chan3_col_name","label_name_val"], pixdim=(
#             #     1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#             #SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
#             #DivisiblePadd(keys=["chan3_col_name_val","label_name_val"],k=32) ,
#             #ResizeWithPadOrCropd(keys=["chan3_col_name","label_name_val"],spatial_size=centerCropSize ),

#             #*decide_if_whole_image_train(is_whole_to_train,"chan3_col_name_val","label_name_val"),
#             SelectItemsd(keys=["chan3_col_name_val","label_name_val","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),
#             # ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name")
#         ]
#     )
#     return val_transforms

