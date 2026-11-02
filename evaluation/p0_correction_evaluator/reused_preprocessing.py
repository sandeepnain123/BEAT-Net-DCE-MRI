import torch
from monai.transforms import Compose,LoadImaged,EnsureChannelFirstd,Orientationd,Spacingd,NormalizeIntensityd,CropForegroundd,SpatialPadd,EnsureTyped
TARGET_SPACING=(0.8,0.8,1.0)
PATCH_SIZE=(128,128,64)
def get_preproc_transforms(is_train: bool):
    keys = ["image_0", "image_1", "image_2"]
    tfs = [
        LoadImaged(keys=keys + ["label"]),
        EnsureChannelFirstd(keys=keys + ["label"]),
        Orientationd(keys=keys + ["label"], axcodes="RAS"),
        Spacingd(keys=keys + ["label"], pixdim=TARGET_SPACING, mode=tuple(["bilinear"]*3 + ["nearest"])),
        NormalizeIntensityd(keys=keys, nonzero=True),
    ]
    if is_train:
        tfs.append(CropForegroundd(keys=keys + ["label"], source_key="image_0"))
    tfs.append(SpatialPadd(keys=keys + ["label"], spatial_size=PATCH_SIZE))
    tfs.append(EnsureTyped(keys=keys + ["label"], dtype=torch.float32))
    return Compose(tfs)
