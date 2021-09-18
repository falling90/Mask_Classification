from albumentations import *
from albumentations.pytorch import ToTensorV2
def my_transform(train:bool=True, img_size=(512, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mytransform = None

    if train:
        mytransform = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            # CenterCrop(224,224),
            ShiftScaleRotate(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    else:
        mytransform = Compose([
            Resize(img_size[0], img_size[1]),
            # CenterCrop(224,224),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return mytransform