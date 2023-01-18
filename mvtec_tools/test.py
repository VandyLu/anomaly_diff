import numpy as np
import mmcv

import cv2


import mvtec_ad 
from loader import build_dataloader

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

data_root = '/data1/lfb/MVTecAD/'
dataset = mvtec_ad.MVTecADDataset(data_root, test_mode=True, subset=('bottle', ))


f = '/data1/lfb/MVTecAD/carpet/train/good/000.png'


def test_dataloader():
    dataloader = build_dataloader(dataset, 
        imgs_per_gpu=4, 
        workers_per_gpu=4, 
        num_gpus=1, 
        dist=True) 

    for idx, data in enumerate(dataloader):
        print('No.{} | img: {} | gt: {} | cat: {} | anomaly: {} | info: {}'.format(
            idx, 
            data['img'].data[0].shape,
            data['gt'].data[0].shape,
            data['cat'],
            data['anomaly'],
            data['anomaly_info']))



def test_single_image():
    img = mmcv.imread(f)

    print(img.shape)

    #rimg = mmcv.imrotate(img, 45.0, auto_bound=True)
    #rimg = mmcv.imrotate(img, 90.0, auto_bound=False)
    rm = np.float32([[1,0, 30], [0, 1, -50]])
    rimg = cv2.warpAffine(img, rm, (img.shape[1], img.shape[0]), borderValue=[255.0, 255.0, 255.0])
    
    print(rimg.shape)

    aug = ShiftScaleRotate(shift_limit=0.0625, 
            scale_limit=0.,
            rotate_limit=45,
            #rotate_limit=0.,
            #border_mode=cv2.BORDER_CONSTANT,
            #value=(255,255,0),
            #border_mode=cv2.BORDER_REPLICATE,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0)
    simg = aug(image=img)['image']

    cv2.namedWindow('test', 0)
    cv2.namedWindow('simg', 0)
    
    cv2.imshow('test', img)
    cv2.imshow('simg', simg)
    cv2.waitKey()


if __name__ == '__main__':

    test_single_image()

