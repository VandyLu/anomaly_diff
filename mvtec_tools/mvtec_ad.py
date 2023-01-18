import os
import os.path as osp
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

#from utils import to_tensor
from .utils import to_tensor

from albumentations import (
    Resize, RandomCrop, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from .mvtec_evaluate import evaluate

class MVTecADDataset(Dataset):

    CLASSES = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
               'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
               'transistor', 'wood', 'zipper', 'lead_256'
              )

    def __init__(self,
                 root,
                 test_mode=False,
                 gray_scale=False,
                 stack_gray=False,
                 normalize=True,
                 img_scale=None,
                 crop_size=None,
                 flip_ratio=0.,
                 translation=None,
                 rotation=None,
                 pad_value=None,
                 subset=None):
        self.root = root
        self.stack_gray = stack_gray
        self.gray_scale = gray_scale
        self.subset = subset
        self.normalize = normalize

        if img_scale:
            if not isinstance(img_scale, (list, tuple)):
                img_scale = (img_scale, img_scale) # rectangle
        self.img_scale = img_scale
        self.crop_size = crop_size
        self.flip_ratio = flip_ratio
        assert self.flip_ratio >= 0 and self.flip_ratio <= 1.0

        self.translation = translation
        self.rotation = rotation
        self.pad_value = pad_value

        if self.translation:
            assert isinstance(self.translation, (list, tuple))

        if self.rotation:
            assert isinstance(self.rotation, (list, tuple))

        if self.subset is not None:
            if not isinstance(self.subset, (list, tuple)):
                self.subset = (self.subset, )

        else:
            self.subset = self.CLASSES

        start_time = time.time()

        self.test_mode = test_mode

        self.tags2cats = {tag:id for id, tag in enumerate(self.CLASSES)}
        self.cats2tags = {id:tag for id, tag in enumerate(self.CLASSES)}

        # path of images
        imgs = []
        cats = []
        anomaly = []
        anomaly_info = []
        gts = []

        # number of images for each category
        cls_num = []

        for cls in self.subset:
            cat_id = self.tags2cats[cls]

            suffix = 'test' if self.test_mode else 'train'
            base_dir = osp.join(self.root, cls, suffix)

            defect_attr = os.listdir(base_dir)

            sum = 0
            for attr in defect_attr:
                sub_dir = osp.join(base_dir, attr)
                filenames = os.listdir(sub_dir)
                img_files = [osp.join(sub_dir, f) for f in filenames]

                if attr == 'good':
                    anomaly_value = 0
                    gt_files = [None] * len(img_files)
                else:
                    anomaly_value = 1
                    gt_files = [osp.join(self.root, cls, 'ground_truth', attr, '{}_mask.{}'.format(*f.split('.'))) for f in filenames]

                    gt_files = [ f if osp.exists(f) else None for f in gt_files]

                imgs.extend(img_files)
                gts.extend(gt_files)
                cats.extend([cat_id]*len(img_files))
                anomaly.extend([anomaly_value]*len(img_files))
                anomaly_info.extend([attr]*len(img_files))
                sum += len(img_files)

            cls_num.append(sum)

        self.cls_num = cls_num

        # extend by 10 times
        n_extend = 10 if not self.test_mode else 1
        self.imgs = imgs * n_extend
        self.cats = cats * n_extend
        self.gts = gts * n_extend
        self.anomaly = anomaly * n_extend
        self.anomaly_info = anomaly_info * n_extend

        print('Dataset loaded! t={}'.format(time.time()-start_time))


    def _load_image(self, filename):
        ''' Input image could be RGB scale or gray scale
        '''
        # shape: HxWx3 or HxW
        img = Image.open(filename)
        if self.gray_scale:
            img = img.convert('L')

        if img.mode == 'L':
            img = np.array(img)[:,:, None]
            if self.stack_gray:
                img = np.tile(img, (1, 1, 3))
        else:
            img = np.array(img)

        return img

    def _load_groundtruth(self, filename):
        ''' binary image, 0/1, np.uint8
        '''
        img = Image.open(filename)
        assert img.mode == 'L'

        img = np.uint8(img) 
        return img[..., None]

    def _translation(self, img, x, y, pad_value=None):
        t_m = np.float32([[1, 0, x], [0, 1, y]])

        return cv2.warpAffine(img, t_m, (img.shape[0], img.shape[1]), borderValue=pad_value)


    def prepare_train_img(self, idx):
        img_file = self.imgs[idx]
        img = self._load_image(img_file)
        cat = self.cats[idx]
        tag = self.cats2tags[cat]

        gt_file = self.gts[idx]

        if self.test_mode and gt_file is not None:
            gt = self._load_groundtruth(gt_file)
        else:
            gt = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

        img = np.float64(img) / 255.0

        # resize image
        if self.img_scale:
            # mmcv.imrescale return tensor with size (w, h)
            img = mmcv.imrescale(img, self.img_scale, return_scale=False)
            if len(img.shape) == 2:
                img = img[..., None]

            gt = mmcv.imrescale(gt, self.img_scale, return_scale=False)[..., None]

        
        flip = True if np.random.rand() < self.flip_ratio else False

        if flip:
            img = mmcv.imflip(img)
            gt = mmcv.imflip(gt) 

        if self.translation:
            x_t = np.random.choice(self.translation)
            y_t = np.random.choice(self.translation)
            img = self._translation(img, x_t, y_t)
            gt  = self._translation(gt,  x_t, y_t) 

        if self.rotation:
            r = np.random.choice(self.rotation)
            img = mmcv.imrotate(img, r)
            gt  = mmcv.imrorate(gt, r) 

        #if self.normalize:
            #img = img.astype(np.float32) / 255.0
            #mean = np.array(self.IMG_MEANS[tag])
            #std = np.array(self.IMG_STDS[tag])
            #img -= mean[None, None, ...]
            #img *= 1.0/std[None, None, ...]

        #print(np.mean(img[..., 0]))
        #print(np.std(img[..., 0]))

        img = img.transpose(2, 0, 1) # to (c, h, w)
        gt = np.float64(gt.transpose(2, 0, 1)) # to (c, h, w)

        data = dict(img=DC(to_tensor(img), stack=True),
                    gt=DC(to_tensor(gt), stack=True),
                    #cat=DC(self.cats[idx], cpu_only=True),
                    #anomaly=self.anomaly[idx],
                    #anomaly_info=DC(self.anomaly_info[idx], cpu_only=True))
                    )
        return data


    def prepare_test_img(self, idx):
        ''' currently same as train
        '''
        result = self.prepare_train_img(idx)
        return {k:v.data for k, v in result.items()}

    def save_results(self, results, work_dir, n_epoch, normalize=None):

        res_dir = osp.join(work_dir, 'res_e{:05}'.format(n_epoch))
        if not osp.exists(res_dir):
            os.mkdir(res_dir)

        for i, result in enumerate(results):

            rec_img, result = result

            img_name = self.imgs[i].split('/')[-1]
            atr_name = self.anomaly_info[i]

            if not osp.exists(osp.join(res_dir, atr_name)):
                os.mkdir(osp.join(res_dir, atr_name))

            res_name = osp.join(res_dir, atr_name, img_name)
            prd_name = '{}_pred.{}'.format(*res_name.split('.'))

            if normalize:
                result = normalize(result)
            prd_img = (rec_img+1.0)/2.0 * 255
            prd_img = prd_img.squeeze(0).permute(1,2,0)
            prd_img = prd_img.cpu().numpy().astype(np.uint8)
            #result = (result - result.min())/(result.max()-result.min()+1e-5)

            result = result.cpu().numpy().squeeze((0, 1))
            result = result.astype(np.uint8)
            Image.fromarray(result).save(res_name)
            Image.fromarray(prd_img).save(prd_name)
            #cv2.imwrite(res_name, result)
            #cv2.imwrite(prd_name, prd_img)

    def evaluate(self, 
            results,
            metric='AUROC',
            logger=None,
            scale_ranges=None):
        ''' result: list(rec_img, anomaly)
        '''
        #if not isinstance(metric, str):
        #    assert len(metric) == 1
        #    metric = metric[0]

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric

        results = [t[1] for t in results]

        eval_results = {}
        allowed_metrics = ['AUROC', 'AUPRC']

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

            if metric == 'AUROC':
                num = len(results)
                labels = torch.as_tensor(self.anomaly, dtype=torch.float32)
                scores = torch.as_tensor([ano_map.mean().data for ano_map in results], dtype=torch.float32)
                eval_results['AUROC'] = evaluate(labels, scores, metric='roc')
            elif metric == 'AUPRC':
                num = len(results)
                labels = torch.as_tensor(self.anomaly, dtype=torch.float32)
                scores = torch.as_tensor([ano_map.mean().data for ano_map in results], dtype=torch.float32)
                eval_results['AUPRC'] = evaluate(labels, scores, metric='auprc')

        return eval_results

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)


def get_means_stds(path):
    classes = MVTecADDataset.CLASSES

    # get mean and std for each category
    img_means = {}
    img_stds = {}

    for cls in classes:
        dataset = MVTecADDataset(path, subset=(cls, ), normalize=False)

        img_list = []
        for i in range(len(dataset)):
            img = dataset[i][0].astype(np.float64) / 255.0
            img_list.append(img)
        img_list = np.stack(img_list, 0)

        channels = img_list.shape[-1]
        img_list = img_list.reshape(-1, channels)

        mean = np.mean(img_list, 0)
        std = np.std(img_list, 0)

        print('class: {} | mean: {} | std: {}'.format(cls, mean, std))

        img_means[cls] = mean
        img_stds[cls] = std

    print('img_means: ', img_means)
    print('img_stds: ', img_stds)

if __name__=='__main__':
    path = '/data1/lfb/MVTecAD/'

    #get_means_stds(path)
    #exit()

    import matplotlib.pyplot as plt
    train_dataset = MVTecADDataset(path, translation=(-30, -15, 0, 15, 30), rotation=(0, 90, 180, 270))

    print('trainset size: ', len(train_dataset))
    for i in range(len(train_dataset)):
        if i % 20 == 0:
            result = train_dataset[i]
            print('No.{}: {} | {}'.format(i, result[0].shape, result[1:]))
            plt.imshow(result[0]+0.5)
            plt.show()

    test_dataset = MVTecADDataset(path, test_mode=True)
    print('testset size: ', len(test_dataset))

    for i in range(len(test_dataset)):
        if i % 20 == 0:
            result = test_dataset[i]
            print('No.{}: {} | {}'.format(i, result[0].shape, result[2:]))

