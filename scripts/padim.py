"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import torch as th
import numpy as np
import cv2
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from mvtec_tools import evaluate


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    data_test = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=224,
        class_cond=False,
        random_flip=False,
        anomaly=True,
        infinte_loop=False,
    )
    data_train = load_data(
        data_dir=args.train_data_dir,
        batch_size=1,
        image_size=224,
        class_cond=False,
        random_flip=False,
        anomaly=True,
        infinte_loop=False,
    )
    from guided_diffusion.gaussian_diffusion import Padim
    padim = Padim()
    padim.eval()
    padim.train_padim(data_train)

    anom_metrics = {'roc': []}
    raw_imgs = []
    labels = []
    scores = []
    pred_masks = []
    gt_masks = []
    img_paths = []

    for data in data_test:
        img, gt_mask, model_kwargs = data

        anom_gt = model_kwargs.pop('anom_gt')
        img_path = model_kwargs.pop('img_path')
        labels.append(anom_gt)
        gt_masks.append(gt_mask)
        img_paths.extend(img_path)

        img = img.cuda()
        anoms, score = padim(img)

        # print(img_path, anom_gt, score)

        scores.append(score)
        pred_masks.append(anoms)
        raw_imgs.append(img)

    if dist.get_rank() == 0:
        labels = th.cat(labels, dim=0).long()
        scores = th.cat(scores, dim=0)
        roc = evaluate(labels, scores, metric='roc')
        print('roc: ', roc)
        gt_masks = (th.cat(gt_masks, dim=0)/255).long()
        pred_masks = th.cat(pred_masks, dim=0)

        pred_masks = smooth_result(pred_masks)
        diff_masks = []
        for i in range(len(img_paths)):
            diff = np.load(img_paths[i] + '.npz')['arr_0'][0] # 1, 128, 128
            diff = th.from_numpy(cv2.resize(diff, (224, 224))).float().cuda()
            diff_masks.append(diff)
        diff_masks = th.stack(diff_masks, dim=0)
        diff_masks = smooth_result(diff_masks)
        pred_masks = diff_masks * 5 + pred_masks
        pro = evaluate(gt_masks, pred_masks, metric='pro')
        print('pro: ', pro)
        pproc = evaluate(gt_masks, pred_masks, metric='perpixel_roc')
        print('pproc: ', pproc)
        for i in range(len(img_paths)):
            name = img_paths[i]
            # img = raw_imgs[i].permute(0, 2, 3, 1).cpu().numpy()[0] * np.array([0.229, 0.224, 0.225], dtype=np.float32) + np.array([0.485, 0.456, 0.406], dtype=np.float32)
            img = (raw_imgs[i].permute(0, 2, 3, 1).cpu().numpy()[0] + 1.0) / 2.0
            img = (img * 255).astype(np.uint8)
            result = ((pred_masks[i].cpu() * 3).clamp(0, 255).numpy()).astype(np.uint8)

            cv2.imwrite(name, img[:, :, ::-1])
            cv2.imwrite(name.replace('.png', '_pred.png'), result)



def smooth_result(preds):
    results = []
    for i in range(preds.shape[0]):
        results.append(cv2.GaussianBlur(preds[i].cpu().numpy(), (15, 15), 4.0))
    results = np.array(results)
    return th.from_numpy(results)

def create_argparser():
    defaults = dict(
        data_dir="", train_data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
