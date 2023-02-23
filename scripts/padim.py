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
        image_size=256,
        class_cond=False,
        random_flip=False,
        random_rotate=False,
        anomaly=True,
        infinte_loop=False,
    )
    data_train = load_data(
        data_dir=args.train_data_dir,
        batch_size=1,
        image_size=256,
        class_cond=False,
        random_flip=False,
        random_rotate=False,
        anomaly=True,
        infinte_loop=False,
    )
    from guided_diffusion.gaussian_diffusion import Padim

    anom_metrics = {'roc': []}

    results = dict()


    # noises definition
    beta_start, beta_end = 0.0001, 0.02
    betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    noise_scale = alphas_cumprod[::50][:5] # first 100 steps
    noise_scale = np.array([1.0, *noise_scale], dtype=np.float64)
    noise_scale = np.array([1.0], dtype=np.float64)

    with th.no_grad():
        for idx, scale in enumerate(noise_scale):
            print('idx: {} | scale: {}'.format(idx, scale))
            padim = Padim(256)
            padim.eval()

            # noise = th.zeros((1, 3, 224, 224)).cuda()
            noise = 0
            # noise = th.randn((1, 3, 224, 224)).cuda()
            alpha_bar_t = scale

            padim.train_padim(data_train, scale, noise)
            print('finish padim train...')
            raw_imgs = []
            labels = []
            gt_masks = []
            img_paths = []
            scores = []
            pred_masks = []

            for data in data_test:
                img, gt_mask, model_kwargs = data

                anom_gt = model_kwargs.pop('anom_gt')
                img_path = model_kwargs.pop('img_path')
                # if idx == 0:
                labels.append(anom_gt)
                gt_masks.append(gt_mask)
                img_paths.extend(img_path)
                raw_imgs.append(img)

                img = img.cuda()

                img_noisy = np.sqrt(alpha_bar_t) * img + np.sqrt(1.0 - alpha_bar_t) * noise

                anoms = padim(img_noisy)
                anoms = anoms[:, 0]
                score = padim.get_score(anoms)

                # print(img_path, anom_gt, score)
                img_path = img_path[0]
                if img_path not in results.keys():
                    results[img_path] = anoms.detach() * scale
                else:
                    results[img_path] += anoms.detach() * scale

                scores.append(score)
                pred_masks.append(anoms)

            labels_eval = th.cat(labels, dim=0).long()
            scores = th.cat(scores, dim=0)
            roc = evaluate(labels_eval, scores, metric='roc')
            print('idx: ', idx, 'roc: ', roc)
            gt_masks_eval = (th.cat(gt_masks, dim=0)/255).long()
            pred_masks = th.cat(pred_masks, dim=0)

            pred_masks = smooth_result(pred_masks)
            pro = evaluate(gt_masks_eval, pred_masks, metric='pro')
            print('idx: ', idx, 'pro: ', pro)
            pproc = evaluate(gt_masks_eval, pred_masks, metric='perpixel_roc')
            print('idx: ', idx, 'pproc: ', pproc)

            for i in range(len(img_paths)):
                name = img_paths[i]
                # img = raw_imgs[i].permute(0, 2, 3, 1).cpu().numpy()[0] * np.array([0.229, 0.224, 0.225], dtype=np.float32) + np.array([0.485, 0.456, 0.406], dtype=np.float32)
                img = (raw_imgs[i].permute(0, 2, 3, 1).cpu().numpy()[0] + 1.0) / 2.0
                img = (img * 255).astype(np.uint8)
                result = ((pred_masks[i].cpu() * 3).clamp(0, 255).numpy()).astype(np.uint8)

                cv2.imwrite(name, img[:, :, ::-1])
                cv2.imwrite(name.replace('.png', '_pred.png'), result)
            # exit()

    pred_masks = [results[fname] / len(noise_scale) for fname in img_paths]
    scores = [padim.get_score(mask) for mask in pred_masks]

    if dist.get_rank() == 0:
        labels = th.cat(labels, dim=0).long()
        scores = th.cat(scores, dim=0)
        roc = evaluate(labels, scores, metric='roc')
        print('roc: ', roc)
        gt_masks = (th.cat(gt_masks, dim=0)/255).long()
        pred_masks = th.cat(pred_masks, dim=0)

        pred_masks = smooth_result(pred_masks)
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
