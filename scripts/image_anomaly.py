"""
Approximate the bits/dimension for an image model.
"""

import pickle
import argparse
import os
import cv2

import torch as th
import torch.nn.functional as F
import numpy as np
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

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
    )
    model.to(dist_util.dev())
    model.eval()
    logger.log("creating padim train data loader...")
    data_train = load_data(
        data_dir=args.train_data_dir,
        batch_size=1,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_flip=False,
        random_rotate=False,
        anomaly=True,
        infinte_loop=False,
        name=args.dataset_name
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_flip=False,
        random_rotate=False,
        anomaly=True,
        infinte_loop=False,
        name=args.dataset_name
    )

    logger.log("train padim...")
    from guided_diffusion.gaussian_diffusion import Padim
    padim = Padim(args.image_size, save_dir='./{}_padim_{}_model.pth'.format(args.category, args.image_size))
    padim.eval()
    if not padim.is_trained and args.use_padim:
        padim.train_padim(data_train)
        padim.save_model()

    logger.log("evaluating...")
    run_anomaly_evaluation(model, padim, diffusion, data, args.num_samples, args.clip_denoised, args)


def run_anomaly_evaluation(model, padim, diffusion, data, num_samples, clip_denoised, args):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    anom_metrics = {'roc': []}
    labels = []
    gt_masks = []

    pred_scores = []
    pred_masks = []
    diff_masks = []
    feat_masks = []
    img_paths = []
    for batch, gt_mask, model_kwargs in data:
        anom_gt = model_kwargs.pop('anom_gt')
        img_path = model_kwargs.pop('img_path')

        # save ground truth
        labels.append(anom_gt)
        gt_masks.append(gt_mask)
        img_paths.extend(img_path)

        # diffusion results
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        # model_kwargs['x_target'] = batch
        model_kwargs['padim'] = padim
        model_kwargs['diffusion_model'] = diffusion

        cond_fn = diffusion.feat_cond_fn
        model_kwargs['feature_extractor'] = diffusion.feature_extractor
        def model_fn(x, t, y=None, feats_start=None, get_feature=False, padim=None, feature_extractor=None, x_target=None, diffusion_model=None):
            return model(x, t, y if args.class_cond else None, feats_start=feats_start, get_feature=get_feature)

        output_masks = diffusion.calc_bpd_loop(
            model_fn, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs, visual_dir=args.visual_dir
        )

        diff_mask = output_masks['diff_mask']
        feat_mask = output_masks['feat_mask']

        pred_mask = feat_mask * args.alpha_factor + diff_mask * args.beta_factor
        # if args.smooth:
            # pred_mask = smooth_result(pred_mask)
        # pred_score = padim.get_score(pred_mask)

        pred_score = get_score(pred_mask)

        pred_scores.append(pred_score)
        # reduce dim: N1HW->NHW
        pred_masks.append(pred_mask[:, 0])
        feat_masks.append(feat_mask[:, 0])
        diff_masks.append(diff_mask[:, 0])

        # save results

        # for i, path in enumerate(img_path):
        #     origin_img_vis = ((batch[i] + 1) * 127.5).clip(0, 255).permute(1, 2, 0).detach().cpu().numpy()[..., ::-1].astype(np.uint8)
        #     pred_mask_vis = (5*pred_mask[i]).clip(0, 255).detach().cpu().numpy()
        #     diff_mask_vis = (5*diff_mask[i]).clip(0, 255).detach().cpu().numpy()
        #     feat_mask_vis = (5*feat_mask[i]).clip(0, 255).detach().cpu().numpy()
        #     cv2.imwrite(args.visual_dir + path, origin_img_vis)
        #     cv2.imwrite(args.visual_dir + path.replace('.png', '_pred.png'), pred_mask_vis)
        #     cv2.imwrite(args.visual_dir + path.replace('.png', '_diff.png'), diff_mask_vis)
        #     cv2.imwrite(args.visual_dir + path.replace('.png', '_feat.png'), feat_mask_vis)
    # save results
    gt_masks = (th.cat(gt_masks, dim=0)/255).long().cpu()
    pred_masks = th.cat(pred_masks, dim=0).cpu()
    diff_masks = th.cat(diff_masks, dim=0).cpu()
    feat_masks = th.cat(feat_masks, dim=0).cpu()

    for idx, img_name in enumerate(img_paths):
        basedir = os.path.join(args.save_path, args.category, '_'.join(img_name.split('_')[:-1]))
        filename = img_name.split('_')[-1]
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        print(basedir, filename)
        np.savez(os.path.join(basedir, filename), diff_mask=diff_masks[idx], feat_mask=feat_masks[idx], gt_mask=gt_masks[idx])


    if dist.get_rank() == 0:
        labels = th.cat(labels, dim=0)
        pred_scores = th.cat(pred_scores, dim=0)
        roc = evaluate(labels, pred_scores, metric='roc')
        print('roc: ', roc)
        # for i in range(len(img_paths)):
            # np.savez(img_paths[i], pred_masks[i].cpu().numpy())
        pro = evaluate(gt_masks, pred_masks, metric='pro')
        print('pro: ', pro)
        pproc = evaluate(gt_masks, pred_masks, metric='perpixel_roc')
        print('pixel: ', pproc)

    # if dist.get_rank() == 0:
        # for name, terms in all_metrics.items():
            # out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            # logger.log(f"saving {name} terms to {out_path}")
            # np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")

def smooth_result(preds):
    results = []
    for i in range(preds.shape[0]):
        results.append(cv2.GaussianBlur(preds[i].cpu().numpy(), (15, 15), 4.0))
    results = np.array(results)
    return th.from_numpy(results)

def create_argparser():
    defaults = dict(
        data_dir="", train_data_dir="", clip_denoised=True, num_samples=1000, batch_size=1,
        model_path="", alpha_factor=1.0, beta_factor=0.0, smooth=False, visual_dir="", use_padim=False, category="",
        save_path="./", dataset_name=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def get_score(anoms):
    score = anoms.flatten(1).topk(dim=-1, k=64)[0].mean(-1)
    return score

if __name__ == "__main__":
    main()
