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
        dist_util.load_state_dict(args.model_path, map_location="cpu")
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
    scores = []

    pred_masks = []
    gt_masks = []
    img_paths = []
    idx = 0 
    for batch, gt_mask, model_kwargs in data:
        anom_gt = model_kwargs.pop('anom_gt')
        img_path = model_kwargs.pop('img_path')
        # idx += 1
        # if idx > 10:
        #     break

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

        minibatch_metrics = diffusion.calc_bpd_loop(
            model_fn, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
        )
        # diff_masks.append(minibatch_metrics['pred_mask'])
        diff_mask = minibatch_metrics['pred_mask']

        # padim results
        if args.use_padim:
            with th.no_grad():
                feat_mask = padim(batch)
        else:
            feat_mask = th.zeros_like(diff_mask)
        feat_mask = feat_mask[:, 0]
        # print(feat_mask.max())
        diff_mask = F.interpolate(diff_mask, size=feat_mask.shape[-2:], mode='bilinear')
        diff_mask = diff_mask[:, 0]
        
        pred_mask = feat_mask + diff_mask * args.alpha_factor
        if args.smooth:
            pred_mask = smooth_result(pred_mask)
        pred_score = padim.get_score(pred_mask)

        # save results
        scores.append(pred_score)    
        pred_masks.append(pred_mask)

        # for i, path in enumerate(img_path):
        #     origin_img_vis = ((batch[i] + 1) * 127.5).clip(0, 255).permute(1, 2, 0).detach().cpu().numpy()[..., ::-1].astype(np.uint8)
        #     pred_mask_vis = (5*pred_mask[i]).clip(0, 255).detach().cpu().numpy()
        #     diff_mask_vis = (5*diff_mask[i]).clip(0, 255).detach().cpu().numpy()
        #     feat_mask_vis = (5*feat_mask[i]).clip(0, 255).detach().cpu().numpy()
        #     cv2.imwrite(args.visual_dir + path, origin_img_vis)
        #     cv2.imwrite(args.visual_dir + path.replace('.png', '_pred.png'), pred_mask_vis)
        #     cv2.imwrite(args.visual_dir + path.replace('.png', '_diff.png'), diff_mask_vis)
        #     cv2.imwrite(args.visual_dir + path.replace('.png', '_feat.png'), feat_mask_vis)

    if dist.get_rank() == 0:
        result = dict()
        result['names'] = img_paths
        result['preds'] = pred_masks
        result['masks'] = gt_masks
        with open('result_diff_uni128_{}.pkl'.format(args.category), 'wb') as f:
            pickle.dump(result, f)
    
    if dist.get_rank() == 0:
        labels = th.cat(labels, dim=0)
        scores = th.cat(scores, dim=0)
        roc = evaluate(labels, scores, metric='roc')
        print('roc: ', roc)
        # for i in range(len(img_paths)):
            # np.savez(img_paths[i], pred_masks[i].cpu().numpy())
        gt_masks = (th.cat(gt_masks, dim=0)/255).long()
        pred_masks = th.cat(pred_masks, dim=0)
        pro = evaluate(gt_masks, pred_masks, metric='pro')
        print('pro: ', pro)
        pproc = evaluate(gt_masks, pred_masks, metric='perpixel_roc')
        print('pproc: ', pproc)

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
        model_path="", alpha_factor=1.0, smooth=False, visual_dir="", use_padim=False, category=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
