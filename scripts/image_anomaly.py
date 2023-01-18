"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import torch as th
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

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_flip=False,
        anomaly=True,
        infinte_loop=False,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    anom_metrics = {'roc': []}
    labels = []
    scores = []
    pred_masks = []
    gt_masks = []
    idx = 0 
    for batch, gt_mask, model_kwargs in data:
        anom_gt = model_kwargs.pop('anom_gt')
        labels.append(anom_gt)
        gt_masks.append(gt_mask)

        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        scores.append(minibatch_metrics['total_bpd'])
        pred_masks.append(minibatch_metrics['pred_mask'])

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.log(f"done {num_complete} samples: bpd={total_bpd.item()}")
    
    if dist.get_rank() == 0:
        labels = th.cat(labels, dim=0)
        scores = th.cat(scores, dim=0)
        roc = evaluate(labels, scores, metric='roc')
        print('roc: ', roc)
        gt_masks = (th.cat(gt_masks, dim=0)/255).long()
        pred_masks = th.cat(pred_masks, dim=0)
        pro = evaluate(gt_masks, pred_masks, metric='pro')
        print('pro: ', pro)
        pproc = evaluate(gt_masks, pred_masks, metric='perpixel_roc')
        print('pproc: ', pproc)

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
