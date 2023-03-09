""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
import cv2 
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)

def evaluate(labels, scores, metric='roc'):
    labels = labels.cpu()
    scores = scores.cpu()
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    elif metric == 'perpixel_roc':
        return perpixel_roc(scores, labels)
    elif metric == 'pro':
        return pro(scores, labels)
    else:
        raise NotImplementedError("Check the evaluation metric.")

def perpixel_roc(anom, gt):
    ''' input anom: tensor(NxHxW)
    '''
    assert anom.size() == gt.size()

    labels = torch.flatten(gt)
    scores = torch.flatten(anom)
    return roc(labels, scores)

def pro(anom, gt):
    ''' input anom: tensor(NxHxW)
    '''
    assert anom.size() == gt.size()
    num_imgs = anom.size(0)

    labels = torch.flatten(gt)
    scores = torch.flatten(anom)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    valid_index = fpr <= 0.3

    valid_threshold = thresholds[valid_index]
    valid_fpr = fpr[valid_index]

    if len(valid_fpr) > 1000:
        sample_rate = len(valid_fpr) // 1000
        valid_index = np.arange(len(valid_fpr), step=sample_rate)
        valid_threshold = valid_threshold[valid_index]
        valid_fpr = valid_fpr[valid_index]


    # get connected components of gt
    # range should be (0, 255)
    gt_tensors = (255*gt).cpu().numpy().astype(np.uint8)
    gt_comps = []

    for i in range(gt_tensors.shape[0]):
        num, gt_ccs = cv2.connectedComponents(gt_tensors[i])
        gt_comps.append(gt_ccs)

    thr_1 = valid_threshold[-1]
    thr_2 = valid_threshold[len(valid_threshold)//2]

    pros = np.zeros_like(valid_fpr)
    
    for thr_idx, thr in enumerate(valid_threshold):
        avg_coverage = []
        for i in range(num_imgs):
            ai, gi = anom[i].numpy(), gt_comps[i]

            # binary ai
            bi = ai >= thr

            # if g_range is 0, the image has no defects
            g_range = gi.max()
            for cnum in range(1, g_range+1):
                region = gi == cnum
                overlap = np.logical_and(region, bi)
                coverage = overlap.astype(np.float32).sum() / region.astype(np.float32).sum()
                avg_coverage.append(coverage)
        avg_coverage = np.array(avg_coverage).mean()
        pros[thr_idx] = avg_coverage


    # integral across fpr from 0. to 0.3
    # NOTE: threhsold returned by roc_curve is decreasing

    # trapezoid area
    start_fpr = 0.
    start_pro = 0.
    I = 0.

    for i in range(len(pros)):
        stop_fpr = valid_fpr[i]
        stop_pro = pros[i]

        d_fpr = stop_fpr-start_fpr

        I += 0.5 * d_fpr * (start_pro + stop_pro)
        start_fpr = stop_fpr
        start_pro = stop_pro

    stop_fpr = 0.3
    # integral to 0.3
    I += (stop_fpr-start_fpr)*stop_pro

    saveto = './'
    if saveto:
        plt.figure()
        plt.plot(valid_fpr, pros)
        plt.xlim([0.0, 0.35])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('Overlap')
        plt.title('PRO curve')
        #plt.show()
        plt.savefig(os.path.join(saveto, "PRO.pdf"))
        plt.close()

        plt.figure()
        plt.plot(valid_fpr, valid_threshold)
        plt.xlim([0.0, 0.35])
        plt.xlabel('False Positive Rate')
        plt.ylabel('Threshold')
        plt.title('Threshold curve')
        #plt.show()
        plt.savefig(os.path.join(saveto, "FPR.pdf"))
        plt.close()

    return I / 0.3

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    saveto = './'
    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap


if __name__ == '__main__':
    mask = np.zeros((2, 256, 256))
    mask[0][0:128, 0:128] = 1.0
    anom = np.zeros((2, 256, 256))

    anom[0][:, 0:128] = 0.5

    mask = torch.from_numpy(mask)
    anom = torch.from_numpy(anom)

    pp_roc = evaluate(mask, anom, 'perpixel_roc')
    pro = evaluate(mask, anom, 'pro')
    
    print('perpixel_roc: ', pp_roc)
    print('pro: ', pro)



