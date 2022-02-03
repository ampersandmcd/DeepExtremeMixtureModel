import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import wandb
from pytorch_lightning import Callback
from torch.utils.data import Dataset

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import model as m


class NumpyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


def split_var(x):
    """
    Weird little function that I'm pretty sure I needed
    to make for debugging purposes. Keeping it just in case
    """
    splitted_var = torch.zeros_like(x, device=x.device)
    return x + splitted_var


def compute_class_labels(y, threshes):
    """
    Creates an array of class labels for the target.
    0 means the sample is 0
    1 means the sample is non-zero non-excess
    2 means the sample is excess
    """
    y, threshes = m.to_np(y), m.to_np(threshes)
    labels = np.ones_like(y)
    nans = np.isnan(y)
    y[nans] = 0
    labels[y == 0.] = 0
    labels[y > threshes] = 2
    labels[nans] = np.nan
    return labels


def no_nans(a, b):
    """
    Returns a mask which is true only at indices where
    both a and b are non-nan.
    """
    return (~np.isnan(a)) & (~np.isnan(b))


def pearsonr(a, b):
    """
    Computes pearson correlation between two tensors
    """
    a, b = m.to_np(a), m.to_np(b)
    mask = no_nans(a, b)
    return pearsonr(a.flatten(), b.flatten())[0]


def accuracy(a, b):
    """
    Computes portion of non-nan values where a and b match
    """
    a, b = m.to_np(a), m.to_np(b)
    nonan_mask = no_nans(a, b)
    return np.mean((a == b)[nonan_mask])


def f1(tru, pred):
    """
    Computes f1 micro and macro
    """
    tru, pred = m.to_np(tru), m.to_np(pred)
    nonan_mask = no_nans(tru, pred)
    micro = f1_score(tru[nonan_mask].flatten(), pred[nonan_mask].flatten(), average='micro')
    macro = f1_score(tru[nonan_mask].flatten(), pred[nonan_mask].flatten(), average='macro')
    return micro, macro


def auc(tru_labels, pred_probs):
    """
    Computes one versus one and one versus rest auc
    """
    tru_labels, pred_probs = m.to_np(tru_labels), m.to_np(pred_probs)
    tru_labels = tru_labels.flatten()
    nonans_mask = ~np.isnan(tru_labels)
    pred_probs = pred_probs.reshape([pred_probs.shape[0], -1]).transpose()

    ovo = roc_auc_score(tru_labels[nonans_mask], pred_probs[nonans_mask], average='macro', multi_class='ovo')
    ovr = roc_auc_score(tru_labels[nonans_mask], pred_probs[nonans_mask], average='macro', multi_class='ovr')
    return ovo, ovr


def brier_score(x, y):
    """
    Computes brier score (i.e. MSE)
    """
    return np.nanmean(np.square(x - y))


def to_stats(y, use_evt=True):
    """
    Splits up a tensor y into multiple pieces representing the different parts of the mixture model
    y[:, :2] is the probability of zero rainfall and probability of excess rainfall
    y[:, 2:4] is the GPD statistics
    y[:, 4:6] is the lognormal statistics
    """
    if use_evt:
        return y[:, :2], y[:, 2:4], y[:, 4:6]
    else:
        return y[:, :2], y[:, 2:4]
