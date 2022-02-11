import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

import argparse
import torch
import pytorch_lightning
from torch.utils.data import DataLoader

import os
import sys

from src.model import SpatiotemporalLightningModule, torch_rmse
from src.util import NumpyDataset, to_np, to_item, get_device


def evaluate_extremes(seed):
    print(f"Evaluating seed {seed}")

    # prepare data
    args = argparse.Namespace(seed=seed, n_train=731, n_val=104, batch_size=104)
    x, y, rand_inds = data["x"], data["y"], data["rand_inds"]
    x = x[rand_inds[:, args.seed]]
    y = y[rand_inds[:, args.seed]]
    original_x = np.copy(x)
    x = np.log(x + 1)
    mu_x, sigma_x = np.nanmean(x[:args.n_train + args.n_val]), np.nanstd(x[:args.n_train + args.n_val])
    x = (x - mu_x) / sigma_x

    # keep only test set data
    original_x, x, y = original_x[args.n_train + args.n_val:], x[args.n_train + args.n_val:], y[args.n_train + args.n_val:]
    test_dataset = NumpyDataset(x, y)

    # load models trained on seed 1
    st_modules = {
        "bernoulli-lognormal": SpatiotemporalLightningModule.load_from_checkpoint("subx_models/b-l.ckpt"),
        "bernoulli-lognormal-gp-fixed": SpatiotemporalLightningModule.load_from_checkpoint("subx_models/b-l-g-f.ckpt"),
        "bernoulli-lognormal-gp-variable": SpatiotemporalLightningModule.load_from_checkpoint("subx_models/b-l-g-v.ckpt"),
        "deterministic-cnn": SpatiotemporalLightningModule.load_from_checkpoint("subx_models/d-cnn.ckpt"),
        "vandal": SpatiotemporalLightningModule.load_from_checkpoint("subx_models/vandal.ckpt"),
        "ding": SpatiotemporalLightningModule.load_from_checkpoint("subx_models/ding.ckpt")
    }

    predictions = {}
    threshes = None

    for name, st_module in st_modules.items():
        st_module.st_model.eval()
        st_module.st_model.type(torch.FloatTensor)
        x = torch.from_numpy(test_dataset.x).type(torch.FloatTensor)
        y = torch.from_numpy(test_dataset.y).type(torch.FloatTensor)

        # choose threshold
        if st_module.st_model.variable_thresh:
            # fix threshold at test time and augment predictors
            t = np.nanquantile(to_np(y), st_module.st_model.quantile)
            threshes = torch.ones_like(y) * t
            x = torch.cat([x, threshes[:, np.newaxis].repeat(1, 1, x.shape[2], 1, 1)], axis=1)
        else:
            # generate fixed threshold but do not augment predictors
            t = np.nanquantile(to_np(y), st_module.st_model.quantile)
            threshes = torch.ones_like(y) * t

        # apply appropriate forward pass (logic for each model type is handled in forward() definition
        pred = st_module.st_model(x, threshes, test=True)
        predictions[name] = pred

    # compute ensemble mean prediction on original dataset
    # (b, n, t, h, w) -> (b, h, w), fixing t=7, collapsing n
    mean = np.nanmean(original_x[:, :, [-1]], axis=1)
    predictions["mean"] = torch.from_numpy(mean).type(torch.FloatTensor).to(get_device())

    # compute masks and basic stats
    extreme_mask = torch.where(y > threshes, 1.0, np.float32(np.nan))
    zero_mask = torch.where(y == 0, 1.0, np.float32(np.nan))
    moderate_mask = torch.where(torch.logical_and(torch.isnan(extreme_mask), torch.isnan(zero_mask)), 1.0, np.nan)
    n_extreme = torch.nansum(extreme_mask)
    n_zero = torch.nansum(zero_mask)
    n_moderate = torch.nansum(moderate_mask)
    p_extreme = to_item(n_extreme / (n_extreme + n_zero + n_moderate))
    p_zero = to_item(n_zero / (n_extreme + n_zero + n_moderate))
    p_moderate = to_item(n_moderate / (n_extreme + n_zero + n_moderate))

    extreme_predictions = {}
    zero_predictions = {}
    moderate_predictions = {}
    for name, pred in predictions.items():
        if isinstance(pred, tuple):
            extreme_pred = []
            zero_pred = []
            moderate_pred = []
            for p in pred:
                extreme_pred.append(extreme_mask.unsqueeze(1) * p)
                zero_pred.append(zero_mask.unsqueeze(1) * p)
                moderate_pred.append(moderate_mask.unsqueeze(1) * p)
            extreme_pred = tuple(extreme_pred)
            zero_pred = tuple(zero_pred)
            moderate_pred = tuple(moderate_pred)
        else:
            extreme_pred = extreme_mask * pred
            zero_pred = zero_mask * pred
            moderate_pred = moderate_mask * pred
        extreme_predictions[name] = extreme_pred
        zero_predictions[name] = zero_pred
        moderate_predictions[name] = moderate_pred

    losses = []
    for name in predictions.keys():
        model = name
        if name == "mean":
            # use d-cnn pointwise/deterministic structure to compute losses
            model = "deterministic-cnn"
        zero_loss, zero_nll_loss, zero_rmse_loss = to_item(
            st_modules[model].st_model.compute_losses(pred=zero_predictions[name], y=zero_mask * y, threshes=threshes))
        moderate_loss, moderate_nll_loss, moderate_rmse_loss = to_item(
            st_modules[model].st_model.compute_losses(pred=moderate_predictions[name], y=moderate_mask * y, threshes=threshes))
        extreme_loss, extreme_nll_loss, extreme_rmse_loss = to_item(
            st_modules[model].st_model.compute_losses(pred=extreme_predictions[name], y=extreme_mask * y, threshes=threshes))
        metrics = to_item(st_modules[model].st_model.compute_metrics(y=y, pred=predictions[name], threshes=threshes))
        losses.append({
            "name": name,
            "loss": metrics[0],
            "nll_loss": metrics[1],
            "rmse_loss": metrics[2],
            "zero_brier": metrics[3],
            "moderate_brier": metrics[4],
            "excess_brier": metrics[5],
            "acc": metrics[6],
            "f1_micro": metrics[7],
            "f1_macro": metrics[8],
            "auc_macro_ovo": metrics[9],
            "auc_macro_ovr": metrics[10],
            "zero_nll_loss": zero_nll_loss,
            "zero_rmse_loss": zero_rmse_loss,
            "moderate_nll_loss": moderate_nll_loss,
            "moderate_rmse_loss": moderate_rmse_loss,
            "extreme_nll_loss": extreme_nll_loss,
            "extreme_rmse_loss": extreme_rmse_loss,
            "p_zero": p_zero,
            "p_moderate": p_moderate,
            "p_extreme": p_extreme,
            "seed": seed
        })
        print("*" * 100)
        print(name)
        print(losses[-1])
        print("*" * 100)

    return losses


if __name__ == "__main__":

    with open("subx/all_data.pickle", "rb") as f:
        data = pickle.load(f)

    losses = []
    for seed in range(1, 6):
        loss = evaluate_extremes(seed)
        losses.extend(loss)

    losses_df = pd.DataFrame(losses)
    losses_df.to_csv("results/subx_partition.csv", index=False)
