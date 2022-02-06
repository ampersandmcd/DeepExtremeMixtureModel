import pickle
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from model import SpatiotemporalModel, ExtremeTime2, get_device, make_cnn, to_np
from util import NumpyDataset

if __name__ == "__main__":

    """
    This class has the main training loop and stores the best predictions and saves results.

    Parameters:
   
    use_evt - boolean, whether or not to incoporate EVT into mixture model. If false becomes Hurdle baseline
    variable_thresh - boolean, randomly chooses thresholds for each location and window each batch
    main_func - string, density function for non-zero non-execss values. Must be lognormal
    ymax - scalar, max value of y that must be assigned non-zero density by GPD
    
    mean_multiplier - scalar, weight to assign MSE loss term (the other loss term is NLK)
    dropout_multiplier - scalar, weight for dropout regularization in Vandal et al implementation
    quantile - scalar, what quantile to use to define the excess threshold. If variable_thresh is True then the threshold
               determined by quantile will only be used for evaluation purposes while the mixture model's threshold
               will be random.
    max_epochs - int, number of epochs to train
    cnn_params - dictionary, contains parameters for CNN
    lr - scalar, learning rate
    use_mc - boolean, whether or not to use monte carlo dropout. Setting to True is necessary for Vandal et al
    mc_forwards - int, number of monte carlo forward passes to use for Vandal et al
    continuous_evt - boolean, whether to force the mixture model to be continuous -- doesn't work well
    """

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # add model setup options
    parser.add_argument("--model", default="bernoulli-lognormal", type=str, help="Model to train",
                        choices=[
                            "bernoulli-lognormal-gp-variable",  # proposed model with variable threshold
                            "bernoulli-lognormal-gp-fixed",  # proposed model with fixed threshold
                            "bernoulli-lognormal",  # ablation of proposed model with hurdle only
                            "deterministic-cnn",    # ablation of proposed model with deterministic preds
                            "vandal",  # baseline from Vandal et al.
                            "ding",  # baseline from Ding et al.
                            "kong",  # baseline from Kong et al.
                        ])
    parser.add_argument("--n_train", default=450, type=tuple, help="Number of samples to use for training")
    parser.add_argument("--n_val", default=250, type=tuple, help="Number of samples to use for validation")
    # note that n_test is implicitly defined by n - n_train - n_val
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size to train with")
    parser.add_argument("--mean_multiplier", default=0.1, type=float,
                        help="Weight assigned to RMSE loss term (complement is weight assigned to NLL term)")
    parser.add_argument("--dropout_multiplier", default=1e-2, type=float,
                        help="Weight assigned to dropout loss term in Vandal baseline")
    parser.add_argument("--quantile", default=0.6, type=float,
                        help="Quantile used to define excess threshold in proposed model; used only for evaluation if variable threshold model")
    parser.add_argument("--continuous_evt", default=False, type=eval,
                        help="Whether to constrain mixture to be continuous; appealing in theory but performs poorly in practice")

    # add training setup options
    parser.add_argument("--wandb_name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--n_epoch", default=500, type=int, help="Number of epochs")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    args = parser.parse_args()
    args.max_epochs = args.n_epoch

    # configure data
    with open("../data/processed_data.pickle", "rb") as f:
        data = pickle.load(f)
    x, y = data["x"], data["y"]
    train_dataset = NumpyDataset(x[:args.n_train], y[:args.n_train])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1)

    # configure parameters of model backbone (3D CNN or GRU) and spatiotemporal model
    model_params = {
        "ndim": 11,
        "hdim": 10,
        "odim": 6,
        "ksize": (3, 3, 3),
        "padding": (0, 1, 1),
        "bn": False,
        "act": "relu",
        "variable_thresh": False,
        "use_mc": False
    }
    st_params = {
        "use_evt": False,
        "moderate_func": "lognormal",
        "ymax": 250,
        "mean_multiplier": args.mean_multiplier,
        "dropout_multiplier": args.dropout_multiplier,
        "continuous_evt": args.continuous_evt,
        "quantile": args.quantile,
        "variable_thresh": False,
        "use_mc": False,
        "mc_forwards": 0,
        "backbone": "cnn",
        "deterministic": False,
    }

    # tweak parameters depending on model choice
    if args.model == "bernoulli-lognormal-gp-variable":
        model_params["variable_thresh"] = st_params["variable_thresh"] = True
        st_params["use_evt"] = True
    elif args.model == "bernoulli-lognormal-gp-fixed":
        st_params["use_evt"] = True
    elif args.model == "bernoulli-lognormal":
        pass  # no modifications here
    elif args.model == "deterministic-cnn":
        model_params["odim"] = 1
        st_params["deterministic"] = True
    elif args.model == "vandal":
        model_params["use_mc"] = st_params["use_mc"] = True
        st_params["mc_forwards"] = 5
    elif args.model == "ding":
        st_params["backbone"] = "ding"
        st_params["deterministic"] = True
        model_params = {
            "forecast_horizon": 1,
            "ndim": (11, 29, 59),
            "hdim": 10,
            "odim": (1, 29, 59),
            "window_size": 7,
            "memory_dim": 7,
            "context_size": 7
        }
    elif args.model == "kong":
        raise NotImplementedError()
    else:
        raise ValueError()

    # set up model
    device = get_device()
    if st_params["backbone"] == "cnn":
        model = make_cnn(**model_params).to(get_device())
    elif st_params["backbone"] == "ding":
        model = ExtremeTime2(**model_params)
    else:
        raise ValueError()
    st_model = SpatiotemporalModel(model=model, **st_params).to(get_device())

    # run train batch
    i = 0
    optim = torch.optim.Adam(st_model.parameters(), lr=args.lr)
    for batch in train_dataloader:
        st_model.train()
        x = batch["x"].type(torch.FloatTensor).to(device)
        y = batch["y"].type(torch.FloatTensor).to(device)

        # choose threshold
        if st_model.variable_thresh:
            # generate random thresholds in [0.5, 0.95] and augment predictors
            threshes = 0.45 * torch.rand_like(y) + 0.5
            x = torch.cat([x, threshes[:, np.newaxis].repeat(1, 1, x.shape[2], 1, 1)], axis=1)
        else:
            # generate fixed threshold but do not augment predictors
            t = np.nanquantile(to_np(y), st_model.quantile)
            threshes = torch.ones_like(y) * t

        # apply appropriate forward pass (logic for each model type is handled in forward() definition
        pred = st_model(x, threshes, test=False)

        # compute loss and take optimizer step
        loss, nll_loss, rmse_loss = st_model.compute_losses(pred, y, threshes)
        loss.backward()
        optim.step()  # loss[0] is the loss used for training other elements of loss are other
        optim.zero_grad()
        print("*" * 100)
        print(i)
        print("t_loss", loss)
        print("t_nll_loss", nll_loss)
        print("t_rmse_loss", rmse_loss)  # t for train
        i += 1

    # run test batch
    for batch in train_dataloader:
        st_model.eval()
        x = batch["x"].type(torch.FloatTensor).to(device)
        y = batch["y"].type(torch.FloatTensor).to(device)

        # choose threshold
        if st_model.variable_thresh:
            # generate random thresholds in [0.5, 0.95] and augment predictors
            threshes = 0.45 * torch.rand_like(y) + 0.5
            x = torch.cat([x, threshes[:, np.newaxis].repeat(1, 1, x.shape[2], 1, 1)], axis=1)
        else:
            # generate fixed threshold but do not augment predictors
            t = np.nanquantile(to_np(y), st_model.quantile)
            threshes = torch.ones_like(y) * t

        # apply appropriate forward pass (logic for each model type is handled in forward() definition
        pred = st_model(x, threshes, test=True)
        loss, nll_loss, rmse_loss = st_model.compute_losses(pred, y, threshes)
        metrics = st_model.compute_metrics(y, pred, threshes)
        print("*" * 100)
        print("f_loss", loss)
        print("f_nll_loss", nll_loss)
        print("f_rmse_loss", rmse_loss)  # f for final
        print("*" * 100)
        labeled_metrics = {"loss": metrics[0],
                           "nll_loss": metrics[1],
                           "rmse_loss": metrics[2],
                           "zero_brier": metrics[3],
                           "moderate_brier": metrics[4],
                           "excess_brier": metrics[5],
                           "acc": metrics[6],
                           "f1_micro": metrics[7],
                           "f1_macro": metrics[8],
                           "auc_macro_ovo": metrics[9],
                           "auc_macro_ovr": metrics[10]
                           }
        for metric_name, metric_value in labeled_metrics.items():
            print(f"f_{metric_name}", metric_value)
        print("*" * 100)
        break
