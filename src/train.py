import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import wandb
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import model as m
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
    n_epoch - int, number of epochs to train
    cnn_params - dictionary, contains parameters for CNN
    lr - scalar, learning rate
    use_mc - boolean, whether or not to use monte carlo dropout. Setting to True is necessary for Vandal et al
    mc_forwards - int, number of monte carlo forward passes to use for Vandal et al
    continuous_evt - boolean, whether to force the mixture model to be continuous -- doesn't work well
    """

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--wandb_name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--model", default="bernoulli-lognormal-gp-variable", type=str, help="Model to train",
                        choices=[
                            "bernoulli-lognormal-gp-variable",  # proposed model with variable threshold
                            "bernoulli-lognormal-gp-fixed",     # proposed model with fixed threshold
                            "bernoulli-lognormal",              # ablation of proposed model with hurdle only
                            "vandal",                           # baseline from Vandal et al.
                            "ding",                             # baseline from Ding et al.
                            "kong",                             # baseline from Kong et al.
                        ])
    parser.add_argument("--n_train", default=450, type=tuple, help="Number of samples to use for training")
    parser.add_argument("--n_val", default=250, type=tuple, help="Number of samples to use for validation")
    # note that n_test is implicitly defined by n - n_train - n_val
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size to train with")
    parser.add_argument("--mean_multiplier", default=1/60, type=float, help="Weight assigned to MSE loss term (complement is weight assigned to NLL term)")
    parser.add_argument("--dropout_multiplier", default=1e-2, type=float, help="Weight assigned to dropout loss term in Vandal baseline")
    parser.add_argument("--quantile", default=0.6, type=float, help="Quantile used to define excess threshold in proposed model; used only for evaluation if variable threshold model")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--n_epoch", default=50, type=int, help="Number of epochs to train for")
    parser.add_argument("--continuous_evt", default=False, type=eval, help="Whether to constrain mixture to be continuous; appealing in theory but performs poorly in practice")
    args = parser.parse_args()

    # configure data
    with open("../data/processed_data.pickle", "rb") as f:
        data = pickle.load(f)
    x, y = data["x"], data["y"]
    with open('../data/rand_thresholds.pickle', 'rb') as f:
        rand_threshes = pickle.load(f).reshape(y.shape)  # random threshes have same shape as target
    train_dataset = NumpyDataset(x[:args.n_train], y[:args.n_train])
    val_dataset = NumpyDataset(x[args.n_train:args.n_train + args.n_val], y[args.n_train:args.n_train + args.n_val])
    test_dataset = NumpyDataset(x[args.n_train + args.n_val:], y[args.n_train + args.n_val:])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # configure model with standard CNN backbone and mixture model wrapper
    model = None
    cnn_params = {"ndim": 11, "hdim": 10, "odim": 6, "ksize": (3, 3, 3), "padding": (0, 1, 1),
                  "bn": False, "act": "relu", "variable_thresh": False, "use_mc": False}
    if args.model == "bernoulli-lognormal-gp-variable":
        cnn_params["variable_thresh"] = True
        cnn = m.make_cnn(**cnn_params).to(m.get_device())
        model = m.SpatiotemporalModel(model=cnn, use_evt=True, moderate_func="lognormal", ymax=250,
                                      mean_multiplier=args.mean_multiplier, dropout_multiplier=args.dropout_multiplier,
                                      continuous_evt=args.continuous_evt)
    elif args.model == "bernoulli-lognormal-gp-fixed":
        cnn = m.make_cnn(**cnn_params).to(m.get_device())
        model = m.SpatiotemporalModel(model=cnn, use_evt=True, moderate_func="lognormal", ymax=250,
                                      mean_multiplier=args.mean_multiplier, dropout_multiplier=args.dropout_multiplier,
                                      continuous_evt=args.continuous_evt)
    elif args.model == "bernoulli-lognormal":
        cnn = m.make_cnn(**cnn_params).to(m.get_device())
        model = m.SpatiotemporalModel(model=cnn, use_evt=False, moderate_func="lognormal", ymax=250,
                                      mean_multiplier=args.mean_multiplier, dropout_multiplier=args.dropout_multiplier,
                                      continuous_evt=args.continuous_evt)
    elif args.model == "vandal":
        cnn_params["use_mc"] = True
        cnn = m.make_cnn(**cnn_params).to(m.get_device())
        model = m.SpatiotemporalModel(model=cnn, use_evt=False, moderate_func="lognormal", ymax=250,
                                      mean_multiplier=args.mean_multiplier, dropout_multiplier=args.dropout_multiplier,
                                      continuous_evt=args.continuous_evt)
    elif args.model == "ding":
        raise NotImplementedError()
    elif args.model == "kong":
        raise NotImplementedError()
    else:
        raise ValueError()

    # wandb logging
    wandb.init(project="demm")
    if args.wandb_name != "default":
        wandb.run.name = args.wandb_name  # continue logging on previous run
    wandb_logger = pl.loggers.WandbLogger(project="demm")
    wandb_logger.watch(model, log="all", log_freq=10)
    wandb_logger.experiment.config.update(args)

    # trainer configuration
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="v_loss"))

    # train
    trainer.fit(model, train_dataloader, val_dataloader)

    # test
    model.eval()
    nlls, nans, infs = [], [], []
    for batch in test_dataloader:
        x = batch["x"].type(torch.FloatTensor).to(model.device)
        criterion = model.get_criterion()

        # perform forward pass
        if model.conditional_noise:
            std_min, std_max = 1e-3, 5e-3  # set to near-zero noise for validation
            noise = (std_max - std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + std_min
            noisy_x = x + torch.randn_like(x) * noise
            z, delta_logp = model.forward(noisy_x, noise_level=noise)
        else:
            z, delta_logp = model.forward(x)

        # check nans and infs
        nan_idx = torch.all(torch.logical_or(z != z, delta_logp != delta_logp), dim=1)
        inf_idx = torch.all(torch.logical_not(torch.logical_and(torch.isfinite(z), torch.isfinite(delta_logp))), dim=1)
        keep_idx = torch.logical_not(torch.logical_or(nan_idx, inf_idx))
        z, delta_logp = z[keep_idx], delta_logp[keep_idx]

        # compute and save nll for mean computation
        nll = criterion(z, delta_logp)
        nlls.append(nll)

        # save nan/inf count for later logging
        nans.append(sum(nan_idx))
        infs.append(sum(inf_idx))

    # save and log test nll
    test_nll = sum(nlls) / len(test_dataset)
    wandb.log({"test_nll": test_nll})

    # save and log total number of nan/inf
    wandb.log({"test_nan": sum(nans)})
    wandb.log({"test_inf": sum(infs)})


    # save and log image of samples
    x = model.sample(args.n_test_samples).detach().cpu().numpy()
    util.pairplot(x, title=None, color=data.color)
    wandb.log({"final_pairplot": wandb.Image(plt)})
    plt.close()
