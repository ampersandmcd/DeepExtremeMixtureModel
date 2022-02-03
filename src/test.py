import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import wandb
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from model import SpatiotemporalLightningModule
from util import NumpyDataset


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--name", default="1z1h5m58", type=str, help="Name of wandb run")
    args = parser.parse_args()

    # wandb logging
    wandb.init(id=args.name, project="demm", resume="must")
    api = wandb.Api()
    run = api.run(f"andrewmcdonald/demm/{args.name}")

    # configure data
    with open("../data/processed_data.pickle", "rb") as f:
        data = pickle.load(f)
    x, y = data["x"], data["y"]

    # configure dataloaders
    test_dataset = NumpyDataset(x[run.config["n_train"] + run.config["n_val"]:], y[run.config["n_train"] + run.config["n_val"]:])
    test_dataloader = DataLoader(test_dataset, batch_size=run.config["batch_size"], num_workers=4)

    # load model from best checkpoint
    model = None
    checkpoint_folder = f"./demm/{args.name}/checkpoints"
    checkpoints = os.listdir(checkpoint_folder)
    best_epoch = 0
    best_checkpoint = None
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("=")[1].split("-")[0])
        if epoch > best_epoch:
            best_epoch = epoch
            best_checkpoint = f"{checkpoint_folder}/{checkpoint}"

    lightning_module = SpatiotemporalLightningModule.load_from_checkpoint(best_checkpoint)

    # wandb logging
    wandb_logger = pl.loggers.WandbLogger(project="demm")
    wandb_logger.watch(lightning_module, log="all", log_freq=10)

    # trainer configuration
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger

    # test
    trainer.test(lightning_module, test_dataloader)
