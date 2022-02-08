import pickle
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import wandb
from model import SpatiotemporalLightningModule
from util import get_device, NumpyDataset

if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--name", default="1z1h5m58", type=str, help="Name of wandb run")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")

    # wandb logging
    wandb.init(id=args.name, project="demm", resume="must")
    api = wandb.Api()
    run = api.run(f"andrewmcdonald/demm/{args.name}")

    # configure data
    with open("../data/subx/processed_data.pickle", "rb") as f:
        data = pickle.load(f)
    x, y = data["x"], data["y"]

    # configure dataloaders
    test_dataset = NumpyDataset(x[run.config["n_train"] + run.config["n_val"]:], y[run.config["n_train"] + run.config["n_val"]:])
    test_dataloader = DataLoader(test_dataset, batch_size=run.config["batch_size"], num_workers=4)

    # load model from best checkpoint
    best_model_path = run.config["best_model_path"]
    lightning_module = SpatiotemporalLightningModule.load_from_checkpoint(best_model_path)
    lightning_module.to(device=get_device(), dtype=torch.float)

    # wandb logging
    wandb_logger = pl.loggers.WandbLogger(project="demm")
    wandb_logger.watch(lightning_module, log="all", log_freq=10)

    # trainer configuration
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger

    # test
    print(f"Starting testing with {best_model_path}.")
    trainer.test(lightning_module, test_dataloader)
    print(f"Done testing.")
