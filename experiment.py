from utility import get_datamodule, get_model

import torch
from torch import nn, optim, utils
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import yaml

import argparse


def run(args: argparse.Namespace):
    with open(args.experiment, 'r') as f:
        settings = yaml.safe_load(f)

    wandb_logger = WandbLogger(project="test-project")

    # Global Configuration
    seed = settings.get('seed', 0) # Random seed

    # Dataset Configuration
    dataset_config = settings.get('dataset')
    dataset_name = dataset_config.get('name')
    dataset_kwargs = dataset_config.get('kwargs')
    # Model Configuration
    model_config = settings.get('model')
    model_name = model_config.get('name')
    model_kwargs = model_config.get('kwargs')

    # Setup
    datamodule = get_datamodule(dataset_name, **dataset_kwargs)
    model = get_model(model_name, **model_kwargs)

    # Train
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run an experiment.')
    parser.add_argument(
        '--experiment',
        required=True,
        type=str,
        help='path to experiment file (YAML)',
        metavar='EXPR',
        dest='experiment')

    args = parser.parse_args()
    run(args)