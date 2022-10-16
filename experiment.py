from utility import get_criterion, get_datamodule, get_model, get_optimizer

import torch
from torch import nn, optim, utils
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import yaml

import argparse


def run(args: argparse.Namespace):
    with open(args.experiment, 'r') as f:
        settings = yaml.safe_load(f)


    # Global Configuration
    seed = settings.get('seed', 0) # Random seed

    # W&B Configuration
    wandb_config = settings.get('wandb', dict())
    wandb_project = wandb_config.get('project', 'test-project')
    wandb_dir = wandb_config.get('save_dir', '__pycache__/')

    # Dataset Configuration
    dataset_config = settings.get('dataset')
    dataset_name = dataset_config.get('name')
    dataset_kwargs = dataset_config.get('kwargs', None)

    # Model Configuration
    model_config = settings.get('model')
    model_name = model_config.get('name')
    model_kwargs = model_config.get('kwargs', None)

    criterion_config = model_config.get('criterion')
    criterion_name = criterion_config.get('name')
    criterion_kwargs = criterion_config.get('kwargs', None)

    # Train Configuration
    train_config = settings.get('train')
    max_epochs = train_config.get('max_epochs', 1)
    optimizer_config = train_config.get('optimizer')
    optimizer_name = optimizer_config.get('name')
    optimizer_kwargs = optimizer_config.get('kwargs', None)

    # Setup W&B
    wandb_logger = WandbLogger(
        project=wandb_project,
        save_dir=wandb_dir)

    # Initialize Lightning Module and Data Module
    if dataset_kwargs is not None:
        datamodule = get_datamodule(dataset_name, **dataset_kwargs)
    else:
        datamodule = get_datamodule(dataset_name)

    criterion = get_criterion(criterion_name, **criterion_kwargs)
    optimizer = get_optimizer(optimizer_name, **optimizer_kwargs)
    module_kwargs = {
        'criterion': criterion,
        'optimizer': optimizer,
        'model_kwargs': model_kwargs
    }
    model = get_model(model_name, **module_kwargs)

    # Train Model
    trainer = pl.Trainer(
        limit_train_batches=100, 
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto', 
        logger=wandb_logger)

    trainer.fit(model, datamodule)

    # Evaluate Model
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