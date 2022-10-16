from utility import get_criterion, get_datamodule, get_model, get_optimizer

import torch
from torch import nn, optim, utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import yaml

import argparse
import os


def run(args: argparse.Namespace):
    with open(args.experiment, 'r') as f:
        settings = yaml.safe_load(f)


    # Global Configuration
    experiment_name = settings.get('name')
    seed = settings.get('seed', 0)
    debug = settings.get('debug', False)

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
    model_dir_path = model_config.get('dir_path', 'checkpoints/')

    criterion_config = model_config.get('criterion')
    criterion_name = criterion_config.get('name')
    criterion_kwargs = criterion_config.get('kwargs', None)

    # Train Configuration
    train_config = settings.get('train')
    max_epochs = train_config.get('max_epochs', 1)
    optimizer_config = train_config.get('optimizer')
    optimizer_name = optimizer_config.get('name')
    optimizer_kwargs = optimizer_config.get('kwargs', None)

    # Setup Reproducibility & Debugging
    pl.seed_everything(seed, workers=True)
    limit_train_batches = 0.05 if debug else 1
    limit_val_batches = 0.05 if debug else 1
    limit_test_batches = 0.05 if debug else 1

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

    # Initialize Callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=2, verbose=False, mode='min')
    dirpath = os.path.join(model_dir_path, experiment_name)
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss', 
        save_top_k=1, 
        mode='min',
        dirpath=dirpath,
        filename='{epoch}-{val_loss:.2f}')

    # Train Model
    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        callbacks=[model_checkpoint, early_stop_callback],
        deterministic=True,
        logger=wandb_logger)

    trainer.fit(model, datamodule=datamodule)

    # Evaluate Model
    # trainer.test(model, datamodule)


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