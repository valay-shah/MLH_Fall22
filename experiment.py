from dataset import PretrainDataModule, DownstreamDataModule
from model import Pretrain, Downstream

import torch
from torch import nn, optim, utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
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
    mode = settings.get('mode')
    batch_size = settings.get('batch_size', 32)
    num_workers = settings.get('num_workers', 4)
    checkpoint_path = settings.get('checkpoint_path', 'checkpoints/')
    root_dir = settings.get('root_dir', 'data/')

    # Model Configuration
    model_kwargs = settings.get('model_kwargs', None)

    # Criterion Configuration
    criterion_kwargs = settings.get('criterion_kwargs', None)

    # Optimizer Configuration
    optimizer_kwargs = settings.get('optimizer_kwargs', None)

    # Train Configuration
    train_config = settings.get('train', dict())
    max_epochs = train_config.get('max_epochs', 1)

    # Downtream Configuration
    downstream_config = settings.get('downstream', dict())
    max_epochs = downstream_config.get('max_epochs', 1)
    finetune = downstream_config.get('finetune', True)
    dataset = downstream_config.get('dataset')

    # Setup Reproducibility & Debugging
    pl.seed_everything(seed, workers=True)
    limit_train_batches = 100 if debug else 1.0
    limit_val_batches = 100 if debug else 1.0
    limit_test_batches = 100 if debug else 1.0

    # Setup W&B
    wandb_dir = '__pycache__/'
    wandb_logger = WandbLogger(
        project=experiment_name,
        save_dir=wandb_dir)

    # Initialize Lightning Module and Data Module
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' # https://github.com/huggingface/transformers/issues/5486

    if mode == 'pretrain':
        model = Pretrain(
            model_kwargs=model_kwargs,
            criterion_kwargs=criterion_kwargs,
            optimizer_kwargs=optimizer_kwargs)
        mimic_cxr_root_dir = '/vast/vs2393/mlh_dataset/' # root_dir of MIMIC_CXR data is in /vast (Comment out to inherit otherwise)
        datamodule = PretrainDataModule(
            root_dir=mimic_cxr_root_dir,
            batch_size=batch_size, 
            num_workers=num_workers)

    elif mode == 'downstream':
        model_checkpoint = os.path.join(checkpoint_path, experiment_name, 'pretrain.ckpt')
        model = Downstream(
            model_checkpoint=model_checkpoint,
            finetune=finetune,
            optimizer_kwargs=optimizer_kwargs)
        datamodule = DownstreamDataModule(
            dataset=dataset,
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers)
    elif mode == 'evaluate':
        raise NotImplementedError(mode)
    else:
        raise ValueError(f'Unknown mode {mode}')

    # Initialize Callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=2, verbose=False, mode='min')
    dirpath = os.path.join(checkpoint_path, experiment_name)
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1, 
        mode='min',
        dirpath=dirpath,
        filename=mode)
    lr_monitor = LearningRateMonitor()

    # Train Model
    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
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
