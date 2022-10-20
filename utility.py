from dataset import MURADataModule
from model import TestModule

import pytorch_lightning as pl
import torch
from torch import nn, optim, utils
import torch.nn.functional as F

def get_datamodule(name: str, **kwargs) -> pl.LightningDataModule:
    if name == 'MURA':
        return MURADataModule(**kwargs)
    else:
        raise NameError(f'Unknown datamodule: {name}')

def get_model(name: str, **kwargs) -> pl.LightningModule:
    if name == 'test':
        return TestModule(**kwargs)
    else:
        raise NameError(f'Unknown model: {name}')

def get_criterion(name: str, **kwargs) -> nn:
    if name == 'bce_logits':
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise NameError(f'Unknown criterion: {name}')

# TODO: refactor so it can be tracked in W&B
def get_optimizer(name: str, **kwargs) -> optim:
    if name == 'adam':
        return lambda model: optim.Adam(model.parameters(), **kwargs)
    else:
        raise NameError(f'Unknown optimizer: {optimizer}')
