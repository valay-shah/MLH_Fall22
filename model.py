import pytorch_lightning as pl
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
import torchvision

from typing import Dict


class TestModel(nn.Module):
    '''TestModel taken from PyTorch CIFAR10 tutorial.
    
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes if num_classes > 2 else 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TestModule(pl.LightningModule):
    def __init__(self, criterion: nn, optimizer: optim, model_kwargs: Dict):
        super().__init__()
        self.model = TestModel(**model_kwargs)
        self.criterion = criterion
        self.optimizer = optimizer(self)
        self.save_hyperparameters()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        x = batch['image']
        y = batch['label'].unsqueeze(1).float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)

        return {'loss': loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        x = batch['image'] 
        y = batch['label'].unsqueeze(1).float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

        return {'test_loss': loss,
                'pred_label': y_hat.max(dim=1)[1].detach().cpu(),
                'label': y.detach().cpu()}

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)