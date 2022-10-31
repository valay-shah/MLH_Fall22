import pytorch_lightning as pl
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
import torchvision
from transformers import AutoTokenizer, AutoModel


from typing import Dict, List, Tuple, Union

def get_image_encoder(name: str) -> Tuple[int, nn.Sequential]:
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        out_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-1]) # don't include fully connected layer
        return out_features, model
    else:
        raise ValueError(f'Unknown model {name}')

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, v: torch.Tensor, u: torch.Tensor):
        '''
        v is fixed sample
        '''
        batch_size = v.shape[0]
        numerator = torch.exp(F.cosine_similarity(v, u) / self.temperature)
        denominator = numerator.detach().clone()
        for i in range(denominator.shape[0]):
            for j in range(batch_size):
                denominator[i] += torch.exp(F.cosine_similarity(v[i], u[j], dim=0) / self.temperature)
        loss = -1 * torch.log(numerator / denominator)
        return loss

class ConVIRTLoss(nn.Module):

    def __init__(self, temperature: float = 1.0, weight: float = 0.99):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.weight = weight

    def forward(self, image_batch: torch.Tensor, text_batch: torch.Tensor):
        image2text_loss = self.contrastive_loss(image_batch, text_batch)
        text2image_loss = self.contrastive_loss(text_batch, image_batch)

        loss = torch.mean((self.weight * image2text_loss * (1 - self.weight) * text2image_loss))
        return loss

class ConVIRT(nn.Module):

    def __init__(self, image_encoder: str = 'resnet18', hidden_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.image_out_features, self.image_encoder = get_image_encoder(image_encoder)
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        self.image_projector = nn.Sequential(
            nn.Linear(self.image_out_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

        self.text_projector = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, image_batch: torch.Tensor, text_batch: torch.Tensor):
        hidden_image_batch = self.image_encoder(image_batch).squeeze(-1).squeeze(-1)
        repr_image_batch = self.image_projector(hidden_image_batch)

        text_batch = {k: v.squeeze(1) for k, v in text_batch.items()}
        hidden_text_batch = self.text_encoder(**text_batch).pooler_output
        repr_text_batch = self.text_projector(hidden_text_batch)

        return repr_image_batch, repr_text_batch

class Pretrain(pl.LightningModule):
    def __init__(self, model_kwargs: Dict, criterion_kwargs: Dict, optimizer_kwargs: Dict):
        super().__init__()
        self.model = ConVIRT(**model_kwargs)
        self.criterion = ConVIRTLoss(**criterion_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.save_hyperparameters()

    def training_step(self, batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], batch_idx: int) -> Dict:
        image_batch = batch['image']
        text_batch = batch['report']

        repr_image_batch, repr_text_batch = self(image_batch, text_batch)
        loss = self.criterion(repr_image_batch, repr_text_batch)
        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        image_batch = batch['image']
        text_batch = batch['report']

        repr_image_batch, repr_text_batch = self(image_batch, text_batch)
        loss = self.criterion(repr_image_batch, repr_text_batch)
        self.log('val_loss', loss)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: List[dict]):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        self.log('val_loss', avg_loss)

        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), **self.optimizer_kwargs)

    def forward(self, image_batch: torch.Tensor, text_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(image_batch, text_batch)
