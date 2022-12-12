import pytorch_lightning as pl
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torchmetrics.classification import BinaryF1Score, F1Score, BinaryRecall, Recall, BinaryPrecision, Precision
import torchvision
from transformers import AutoTokenizer, AutoModel

from typing import Dict, List, Tuple, Union

def get_image_encoder(name: str) -> Tuple[int, nn.Sequential]:
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        out_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-1]) # Don't include fully connected layer
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
        denominator = numerator.detach().clone().zero_()
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

        loss = torch.mean((self.weight * image2text_loss + (1 - self.weight) * text2image_loss))
        return loss

class ConVIRT(nn.Module):

    def __init__(self, image_encoder: str = 'resnet18', hidden_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.image_out_features, self.image_encoder = get_image_encoder(image_encoder)
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.out_dim = out_dim
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

class ModifiedConVIRTLoss(nn.Module):

    def __init__(self, scale: float = 0.75, temperature: float = 1.0, weight: float = 0.99):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.weight = weight
        self.scale = scale

    def forward(self, image_batch: torch.Tensor, findings_batch: torch.Tensor, impressions_batch):
        findings2impressions_loss = self.contrastive_loss(findings_batch, impressions_batch)
        impressions2findings_loss = self.contrastive_loss(impressions_batch, findings_batch)

        text_batch = torch.mean(torch.stack([findings_batch, impressions_batch], axis=0), axis=0)

        image2text_loss = self.contrastive_loss(image_batch, text_batch)
        text2image_loss = self.contrastive_loss(text_batch, image_batch)

        local_loss = torch.mean((self.weight * findings2impressions_loss + (1 - self.weight) * impressions2findings_loss))
        global_loss = torch.mean((self.weight * image2text_loss + (1 - self.weight) * text2image_loss))
        loss = self.scale * global_loss + (1 - self.scale) * local_loss

        return loss

class ModifiedConVIRT(nn.Module):

    def __init__(self, image_encoder: str = 'resnet18', hidden_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.image_out_features, self.image_encoder = get_image_encoder(image_encoder)
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.out_dim = out_dim
        self.image_projector = nn.Sequential(
            nn.Linear(self.image_out_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

        self.text_projector = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, image_batch: torch.Tensor, findings_batch: torch.Tensor, impressions_batch: torch.Tensor):
        hidden_image_batch = self.image_encoder(image_batch).squeeze(-1).squeeze(-1)
        repr_image_batch = self.image_projector(hidden_image_batch)

        findings_batch = {k: v.squeeze(1) for k, v in findings_batch.items()}
        impressions_batch = {k: v.squeeze(1) for k, v in impressions_batch.items()}

        hidden_findings_batch = self.text_encoder(**findings_batch).pooler_output
        repr_findings_batch = self.text_projector(hidden_findings_batch)

        hidden_impressions_batch = self.text_encoder(**impressions_batch).pooler_output
        repr_impressions_batch = self.text_projector(hidden_impressions_batch)

        return repr_image_batch, repr_findings_batch, repr_impressions_batch

class ModifiedPretrain(pl.LightningModule):
    def __init__(self, model_kwargs: Dict, criterion_kwargs: Dict, optimizer_kwargs: Dict):
        super().__init__()
        self.model = ModifiedConVIRT(**model_kwargs)
        self.criterion = ModifiedConVIRTLoss(**criterion_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.save_hyperparameters()

    def training_step(self, batch: Dict[str, Union[torch.Tensor, Dict]], batch_idx: int) -> Dict:
        image_batch = batch['image']
        findings_batch = batch['findings']
        impressions_batch = batch['impressions']
        repr_image_batch, repr_findings_batch, repr_impressions_batch = self(image_batch, findings_batch, impressions_batch)

        loss = self.criterion(repr_image_batch, repr_findings_batch, repr_impressions_batch)
        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        image_batch = batch['image']
        findings_batch = batch['findings']
        impressions_batch = batch['impressions']
        repr_image_batch, repr_findings_batch, repr_impressions_batch = self(image_batch, findings_batch, impressions_batch)

        loss = self.criterion(repr_image_batch, repr_findings_batch, repr_impressions_batch)
        self.log('val_loss', loss)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: List[dict]):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        self.log('val_loss', avg_loss)

        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), **self.optimizer_kwargs)

    def forward(self, image_batch: torch.Tensor, findings_batch: Dict[str, torch.Tensor], impressions_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(image_batch, findings_batch, impressions_batch)

class Pretrain(pl.LightningModule):
    def __init__(self, model_kwargs: Dict, criterion_kwargs: Dict, optimizer_kwargs: Dict):
        super().__init__()
        self.model = ConVIRT(**model_kwargs)
        self.criterion = ConVIRTLoss(**criterion_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.save_hyperparameters()

    def training_step(self, batch: Dict[str, Union[torch.Tensor, Dict]], batch_idx: int) -> Dict:
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

class Downstream(pl.LightningModule):
    def __init__(self, model_checkpoint: str, optimizer_kwargs: Dict, modified_model: bool = False, num_classes: int = 8, finetune: bool = False):
        super().__init__()
        self.finetune = finetune
        pretrain_module = ModifiedPretrain if modified_model else Pretrain
        convirt_model = pretrain_module.load_from_checkpoint(model_checkpoint).model
        self.image_encoder = convirt_model.image_encoder
        out_dim = convirt_model.out_dim
        del convirt_model
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = 1 if num_classes == 2 else num_classes
        self.f1 = BinaryF1Score() if self.num_classes == 1 else F1Score(task='multiclass', num_classes=self.num_classes, top_k = 1)
        self.recall = BinaryRecall() if self.num_classes == 1 else Recall(task='multiclass', average='macro', num_classes=self.num_classes, top_k = 1)
        self.prec = BinaryPrecision() if self.num_classes == 1 else Precision(task='multiclass', average='macro', num_classes=self.num_classes,  top_k = 1)
        self.classifier = nn.Linear(out_dim, self.num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits if self.num_classes == 1 else F.cross_entropy

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        images = batch['image']
        y = batch['label'].unsqueeze(1).float()
        y = torch.reshape(y, (-1,)).type(torch.LongTensor).to('cuda:0')
        y_hat = self(images)
        loss = self.loss_func(y_hat, y)
        f1 = self.f1(y_hat, y)
        prec = self.prec(y_hat, y)
        recall = self.recall(y_hat, y)
        metrics = {'train_loss': loss, 'train_f1': f1,  'train_prec': prec, 'train_recall': recall}
        self.log_dict(metrics)
        # self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        
        images = batch['image']
        y = batch['label'].unsqueeze(1).float()
        y = torch.reshape(y, (-1,)).type(torch.LongTensor).to('cuda:0')

        y_hat = self(images)
       


        loss = self.loss_func(y_hat, y)
        f1 = self.f1(y_hat, y)
        prec = self.prec(y_hat, y)
        recall = self.recall(y_hat, y)
        metrics = {'val_loss': loss, 'val_f1': f1,  'val_prec': prec, 'val_recall': recall}
        self.log_dict(metrics)
        # self.log('val_loss', loss)

        return {'val_loss': loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        images = batch['image']
        y = batch['label'].unsqueeze(1).float()

        y_hat = self(images)
        loss = self.loss_func(y_hat, y)
        loss = self.loss_func(y_hat, y)
        f1 = self.f1(y_hat, y)
        prec = self.prec(y_hat, y)
        recall = self.recall(y_hat, y)
        metrics = {'test_loss': loss, 'test_f1': f1,  'test_prec': prec, 'test_recall': recall}
        self.log_dict(metrics)
        # self.log('test_loss', loss)

        return {'test_loss': loss}

    def validation_epoch_end(self, outputs: List[dict]):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss}

    def test_epoch_end(self, outputs: List[dict]):
        avg_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        self.log('test_loss', avg_loss)
        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), **self.optimizer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.finetune or self.global_step < 200:
            self.image_encoder.eval()
            with torch.no_grad():
                h = self.image_encoder(x).squeeze(-1).squeeze(-1)
        else:
            h = self.image_encoder(x).squeeze(-1).squeeze(-1)

        logits = self.classifier(h)
        return logits
