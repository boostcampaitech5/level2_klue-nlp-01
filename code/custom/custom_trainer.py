from transformers import Trainer
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from utils.config import load_config


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):

        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32).to(device)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.long).to(device)

        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomTrainer(Trainer):
    """커스텀 된 트레이너를 만드는 클래스입니다."""
    def __init__(self, *args, loss_type=None, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_loss = FocalLoss(1.0, 0.0)

    def compute_loss(self, model, inputs, return_outputs=False):
        """ default loss: ce_loss """

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_type == 'focal':
            loss = self.focal_loss(logits, labels)
        else:
            loss = self.ce_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss