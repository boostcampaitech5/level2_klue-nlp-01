from transformers import Trainer
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from utils.config import load_config


class FocalLoss(nn.Module):
    """ 
    하이퍼파라미터 alpha, gamma 값을 받아 Focal Loss를 반환합니다.
    reduction='mean'이 default이며 배치 loss들의 평균을 계산합니다.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', device=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device=device

    def forward(self, input, target):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32).to(self.device)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.long).to(self.device)

        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss


class CustomTrainer(Trainer):
    """커스텀 된 트레이너를 만드는 클래스입니다."""
    def __init__(self, *args, loss_type=None, alpha=1.0, gamma=2.0, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.loss = FocalLoss(alpha=alpha, gamma=gamma, device=device) if loss_type=="focal" else FocalLoss(1.0, 0.0, device=device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """ 기존 loss를 커스텀loss로 변경합니다. (None:CE loss) """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss(logits, labels)
        return (loss, outputs) if return_outputs else loss