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
    https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/16
    parameter:
        gamma: easy/ hard 가중치
    """
    def __init__(self, gamma=2.0, device=None):
        super().__init__()
        self.gamma = gamma
        self.device=device

    def forward(self, input, target):
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32).to(self.device)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.long).to(self.device)

        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

        

class ClassWeights(nn.Module):
    """"Effective Number of Samples(ENS) 에포크 초기일수록 소수 클래스에 집중됩니다"""
    def __init__(self, class_num_list, beta=0.9999, device=None):
        super().__init__()
        self.class_num_list = class_num_list
        self.beta = beta
        self.weights = (1.0 - self.beta) / (1.0 - np.power(self.beta, class_num_list))
        self.device=device

    def get_weights(self, epoch, num_epochs):
        weights = self.weights * (self.beta ** (num_epochs - epoch))
        return torch.from_numpy(weights).float().to(self.device)


class LDAMLoss(nn.Module):
    """
    https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    parameter:
        cls_num_list: 클래스 별 비율(빈도)
        max_m: 최대 마진
        weight: 클래스 별 가중치
        s: 하이퍼 파라미터
    """
    def __init__(self, class_num_list, max_m=0.5, weight=None, s=30, device=None):
        super().__init__()
        delta = 1.0 / np.sqrt(np.sqrt(class_num_list))
        delta = delta * (max_m / np.max(delta))
        delta = torch.cuda.FloatTensor(delta)
        self.delta = delta
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1) 
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.delta[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class CustomTrainer(Trainer):
    """커스텀 된 트레이너를 만드는 클래스입니다."""
    def __init__(self, *args, loss_type=None, focal_loss_gamma=2.0, class_num_list=None, max_m=0.5, weight=None, s=30, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

        if loss_type=="focal":
            self.loss = FocalLoss(gamma=focal_loss_gamma, device=device)
        elif self.loss_type=="ldam":
            assert class_num_list is not None, "class_num_list가 필요합니다."
            self.loss = LDAMLoss(class_num_list, max_m=0.5, weight=None, s=30, device=device)
            self.class_weights = ClassWeights(class_num_list, device=device)
        else:
            self.loss = FocalLoss(0.0, device=device)


    def compute_loss(self, model, inputs, return_outputs=False):
        """ 기존 loss를 커스텀loss로 변경합니다. (None:CE loss) """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_type == "ldam":
            self.loss.weight = self.class_weights.get_weights(self.state.epoch, self.args.num_train_epochs)
            
        loss = self.loss(logits, labels)
        return (loss, outputs) if return_outputs else loss