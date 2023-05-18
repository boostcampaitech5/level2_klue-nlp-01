from transformers import Trainer
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from utils.config import load_config
import os
import json
import wandb

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import nested_detach

class FocalLoss(nn.Module):
    """
    gamma 하이퍼파라미터를 받아 focal loss를 리턴시킵니다.
    https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/16
    parameter:
        gamma: easy/ hard 가중치
    """

    def __init__(self, gamma=2.0, device=None):
        super().__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, input, target):
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32).to(self.device)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.long).to(self.device)

        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class ClassWeights(nn.Module):
    """ "Effective Number of Samples(ENS) 에포크 초기일수록 소수 클래스에 집중됩니다"""

    def __init__(self, class_num_list, beta=0.9999, device=None):
        super().__init__()
        self.class_num_list = class_num_list
        self.beta = beta
        self.weights = (1.0 - self.beta) / (1.0 - np.power(self.beta, class_num_list))
        self.device = device

    def get_weights(self, epoch, num_epochs):
        weights = self.weights * (self.beta ** (num_epochs - epoch))
        return torch.from_numpy(weights).float().to(self.device)


class LDAMLoss(nn.Module):
    """
    아래 명시된 parameter들을 받아 LDAM Loss를 반환합니다.
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
        batch_m = torch.matmul(self.delta[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class CustomTrainer(Trainer):
    """커스텀 된 트레이너를 만드는 클래스입니다."""

    def __init__(
        self,
        *args,
        loss_type=None,
        focal_loss_gamma=2.0,
        class_num_list=None,
        alpha,
        gamma,
        max_m=0.5,
        weight=None,
        s=30,
        device=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.loss_type = loss_type

        if loss_type == "focal":
            self.loss = FocalLoss(gamma=focal_loss_gamma, device=device)
        elif self.loss_type == "ldam":
            assert class_num_list is not None, "class_num_list가 필요합니다."
            self.loss = LDAMLoss(class_num_list, max_m=0.5, weight=None, s=30, device=device)
            self.class_weights = ClassWeights(class_num_list, device=device)
        else:
            self.loss = FocalLoss(0.0, device=device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """기존 loss를 커스텀loss로 변경합니다. (None:CE loss)"""
        labels = inputs["labels"]
        
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "subject_mask": inputs["subject_mask"].to(self.device),
            # 'token_type_ids' # NOT FOR ROBERTA!
            "object_mask": inputs["object_mask"].to(self.device),
            "label": labels.to(self.device),
        }
        
        outputs = model(**inputs)

        if self.loss_type == "ldam":
            self.loss.weight = self.class_weights.get_weights(
                self.state.epoch, self.args.num_train_epochs
            )

        loss = self.loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self,model,inputs,prediction_loss_only,ignore_keys = None,):
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                pass
            else:
                # train 중 일 때는 여기로 가는 듯
                if has_labels:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    logits = outputs
                    
                else:
                    loss = None
                    outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        torch.save(self.model.state_dict(),os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.model.config.to_dict(), f)
            
    def evaluation_loop(self, *args, **kwargs):
        """evaluation단계에서 confusion matrix를 생성하기위한 작업을 추가합니다."""
        output = super().evaluation_loop(*args, **kwargs)

        preds = output.predictions
        labels = output.label_ids
        self.draw_confusion_matrix(preds, labels)

        return output

    def draw_confusion_matrix(self, pred, label_ids):
        """seaborn을 사용하여 confusion matrix를 출력하고 wandb로 전달합니다."""
        cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))
        cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        cmn = cmn.astype("int")
        fig = plt.figure(figsize=(22, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # > non-normalized confusion matrix
        cm_plot = sns.heatmap(cm, cmap="Blues", fmt="d", annot=True, ax=ax1)
        cm_plot.set_xlabel("pred")
        cm_plot.set_ylabel("true")
        cm_plot.set_title("confusion matrix")

        # > normalized confusion matrix
        cmn_plot = sns.heatmap(cmn, cmap="Blues", fmt="d", annot=True, ax=ax2)
        cmn_plot.set_xlabel("pred")
        cmn_plot.set_ylabel("true")
        cmn_plot.set_title("confusion matrix normalize")
        wandb.log({"confusion_matrix": wandb.Image(fig)})
