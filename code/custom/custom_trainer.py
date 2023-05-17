from transformers import Trainer
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from utils.config import load_config
import os
import json

from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import nested_detach

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
        self.device = device
        self.loss_type = loss_type
        self.loss = FocalLoss(alpha=alpha, gamma=gamma, device=device) if loss_type=="focal" else FocalLoss(1.0, 0.0, device=device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """ 기존 loss를 커스텀loss로 변경합니다. (None:CE loss) """
        labels = inputs.get("labels")
        
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "subject_mask": inputs["subject_mask"].to(self.device),
            # 'token_type_ids' # NOT FOR ROBERTA!
            "object_mask": inputs["object_mask"].to(self.device),
            "label": labels,
        }
        
        outputs = model(**inputs)
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