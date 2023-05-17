import os
import torch
import json
from torch import nn
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

from transformers import AutoModelForSequenceClassification, PreTrainedModel, BertPreTrainedModel

class CustomModel(PreTrainedModel):
    """새로운 레이어를 추가하거나, loss fucntion을 수정하는 등, 모델을 커스텀 하기 위한 클래스입니다.
    
       pretrain된 모델을 불러올때, 모델마다 무엇을 input해야 하는지 다를 수 있기 때문에, 주의 하셔야 합니다.
    """    
    def __init__(self, model_config, model_name):
        super().__init__(config=model_config)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=30)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids, labels=labels)
        return x

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=False):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
            
        return self.linear(x)


class RBERT(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py"""
    def __init__(self,
                 model_name: str = "klue/bert-base",
                 num_labels: int = 30,
                 dropout_rate: float = 0.2,
                 special_tokens_dict: dict = None,
                 tokenizer: object = None,
                 is_train: bool = True,):
        
        super(RBERT, self).__init__()

        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer != None else AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.plm = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels

        # add special tokens
        self.special_tokens_dict = special_tokens_dict
        self.plm.resize_token_embeddings(len(self.tokenizer))

        self.cls_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, self.dropout_rate)
        self.entity_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, self.dropout_rate)
        self.label_classifier = FCLayer(
            self.config.hidden_size * 3,
            self.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / (length_tensor.float() + 1e-6)  # broadcasting
        
        return avg_vector

    def forward(self,input_ids,attention_mask,subject_mask=None,object_mask=None,label=None,):
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]  # [CLS] token's hidden featrues(hidden state)

        # hidden state's average in between entities
        e1_h = self.entity_average(sequence_output, subject_mask)  # token in between subject entities
        e2_h = self.entity_average(sequence_output, object_mask)  # token in between object entities

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)  # [CLS] token -> hidden state
        e1_h = self.entity_fc_layer(e1_h)  # subject entity's fully connected layer
        e2_h = self.entity_fc_layer(e2_h)  # object entity's fully connected layer

        # Concat -> fc_layer / [CLS], subject_average, object_average
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        
        return logits

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        # model 저장
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # config 저장
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)