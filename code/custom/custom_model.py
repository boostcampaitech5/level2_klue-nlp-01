import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, RobertaModel, PreTrainedModel, AutoTokenizer, RobertaForSequenceClassification, RobertaPreTrainedModel

class RE_Model(PreTrainedModel):
    """새로운 레이어를 추가하거나, loss fucntion을 수정하는 등, 모델을 커스텀 하기 위한 클래스입니다.
    
       pretrain된 모델을 불러올때, 모델마다 무엇을 input해야 하는지 다를 수 있기 때문에, 주의 하셔야 합니다.
    """    
    def __init__(self, config, ref_input_ids: torch.tensor, ref_mask: torch.tensor,  n_class=25, hidden_size=768, PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'):
        super().__init__(config)
        self.batch_n = 0
        self.hidden_size = hidden_size
        self.ref_input_ids = ref_input_ids
        self.ref_mask = ref_mask
        self.bone = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME) # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

        self.ref_value = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) 
        self.ref_key = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) 
        self.pool_cat = nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)
        self.gate = nn.Linear(in_features=self.hidden_size, out_features=1) 

        self.out = nn.Linear(in_features=self.hidden_size, out_features=n_class)

    def forward(self, input_ids, attention_mask, sub_idx, obj_idx, attn_guide, device):
        """
        - E1: encoder(input_ids) -> rep.    
        - E2: encoder(reference_sent_ids) -> rep.
        - Q = IN_rep: relu(pool(concat(E1_sub, E1_obj))) -> input sentence rep.
        - K, V: linear_k(E2_cls), linear_v(E2_cls) # vectors for each reference-sentence (each V represent reference-sentence)
        - attn score = dot_prod(Q, K) / root(dim_k) # attn score for each reference-sentence
        - final input rep: Q + gate*weighted_sum(attn_score * V)
            - gate = tanh(linear_gate(Q))
        - output1: linear(final input rep)
        - output2: Q + gate*weighted_sum(flipped_attn_score * V)
        - output3: linear(Vs)
        """
        # BERT out
        self.batch_n = input_ids.size()[0]
        
        o1 = self.bone(input_ids=self.ref_input_ids.to(device), attention_mask=self.ref_mask.to(device), token_type_ids=None, position_ids=None) # ref_pooled_output: (n_ref, dim)
        ref_last_hidn_state = o1.last_hidden_state
        
        cls_token = ref_last_hidn_state[:, 0, :] #(n_ref, dim)
        ref_value = self.ref_value(cls_token) #(n_ref, dim)
        ref_key = self.ref_key(cls_token) #(n_ref, dim)

        o2 = self.bone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None) # out: bs, max_seq_len, hidn_dim
        last_hidn_state= o2.last_hidden_state
        bs, seq_len, _ = last_hidn_state.size()
        
        subj_special_start_id = torch.zeros(bs, seq_len).to(device)
        subj_special_start_id = subj_special_start_id.scatter_(1, sub_idx.unsqueeze(1), 1)        
        subj_rep = torch.bmm(subj_special_start_id.unsqueeze(1), last_hidn_state)
        
        obj_special_start_id = torch.zeros(bs, seq_len).to(device)
        obj_special_start_id = obj_special_start_id.scatter_(1, obj_idx.unsqueeze(1), 1)        
        obj_rep = torch.bmm(obj_special_start_id.unsqueeze(1), last_hidn_state)

        bert_sub_obj_pos_cat = torch.cat([subj_rep.squeeze(1), obj_rep.squeeze(1)], dim=1) #(bs, dim)
        pooled = nn.functional.relu(self.pool_cat(bert_sub_obj_pos_cat)) #(bs, dim)
        gating = torch.tanh(self.gate(pooled)) 

        ####attn
        attn_score = torch.matmul(pooled, ref_key.T)/(self.hidden_size**(1/2)) # (bs, n_ref)
        
        ###############################################################################################attn guided
        ref_n_class = attn_score.shape[1]
        weighted_ref = ref_value.repeat(self.batch_n, 1, 1) * attn_score.contiguous().view(self.batch_n, ref_n_class, 1) # (bs, n_class, dim)
        weighted_sum_vec = torch.sum(weighted_ref, dim=1) # (bs, dim)

        attn_score2 = torch.matmul(pooled, ref_key.T)/(self.hidden_size**(1/2)) # (bs, n_class)
        attn_score2[attn_guide==1] *= -1
        weighted_sum_vec2 = torch.sum(ref_value.repeat(self.batch_n, 1, 1) * attn_score2.contiguous().view(self.batch_n, ref_n_class, 1), dim=1)

        ####attn
        cat = torch.cat([pooled + gating*weighted_sum_vec, pooled + gating*weighted_sum_vec2, ref_value], dim=0)
        out = self.out(cat) # (bs*2+n_ref, n_class)
        
        return out[:self.batch_n], torch.softmax(out[self.batch_n:-ref_n_class], dim=1), out[-ref_n_class:], attn_score, gating


class CustomModel(PreTrainedModel):
    """새로운 레이어를 추가하거나, loss fucntion을 수정하는 등, 모델을 커스텀 하기 위한 클래스입니다.
    
       pretrain된 모델을 불러올때, 모델마다 무엇을 input해야 하는지 다를 수 있기 때문에, 주의 하셔야 합니다.
    """    
    def __init__(self, model_config, model_name):
        super().__init__(config=model_config)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=30)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask,
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