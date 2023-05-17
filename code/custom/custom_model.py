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


class CustomModel(RobertaPreTrainedModel):
    """새로운 레이어를 추가하거나, loss fucntion을 수정하는 등, 모델을 커스텀 하기 위한 클래스입니다.
    
       pretrain된 모델을 불러올때, 모델마다 무엇을 input해야 하는지 다를 수 있기 때문에, 주의 하셔야 합니다.
    """    
    def __init__(self, model_name, config):
        super().__init__(config=config)

        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=30)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # entity special token를 tokenizer에 추가
        self.special_token_list = []
        with open("custom/entity_special_token.txt", "r", encoding="UTF-8") as f:
            for token in f:
                self.special_token_list.append(token.split("\n")[0])

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(set(self.special_token_list))}
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.init_weights()
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids, labels=labels)
                      
        return x