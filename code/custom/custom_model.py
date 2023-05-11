from transformers import AutoModelForSequenceClassification, PreTrainedModel

class CustomModel(PreTrainedModel):
    """새로운 레이어를 추가하거나, loss fucntion을 수정하는 등, 모델을 커스텀 하기 위한 클래스입니다.
    
       pretrain된 모델을 불러올때, 모델마다 무엇을 input해야 하는지 다를 수 있기 때문에, 주의 하셔야 합니다.
    """    

    def __init__(self, model_name, model_config, tokenizer):
        super().__init__(config=model_config)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=30)
        self.tokenizer = tokenizer
        
        self.bert.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids, labels=labels)
        return x
