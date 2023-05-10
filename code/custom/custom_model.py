from transformers import AutoModelForSequenceClassification, PreTrainedModel

class CustomModel(PreTrainedModel):
    """새로운 레이어를 추가하거나, loss fucntion을 수정하는 등, 모델을 커스텀 하기 위한 클래스입니다.
    
       pretrain된 모델을 불러올때, 모델마다 무엇을 input해야 하는지 다를 수 있기 때문에, 주의 하셔야 합니다.
    """    
    def __init__(self, model_config, config):
        super().__init__(model_config)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=30)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids, labels=labels)
        return x
