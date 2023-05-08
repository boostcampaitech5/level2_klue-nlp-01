from transformers import AutoModelForSequenceClassification, PreTrainedModel

class CustomModel(PreTrainedModel):
    """새로운 레이어를 추가하거나, loss fucntion을 수정하는 등, 모델을 커스텀 하기 위한 클래스입니다."""    
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=30)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids, labels=labels)
        return x
