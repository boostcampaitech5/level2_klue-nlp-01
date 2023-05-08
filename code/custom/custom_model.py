from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel


class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=768)
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 192)
        self.linear3 = nn.Linear(192, 30)

    def forward(self, x):
        x = self.bert(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
