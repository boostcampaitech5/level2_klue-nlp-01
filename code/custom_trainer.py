from transformers import Trainer

class CustomTrainer(Trainer):
    """
        Trainer 내부 feature를 수정하고 싶을때 이용하시면 됩니다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
