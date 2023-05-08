from transformers import Trainer

class CustomTrainer(Trainer):
    """커스텀 된 트레이너를 만드는 클래스입니다."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
