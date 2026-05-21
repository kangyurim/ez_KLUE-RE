from __future__ import annotations

from transformers import Trainer

try:
    from .losses import classification_loss
except ImportError:
    from losses import classification_loss


class LossTrainer(Trainer):
    """class weighting과 focal loss를 선택적으로 적용하기 위한 Trainer."""

    def __init__(self, *args, class_weights=None, focal_gamma=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        if labels is None:
            labels = inputs.pop("label", None)
        if labels is None:
            raise KeyError(f"LossTrainer expected labels in inputs, but got keys: {sorted(inputs.keys())}")

        outputs = model(**inputs)
        loss = classification_loss(
            outputs.logits,
            labels,
            class_weights=self.class_weights,
            focal_gamma=self.focal_gamma,
        )
        return (loss, outputs) if return_outputs else loss
