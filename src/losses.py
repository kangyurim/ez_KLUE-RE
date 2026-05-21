from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits, labels, class_weights=None, focal_gamma=None):
    """Cross entropy, class weighting, focal loss를 하나의 함수에서 처리한다."""
    weights = class_weights.to(logits.device) if class_weights is not None else None
    ce_loss = F.cross_entropy(logits, labels, weight=weights, reduction="none")

    if focal_gamma is not None:
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** float(focal_gamma) * ce_loss).mean()

    return ce_loss.mean()
