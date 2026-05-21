from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class ReArchitectureClassifier(nn.Module):
    """CLS/entity marker representation을 비교하기 위한 RE classifier."""

    def __init__(self, model_name: str, num_labels: int, tokenizer, architecture: str, id2label=None, label2id=None):
        super().__init__()
        self.architecture = architecture
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        if id2label is not None:
            self.config.id2label = id2label
        if label2id is not None:
            self.config.label2id = label2id

        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.encoder.resize_token_embeddings(len(tokenizer))

        multiplier = 3 if architecture == "entity_start" else 5
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * multiplier, hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, num_labels),
        )

    def _gather_marker_states(self, hidden_states: torch.Tensor, marker_ids: torch.Tensor) -> torch.Tensor:
        cls_state = hidden_states[:, 0, :]
        marker_order = [2, 4] if self.architecture == "entity_start" else [2, 3, 4, 5]
        batch_vectors = []

        for batch_idx in range(hidden_states.size(0)):
            marker_vectors = []
            for marker_value in marker_order:
                positions = (marker_ids[batch_idx] == marker_value).nonzero(as_tuple=False).flatten()
                position = positions[0] if len(positions) else torch.tensor(0, device=hidden_states.device)
                marker_vectors.append(hidden_states[batch_idx, position, :])
            batch_vectors.append(torch.cat(marker_vectors, dim=-1))

        return torch.cat([cls_state, torch.stack(batch_vectors)], dim=-1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, matching_the_blanks_ids=None, labels=None, **kwargs):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**model_inputs)
        pooled = self._gather_marker_states(outputs.last_hidden_state, matching_the_blanks_ids)
        logits = self.classifier(pooled)
        return SequenceClassifierOutput(logits=logits)


def build_model(config, tokenizer, label_names, id2label, label2id):
    """config의 architecture 값에 맞는 분류 모델을 생성한다."""
    architecture = config.get("architecture", "cls")
    model_name = config["model_name_or_path"]

    if architecture in {"entity_start", "entity_start_end"}:
        return ReArchitectureClassifier(
            model_name,
            num_labels=len(label_names),
            tokenizer=tokenizer,
            architecture=architecture,
            id2label=id2label,
            label2id=label2id,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model
