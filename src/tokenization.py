from __future__ import annotations

import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding


ARCHITECTURE_STYLES = {"entity_start", "entity_start_end"}


def marker_position_ids(input_ids: list[int], subject_marker_id: int, object_marker_id: int, max_length: int) -> list[int]:
    """input_ids와 동일한 길이의 Matching the Blanks 위치 id를 생성한다."""
    loc_ids = [0] * len(input_ids)
    if loc_ids:
        loc_ids[0] = 1

    subject_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == subject_marker_id]
    object_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == object_marker_id]

    if len(subject_positions) >= 2:
        loc_ids[subject_positions[0]] = 2
        loc_ids[subject_positions[1]] = 3
    if len(object_positions) >= 2:
        loc_ids[object_positions[0]] = 4
        loc_ids[object_positions[1]] = 5

    if len(loc_ids) < max_length:
        loc_ids.extend([0] * (max_length - len(loc_ids)))
    return loc_ids[:max_length]


def tokenize_dataset(df, tokenizer, max_length: int, architecture: str = "cls") -> Dataset:
    """전처리된 input_text를 tokenizer로 변환하여 Trainer용 Dataset을 생성한다."""
    dataset = Dataset.from_pandas(df[["input_text", "label"]], preserve_index=False)
    subject_marker_id = tokenizer.convert_tokens_to_ids("@")
    object_marker_id = tokenizer.convert_tokens_to_ids("#")
    use_mtb = architecture in ARCHITECTURE_STYLES

    def tokenize(batch):
        tokenized = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length" if use_mtb else False,
        )

        if use_mtb:
            tokenized["matching_the_blanks_ids"] = [
                marker_position_ids(input_ids, subject_marker_id, object_marker_id, max_length)
                for input_ids in tokenized["input_ids"]
            ]
        return tokenized

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["input_text"])
    if "label" in tokenized_dataset.column_names and "labels" not in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    return tokenized_dataset


class ReDataCollator:
    """기본 padding에 matching_the_blanks_ids padding/tensor 변환을 추가한다."""

    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        mtb_ids = None
        if "matching_the_blanks_ids" in features[0]:
            mtb_ids = [feature.pop("matching_the_blanks_ids") for feature in features]

        batch = self.base_collator(features)

        if mtb_ids is not None:
            target_length = batch["input_ids"].shape[1]
            padded = []
            for ids in mtb_ids:
                ids = list(ids)
                if len(ids) < target_length:
                    ids = ids + [0] * (target_length - len(ids))
                padded.append(ids[:target_length])
            batch["matching_the_blanks_ids"] = torch.tensor(padded, dtype=torch.long)

        return batch
