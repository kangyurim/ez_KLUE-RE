from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .preprocess import find_entity_span, preprocess_example
except ImportError:
    from preprocess import find_entity_span, preprocess_example


def predict_single(
    model_dir: str,
    sentence: str,
    subject: str,
    obj: str,
    subject_type: str = "PER",
    object_type: str = "ORG",
    input_style: str = "basic_marker",
    max_length: int = 256,
) -> dict:
    subj_start, subj_end = find_entity_span(sentence, subject)
    obj_start, obj_end = find_entity_span(sentence, obj)
    example = {
        "sentence": sentence,
        "subject_entity": {"word": subject, "start_idx": subj_start, "end_idx": subj_end, "type": subject_type},
        "object_entity": {"word": obj, "start_idx": obj_start, "end_idx": obj_end, "type": object_type},
    }
    input_text = preprocess_example(example, input_style)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    label = model.config.id2label.get(pred_id, str(pred_id))
    return {"label_id": pred_id, "label": label, "confidence": float(probs[pred_id]), "input_text": input_text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--subject_type", default="PER")
    parser.add_argument("--object_type", default="ORG")
    parser.add_argument(
        "--input_style",
        default="entity_marker",
        choices=[
            "baseline",
            "s_sep_o",
            "s_and_o",
            "question",
            "type_prompt",
            "entity_marker",
            "entity_marker_punct",
            "entity_marker_punct_s_and_o",
            "entity_marker_punct_question",
            "typed_entity_marker",
            "typed_entity_marker_punct",
            "typed_entity_marker_punct_s_and_o",
            "typed_entity_marker_punct_question",
        ],
    )
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()
    print(
        predict_single(
            args.model_dir,
            args.sentence,
            args.subject,
            args.object,
            args.subject_type,
            args.object_type,
            args.input_style,
            args.max_length,
        )
    )


if __name__ == "__main__":
    main()
