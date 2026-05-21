from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


INPUT_STYLES = {
    "baseline",
    "basic_marker",
    "typed_marker",
    "s_sep_o",
    "s_and_o",
    "question",
    "type_prompt",
    "entity_mask",
    "entity_marker",
    "entity_marker_punct",
    "entity_marker_punct_s_and_o",
    "entity_marker_punct_question",
    "typed_entity_marker",
    "typed_entity_marker_punct",
    "typed_entity_marker_punct_s_and_o",
    "typed_entity_marker_punct_question",
    "typed_entity_marker_punct_v2",
    "typed_entity_marker_non_object_type",
}

TYPE_KO = {
    "PER": "사람",
    "ORG": "조직",
    "LOC": "장소",
    "DAT": "날짜",
    "POH": "기타",
    "NOH": "수량",
}

TYPE_DESC = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "DAT": "date",
    "POH": "other",
    "NOH": "number",
}


def _entity_to_dict(entity) -> Dict:
    if isinstance(entity, dict):
        return entity
    if hasattr(entity, "as_py"):
        return entity.as_py()
    raise TypeError(f"Unsupported entity format: {type(entity)}")


def _insert_span_text(sentence: str, subject: Dict, obj: Dict, subject_text: str, object_text: str) -> str:
    spans = [
        (int(subject["start_idx"]), int(subject["end_idx"]), subject_text),
        (int(obj["start_idx"]), int(obj["end_idx"]), object_text),
    ]
    pieces = []
    cursor = 0
    for start, end, text in sorted(spans, key=lambda x: x[0]):
        pieces.append(sentence[cursor:start])
        pieces.append(text)
        cursor = end + 1
    pieces.append(sentence[cursor:])
    return "".join(pieces)


def _marker_sentence(sentence: str, subject: Dict, obj: Dict, input_style: str) -> str:
    subject = _entity_to_dict(subject)
    obj = _entity_to_dict(obj)
    subj_type = str(subject["type"]).upper()
    obj_type = str(obj["type"]).upper()

    if input_style in {"basic_marker", "entity_marker"}:
        subject_text = f"[E1] {subject['word']} [/E1]"
        object_text = f"[E2] {obj['word']} [/E2]"
    elif input_style in {"typed_marker", "typed_entity_marker"}:
        subject_text = f"<S:{subj_type}> {subject['word']} </S:{subj_type}>"
        object_text = f"<O:{obj_type}> {obj['word']} </O:{obj_type}>"
    elif input_style == "entity_mask":
        subject_text = f"[SUB-{subj_type}]"
        object_text = f"[OBJ-{obj_type}]"
    elif input_style == "entity_marker_punct":
        subject_text = f"@ {subject['word']} @"
        object_text = f"# {obj['word']} #"
    elif input_style == "typed_entity_marker_punct":
        subject_text = f"@ {TYPE_KO.get(subj_type, subj_type)} {subject['word']} @"
        object_text = f"# ^ {TYPE_KO.get(obj_type, obj_type)} ^ {obj['word']} #"
    elif input_style == "typed_entity_marker_punct_v2":
        subject_text = f"@ {TYPE_KO.get(subj_type, subj_type)} {subject['word']} @"
        object_text = f"$ ^ {TYPE_KO.get(obj_type, obj_type)} ^ {obj['word']} $"
    elif input_style == "typed_entity_marker_non_object_type":
        subject_text = f"<S:{subj_type}> {subject['word']} </S:{subj_type}>"
        object_text = obj["word"]
    else:
        raise ValueError(f"Marker insertion does not support {input_style}")
    return _insert_span_text(sentence, subject, obj, subject_text, object_text)


def _combined_marker_sentence(sentence: str, subject: Dict, obj: Dict, typed: bool = False) -> str:
    subject = _entity_to_dict(subject)
    obj = _entity_to_dict(obj)

    if typed:
        subj_type = str(subject["type"]).upper()
        obj_type = str(obj["type"]).upper()
        subj_desc = TYPE_DESC.get(subj_type, str(subject["type"]).lower())
        obj_desc = TYPE_DESC.get(obj_type, str(obj["type"]).lower())
        subject_text = f"@ * {subj_desc} * {subject['word']} @"
        object_text = f"# ^ {obj_desc} ^ {obj['word']} #"
    else:
        subject_text = f"@ {subject['word']} @"
        object_text = f"# {obj['word']} #"

    return _insert_span_text(sentence, subject, obj, subject_text, object_text)


def preprocess_example(example: Dict, input_style: str = "entity_marker") -> str:
    if input_style not in INPUT_STYLES:
        raise ValueError(f"input_style must be one of {sorted(INPUT_STYLES)}")

    subject = _entity_to_dict(example["subject_entity"])
    obj = _entity_to_dict(example["object_entity"])
    sentence = example["sentence"]

    if input_style == "baseline":
        return sentence
    if input_style == "s_sep_o":
        return f"{subject['word']}[SEP]{obj['word']}[SEP]{sentence}"
    if input_style == "s_and_o":
        return f"{subject['word']}과 {obj['word']}의 관계[SEP]{sentence}"
    if input_style == "question":
        return f"{sentence}[SEP]{subject['word']}과 {obj['word']}의 관계는 무엇입니까?"
    if input_style == "type_prompt":
        subj_desc = TYPE_DESC.get(str(subject["type"]).upper(), str(subject["type"]).lower())
        obj_desc = TYPE_DESC.get(str(obj["type"]).upper(), str(obj["type"]).lower())
        return (
            f"{sentence} [SEP] "
            f"subject는 {subj_desc} 유형의 개체이고 object는 {obj_desc} 유형의 개체이다. "
            f"문맥을 바탕으로 두 개체 사이의 관계를 분류하시오."
        )
    if input_style == "entity_marker_punct_s_and_o":
        marked_sentence = _combined_marker_sentence(sentence, subject, obj, typed=False)
        return f"{subject['word']}과 {obj['word']}의 관계 [SEP] {marked_sentence}"
    if input_style == "entity_marker_punct_question":
        marked_sentence = _combined_marker_sentence(sentence, subject, obj, typed=False)
        return f"{marked_sentence} [SEP] {subject['word']}과 {obj['word']}의 관계는 무엇입니까?"
    if input_style == "typed_entity_marker_punct_s_and_o":
        marked_sentence = _combined_marker_sentence(sentence, subject, obj, typed=True)
        return f"{subject['word']}과 {obj['word']}의 관계 [SEP] {marked_sentence}"
    if input_style == "typed_entity_marker_punct_question":
        marked_sentence = _combined_marker_sentence(sentence, subject, obj, typed=True)
        return f"{marked_sentence} [SEP] {subject['word']}과 {obj['word']}의 관계는 무엇입니까?"
    return _marker_sentence(sentence, subject, obj, input_style)


def preprocess_dataframe(df: pd.DataFrame, input_style: str = "entity_marker") -> pd.DataFrame:
    processed = df.copy()
    processed["input_text"] = [
        preprocess_example(row, input_style=input_style) for row in processed.to_dict("records")
    ]
    return processed


def make_preprocess_examples(example: Dict) -> pd.DataFrame:
    rows = [{"input_style": style, "input_text": preprocess_example(example, style)} for style in sorted(INPUT_STYLES)]
    return pd.DataFrame(rows)


def get_special_tokens(input_style: str, entity_types: Iterable[str] | None = None) -> List[str]:
    types = sorted({str(t).upper() for t in (entity_types or ["PER", "ORG", "DAT", "LOC", "POH", "NOH"])})
    if input_style in {
        "baseline",
        "s_sep_o",
        "s_and_o",
        "question",
        "type_prompt",
    }:
        return []
    if input_style in {"basic_marker", "entity_marker"}:
        return ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    if input_style in {
        "entity_marker_punct",
        "entity_marker_punct_s_and_o",
        "entity_marker_punct_question",
    }:
        return ["@", "#"]
    if input_style == "entity_mask":
        return [token for t in types for token in (f"[SUB-{t}]", f"[OBJ-{t}]")]
    if input_style in {"typed_marker", "typed_entity_marker", "typed_entity_marker_non_object_type"}:
        tokens = []
        for t in types:
            tokens.extend([f"<S:{t}>", f"</S:{t}>"])
            if input_style != "typed_entity_marker_non_object_type":
                tokens.extend([f"<O:{t}>", f"</O:{t}>"])
        return tokens
    if input_style == "typed_entity_marker_punct":
        return ["@", "#", "^", *sorted(set(TYPE_KO.values()))]
    if input_style in {
        "typed_entity_marker_punct_s_and_o",
        "typed_entity_marker_punct_question",
    }:
        return ["@", "#", "*", "^", *sorted(set(TYPE_DESC.values()))]
    if input_style == "typed_entity_marker_punct_v2":
        return ["@", "$", "^", *sorted(set(TYPE_KO.values()))]
    raise ValueError(f"input_style must be one of {sorted(INPUT_STYLES)}")


def add_special_tokens(tokenizer, input_style: str, entity_types: Iterable[str] | None = None) -> int:
    tokens = get_special_tokens(input_style, entity_types)
    if not tokens:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": tokens})


def find_entity_span(sentence: str, word: str) -> tuple[int, int]:
    start = sentence.find(word)
    if start < 0:
        raise ValueError(f"Cannot find entity word in sentence: {word}")
    return start, start + len(word) - 1
