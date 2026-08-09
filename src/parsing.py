import json
import re


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])


def strip_hidden_field(obj: dict, field: str = "ground_truth_is_fraud") -> dict:
    obj_copy = dict(obj)
    obj_copy.pop(field, None)
    return obj_copy
