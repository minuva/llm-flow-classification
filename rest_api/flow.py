import numpy as np

from tokenizers import Tokenizer
from src.onnx_model import OnnxTransformer

CLASSES_TO_IGNORE = ["OTHER"]


def postprocess(outputs, model, threshold=0.80):
    e_x = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    scores = e_x / e_x.sum(axis=1, keepdims=True)
    results = []

    for item in scores:
        labels = []
        scores = []
        for idx, s in enumerate(item):
            label = model.config["id2label"][str(idx)]
            if s > threshold and label not in CLASSES_TO_IGNORE:
                labels.append(label)
                scores.append(s)

        if scores and labels:
            results.append({"labels": labels, "scores": scores})

    return results


def predict(
    model: OnnxTransformer,
    tokenizer: Tokenizer,
    text: str,
    threshold=0.3,
):
    encoding = tokenizer.encode(text)

    inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
    }
    output = model.predict(inputs)
    result = postprocess(output, model, threshold)

    return result


def get_task_flow_labels(
    text,
    model,
    tokenizer,
) -> str:

    flow = predict(model, tokenizer, text)
    return flow[0]["labels"][0] if flow and flow[0]["labels"] else ""
