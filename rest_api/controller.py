import logging
import os
import numpy as np

from typing import List
from fastapi import APIRouter
from src.onnx_model import OnnxTransformer
from tokenizers import Tokenizer
from pydantic import BaseModel
from enum import Enum
from pydantic.fields import Field

logger = logging.getLogger(__name__)
router = APIRouter()


class SpeakerEnum(str, Enum):
    agent = "agent"
    user = "user"


class TextSpeaker(BaseModel):
    text: str
    speaker: SpeakerEnum = Field(..., description="Either 'agent' or 'user'")


class RequestBody(BaseModel):
    messages: List[TextSpeaker]


def get_texts_by_speaker(texts_speaker, speaker):
    return [
        (idx, text.text)
        for idx, text in enumerate(texts_speaker)
        if text.speaker == speaker
    ]


# Flow cfg
agent_flow_model_name = "MiniLMv2-agentflow-v2-onnx"
user_flow_model_name = "MiniLMv2-userflow-v2-onnx"

agent_flow_model = OnnxTransformer(agent_flow_model_name)
user_flow_model = OnnxTransformer(user_flow_model_name)

flow_tokenizer_user = Tokenizer.from_file(
    os.path.join(user_flow_model_name, "tokenizer.json")
)

flow_tokenizer_user.enable_padding(pad_token="<pad>", pad_id=1)
flow_tokenizer_user.enable_truncation(max_length=256)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    scores = e_x / e_x.sum(axis=1, keepdims=True)
    return scores


@router.post("/flow")
def conversation_flow(request: RequestBody):

    texts_speaker = request.messages
    user_texts_pos = get_texts_by_speaker(texts_speaker, "user")
    agent_texts_pos = get_texts_by_speaker(texts_speaker, "agent")
    user_preds, agent_preds = [], []
    user_labels_map = user_flow_model.config["id2label"]
    agent_labels_map = agent_flow_model.config["id2label"]

    if not user_texts_pos and not agent_texts_pos:
        raise Exception("No user or agent texts provided")

    utexts = [text for _, text in user_texts_pos]
    atexts = [text for _, text in agent_texts_pos]
    upos = [idx for idx, _ in user_texts_pos]
    apos = [idx for idx, _ in agent_texts_pos]

    if utexts:
        output_user = user_flow_model.predict(flow_tokenizer_user, utexts, 16)
        user_preds = [
            (idx, user_labels_map[str(np.argmax(pred))])
            for idx, pred in zip(upos, output_user)
        ]
    if atexts:
        output_agent = agent_flow_model.predict(flow_tokenizer_user, atexts, 16)
        agent_preds = [
            (idx, agent_labels_map[str(np.argmax(pred))])
            for idx, pred in zip(apos, output_agent)
        ]

    all_preds = user_preds + agent_preds
    all_preds = sorted(all_preds, key=lambda x: x[0])
    results = [pred for idx, pred in all_preds]

    return results
