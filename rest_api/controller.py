import logging
import os
import numpy as np

from fastapi import APIRouter
from src.onnx_model import OnnxTransformer
from tokenizers import Tokenizer
from typing import Dict
from .schema import TaskInput
from .flow import get_task_flow_labels


logger = logging.getLogger(__name__)
router = APIRouter()

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

flow_tokenizer_agent = Tokenizer.from_file(
    os.path.join(agent_flow_model_name, "tokenizer.json")
)
flow_tokenizer_agent.enable_padding(pad_token="<pad>", pad_id=1)
flow_tokenizer_agent.enable_truncation(max_length=256)


@router.post("/conversation_flow_plugin", response_model=Dict[str, str])
async def task_flow(request: TaskInput):
    user_flow = get_task_flow_labels(
        request.llm_input, user_flow_model, flow_tokenizer_user
    )
    agent_flow = get_task_flow_labels(
        request.llm_output, agent_flow_model, flow_tokenizer_agent
    )
    return {"user_flow": user_flow, "agent_flow": agent_flow}
