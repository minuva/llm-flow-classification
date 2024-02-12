# Intro

A simple API server using FastAPI for serving two small flow classification models with onnxruntime package for fast CPU inference.

This repository uses two models that classify the agents and users dialog. The models identify common events and patterns within the conversation flow. Such events include an apology, where the agent acknowledges a mistake, and a complaint, when a user expresses dissatisfaction.


# Install from source
```bash
git clone https://github.com/minuva/flow-cloudrun.git
cd flow-cloudrun.git
pip install -r requirements.txt
```


# Run locally

Run the following command to start the server (from the root directory):

```bash
chmod +x ./run.sh
./run.sh
```

Check `config.py` for more configuration options.


# Run with Docker

Run the following command to start the server (the root directory):

```bash
docker build --tag flow .
docker run -p 9612:9612 -it flow
```

# Example usage

```bash
curl -X 'POST' \
  'http://127.0.0.1:9612/flow' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "text": "My apologies",
      "speaker": "agent"
    },
{
      "text": "That is fine, can you expand on that list",
      "speaker": "user"
    }
  ]
}'
```

And returns

```json
[
  "agent_apology_error_mistake",
  "more_listing_or_expand"
]
```



# Models

| Model | Description |
| --- | -- |
| [minuva/MiniLMv2-agentflow-v2-onnx](https://huggingface.co/minuva/MiniLMv2-agentflow-v2-onnx) | Agent flow model |
| [minuva/MiniLMv2-userflow-v2-onnx](https://huggingface.co/minuva/MiniLMv2-userflow-v2-onnx) | User flow model |
