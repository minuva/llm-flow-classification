# Intro

An efficient open-source LLM conversation flow classification system built on FastAPI. It uses two models to identify common events and patterns within the user interactions with LLMs. For example, to identify an apology where the agent acknowledges a mistake, or a complaint when a user expresses dissatisfaction. The labels can be used as building blocks for advanced LLM analytics. 

The models are optimized and run on onnx for fast CPU-based inference. This server can be deployed in serveless platforms. 

# Install from source
```bash
git clone https://github.com/minuva/llm-flow-classification.git
cd llm-flow-classification.git
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
docker build --tag llmflow .
docker run -p 9612:9612 -it llmflow
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
