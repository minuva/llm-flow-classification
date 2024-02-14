# LLM conversation flow classification 💬

An effective open-source system for classifying conversation flows in large language models (LLMs) using FastAPI. This system incorporates two models to detect typical events and patterns in interactions with LLMs, such as recognizing an apology where the LLM admits a mistake, or identifying a complaint when a user shows dissatisfaction. These labels serve as foundational elements for sophisticated LLM analytics 📊.

The models have been optimized for rapid CPU-based inference using ONNX, enabling efficient performance ⚡. Furthermore, this system is designed for deployment on serverless platforms.

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
