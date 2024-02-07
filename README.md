# Intro

A simple API server using FastAPI for serving two small flow classification models with onnxruntime package for CPU inference on Google Cloud Run.

This example uses two models that classify the agents and users dialog. The models identify common events and patterns within the conversation flow. Such events include an apology, where the agent acknowledges a mistake, and a complaint, when a user expresses dissatisfaction.


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

# Deploy to cloun Run

The following commands will deploy the model to Google Cloud Run:

```bash
gcloud projects create flow-cloudrun
gcloud config set project flow-cloudrun
docker build --tag gcr.io/flow-cloudrun/flowml .
docker push gcr.io/flow-cloudrun/flowml
gcloud run deploy flow-ml-app --platform managed --region europe-west3 --image gcr.io/flow-cloudrun/flowml --service-account yourservice-account --allow-unauthenticated
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

| Model | 
| --- |
| [minuva/MiniLMv2-agentflow-v2-onnx](https://huggingface.co/minuva/MiniLMv2-agentflow-v2-onnx)
| [minuva/MiniLMv2-userflow-v2-onnx](https://huggingface.co/minuva/MiniLMv2-userflow-v2-onnx)
