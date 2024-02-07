# Intro

A simple API server using FastAPI for serving a small and high quality flow classification model onnxruntime only for CPU inference on Google Cloud Run.

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

The following commands will deploy the model to Google Cloud Run.

```bash
gcloud projects create flow-cloudrun
gcloud config set project flow-cloudrun
docker build --tag gcr.io/flow-cloudrun/flowml .
docker push gcr.io/flow-cloudrun/flowml
gcloud run deploy flow-ml-app --platform managed --region europe-west3 --image gcr.io/flow-cloudrun/flowml --service-account yourservice account --allow-unauthenticated
```




# Models

| Model | 
| --- |
| [minuva/MiniLMv2-toxic-jigsaw](https://huggingface.co/minuva/MiniLM-L6-toxic-all-labels)
| [minuva/MiniLMv2-toxic-jigsaw-onnx](https://huggingface.co/Ngit/MiniLM-L6-toxic-all-labels-onnx)
