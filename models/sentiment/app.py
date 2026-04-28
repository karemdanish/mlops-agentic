from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis Service")

# Load model at startup
logger.info("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
logger.info("Model loaded successfully!")

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    model: str
    task: str
    input_text: str
    result: dict
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "healthy", "model": "sentiment-analysis"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    start_time = time.time()
    result = sentiment_pipeline(request.text)[0]
    latency = (time.time() - start_time) * 1000
    
    logger.info(f"Prediction: {result} | Latency: {latency:.2f}ms")
    
    return PredictionResponse(
        model="distilbert-base-uncased-finetuned-sst-2-english",
        task="sentiment-analysis",
        input_text=request.text,
        result=result,
        latency_ms=round(latency, 2)
    )

@app.get("/")
def root():
    return {"service": "Sentiment Analysis", "status": "running"}
