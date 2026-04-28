from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Summarization Service")

logger.info("Loading summarization model...")
summarization_pipeline = pipeline(
    "summarization",
    model="t5-small"
)
logger.info("Model loaded successfully!")

class TextRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

class PredictionResponse(BaseModel):
    model: str
    task: str
    input_text: str
    result: dict
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "healthy", "model": "summarization"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    start_time = time.time()
    input_text = "summarize: " + request.text
    result = summarization_pipeline(
        input_text,
        max_length=request.max_length,
        min_length=request.min_length,
    )[0]
    latency = (time.time() - start_time) * 1000

    logger.info(f"Summary generated | Latency: {latency:.2f}ms")

    return PredictionResponse(
        model="t5-small",
        task="summarization",
        input_text=request.text,
        result=result,
        latency_ms=round(latency, 2)
    )

@app.get("/")
def root():
    return {"service": "Text Summarization", "status": "running"}
