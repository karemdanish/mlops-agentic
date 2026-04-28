from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Named Entity Recognition Service")

logger.info("Loading NER model...")
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)
logger.info("Model loaded successfully!")

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    model: str
    task: str
    input_text: str
    result: list
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ner"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    start_time = time.time()
    result = ner_pipeline(request.text)
    latency = (time.time() - start_time) * 1000

    # Clean result for serialization
    cleaned_result = [
        {
            "entity": entity["entity_group"],
            "word": entity["word"],
            "score": round(float(entity["score"]), 4)
        }
        for entity in result
    ]

    logger.info(f"Entities found: {len(cleaned_result)} | Latency: {latency:.2f}ms")

    return PredictionResponse(
        model="dslim/bert-base-NER",
        task="named-entity-recognition",
        input_text=request.text,
        result=cleaned_result,
        latency_ms=round(latency, 2)
    )

@app.get("/")
def root():
    return {"service": "Named Entity Recognition", "status": "running"}
