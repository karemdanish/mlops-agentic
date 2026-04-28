from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import time
import logging
from datetime import datetime
from typing import Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Baseline Rule-Based Orchestrator")

MODEL_SERVICES = {
    "sentiment": "http://sentiment-service:8000",
    "summarization": "http://summarization-service:8001",
    "ner": "http://ner-service:8002"
}

class OrchestratorRequest(BaseModel):
    text: str

class OrchestratorResponse(BaseModel):
    orchestrator: str
    routing_decision: str
    model_used: str
    result: Union[dict, list]
    total_latency_ms: float
    timestamp: str

def rule_based_routing(text: str) -> str:
    text_lower = text.lower()
    
    sentiment_keywords = [
        "sentiment", "feeling", "opinion", "review",
        "good", "bad", "terrible", "amazing", "awful",
        "happy", "sad", "positive", "negative", "like",
        "dislike", "love", "hate", "worst", "best"
    ]
    
    summarization_keywords = [
        "summarize", "summary", "summarization", "shorten",
        "brief", "condense", "overview", "tldr", "main points"
    ]
    
    ner_keywords = [
        "entity", "entities", "person", "organization",
        "location", "extract", "identify", "find", "named",
        "who", "where", "company", "place", "people"
    ]
    
    sentiment_score = sum(1 for k in sentiment_keywords if k in text_lower)
    summarization_score = sum(1 for k in summarization_keywords if k in text_lower)
    ner_score = sum(1 for k in ner_keywords if k in text_lower)
    
    scores = {
        "sentiment": sentiment_score,
        "summarization": summarization_score,
        "ner": ner_score
    }
    
    best_match = max(scores, key=scores.get)
    
    if scores[best_match] == 0:
        return "sentiment"
    
    return best_match

@app.get("/health")
def health():
    return {"status": "healthy", "orchestrator": "baseline"}

@app.post("/orchestrate", response_model=OrchestratorResponse)
async def orchestrate(request: OrchestratorRequest):
    start_time = time.time()
    
    routing_decision = rule_based_routing(request.text)
    logger.info(f"Routing decision: {routing_decision}")
    
    service_url = MODEL_SERVICES[routing_decision]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{service_url}/predict",
                json={"text": request.text}
            )
            model_result = response.json()
    except Exception as e:
        logger.error(f"Model service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    total_latency = (time.time() - start_time) * 1000
    
    return OrchestratorResponse(
        orchestrator="baseline-rule-based",
        routing_decision=routing_decision,
        model_used=model_result.get("model", "unknown"),
        result=model_result.get("result", {}),
        total_latency_ms=round(total_latency, 2),
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/")
def root():
    return {"service": "Baseline Orchestrator", "type": "rule-based"}
