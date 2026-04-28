from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import time
import logging
import json
from datetime import datetime
from typing import Union
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic AI Orchestrator")

client = OpenAI()

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
    agent_reasoning: str
    model_used: str
    result: Union[dict, list]
    total_latency_ms: float
    timestamp: str

async def check_model_health(model_name: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client_http:
            response = await client_http.get(
                f"{MODEL_SERVICES[model_name]}/health"
            )
            return response.status_code == 200
    except:
        return False

async def agent_routing_decision(text: str) -> tuple[str, str]:
    health_status = {}
    for model_name in MODEL_SERVICES.keys():
        health_status[model_name] = await check_model_health(model_name)
    
    logger.info(f"Model health status: {health_status}")
    
    health_context = "\n".join([
        f"- {model}: {'HEALTHY' if status else 'UNHEALTHY'}"
        for model, status in health_status.items()
    ])
    
    prompt = f"""You are an intelligent AI orchestration agent responsible for routing user requests to the most appropriate AI model.

Available models and their current health status:
{health_context}

Model capabilities:
- sentiment: Analyzes emotional tone of text (positive/negative). Best for reviews, opinions, feedback.
- summarization: Condenses long text into shorter summary. Best for articles, reports, long documents.
- ner: Extracts named entities (persons, organizations, locations) from text. Best for identifying who, what, where.

User request: "{text}"

Instructions:
1. Analyze the user request carefully
2. Consider which model best fits the request
3. ONLY route to HEALTHY models
4. If best model is unhealthy, choose next best healthy model
5. Respond in JSON format only:
{{
    "model": "sentiment|summarization|ner",
    "reasoning": "Brief explanation of your decision"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    
    response_text = response.choices[0].message.content.strip()
    logger.info(f"Agent response: {response_text}")
    
    try:
        decision = json.loads(response_text)
        return decision["model"], decision["reasoning"]
    except:
        return "sentiment", "Fallback to sentiment due to parsing error"

@app.get("/health")
def health():
    return {"status": "healthy", "orchestrator": "agentic"}

@app.post("/orchestrate", response_model=OrchestratorResponse)
async def orchestrate(request: OrchestratorRequest):
    start_time = time.time()
    
    routing_decision, agent_reasoning = await agent_routing_decision(request.text)
    logger.info(f"Agent routing: {routing_decision} | Reasoning: {agent_reasoning}")
    
    service_url = MODEL_SERVICES[routing_decision]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client_http:
            response = await client_http.post(
                f"{service_url}/predict",
                json={"text": request.text}
            )
            model_result = response.json()
    except Exception as e:
        logger.error(f"Model service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    total_latency = (time.time() - start_time) * 1000
    
    return OrchestratorResponse(
        orchestrator="agentic-ai",
        routing_decision=routing_decision,
        agent_reasoning=agent_reasoning,
        model_used=model_result.get("model", "unknown"),
        result=model_result.get("result", {}),
        total_latency_ms=round(total_latency, 2),
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/")
def root():
    return {"service": "Agentic Orchestrator", "type": "llm-powered"}
