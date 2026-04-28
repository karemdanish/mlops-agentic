FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn==0.46.0 \
    torch==2.11.0+cpu \
    transformers==4.40.0 \
    pydantic==2.13.3 \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app.py .

RUN python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')"

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
