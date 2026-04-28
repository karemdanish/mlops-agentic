# Agentic Orchestration for Multi-Model AI Systems

## Overview
This repository contains the complete implementation 
and experimental results for our paper:
"Agentic Orchestration for Multi-Model AI Systems: 
An Experimental Validation and Extension of 
Kubernetes-Based MLOps Architecture"

## Requirements
- AWS EC2 t3.large (or similar)
- K3s Kubernetes
- Python 3.11
- Docker
- OpenAI API Key

## Infrastructure Setup
1. Launch EC2 t3.large with Ubuntu 24.04
2. Install K3s
3. Install Helm
4. Deploy model services
5. Deploy orchestrators

## Models Used
- Sentiment: distilbert-base-uncased-finetuned-sst-2-english
- Summarization: t5-small
- NER: dslim/bert-base-NER

## Running Experiments
1. Set OPENAI_API_KEY environment variable
2. Deploy all services to Kubernetes
3. Run: python3 experiments/run_experiments.py

## Results
All experiment results are in experiments/ directory
All graphs are in graphs/ directory

## Paper
[paper link]
