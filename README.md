# Agentic Orchestration for Multi-Model AI Systems

## Paper
**Title:** Agentic Orchestration for Multi-Model AI Systems: An Experimental Validation and Extension of Kubernetes-Based MLOps Architecture

**Authors:** Danish Karim, Dr. Muhammad Mateen Yaqoob, Dr. Usama Imtiaz


**Abstract:** This repository contains the complete implementation, experiment scripts, and results for our paper. We experimentally validate the Kubernetes-based orchestration architecture proposed by Jebadoss (2025) and extend it with an agentic AI layer powered by GPT-4o-mini. We deploy three NLP model microservices on a K3s Kubernetes cluster and compare a rule-based baseline orchestrator against our agentic extension across four experiments.

---

## Repository Structure
```
mlops-agentic/
├── models/
│   ├── sentiment/
│   │   ├── app.py
│   │   └── Dockerfile
│   ├── summarization/
│   │   ├── app.py
│   │   └── Dockerfile
│   └── ner/
│       ├── app.py
│       └── Dockerfile
│
├── orchestrator/
│   ├── baseline/
│   │   ├── app.py
│   │   └── Dockerfile
│   └── agentic/
│       ├── app.py
│       └── Dockerfile
│
├── k8s/
│   ├── models/
│   │   ├── sentiment-deployment.yaml
│   │   ├── summarization-deployment.yaml
│   │   └── ner-deployment.yaml
│   └── orchestrator/
│       ├── baseline-deployment.yaml
│       └── agentic-deployment.yaml
│
├── experiments/
│   ├── run_experiments.py
│   ├── regenerate_graphs.py
│   ├── exp1_baseline_routing.csv
│   ├── exp1_agentic_routing.csv
│   ├── exp2_ambiguous_requests.csv
│   ├── exp3_failure_recovery.csv
│   └── exp4_latency.csv
│
├── graphs/
│   ├── graph1_routing_accuracy.png
│   ├── graph2_accuracy_by_task.png
│   ├── graph3_latency_by_task.png
│   ├── graph4_failure_recovery.png
│   ├── graph5_latency_distribution.png
│   ├── graph6_latency_over_time.png
│   ├── graph7_latency_stats.png
│   ├── graph8_ambiguous_routing.png
│   └── graph9_overall_dashboard.png
│
└── README.md
```
---

## Prerequisites

### AWS Account
- AWS account with EC2 access
- Recommended instance: **t3.large** (2 vCPUs, 8GB RAM)
- Storage: **60GB SSD**
- OS: **Ubuntu 24.04 LTS**

### OpenAI API Key
- Active OpenAI account
- API key with access to **gpt-4o-mini** model
- Get yours at: https://platform.openai.com/api-keys

### Local Machine Requirements
- SSH client (Windows CMD, Mac/Linux Terminal)
- Web browser (for AWS Console and Grafana)

---

## Step-by-Step Setup Guide

### Phase 1: AWS EC2 Setup

#### Step 1: Launch EC2 Instance
1. Go to AWS Console → EC2 → Launch Instance
2. Configure:
   - **Name:** mlops-agentic-experiment
   - **OS:** Ubuntu 22.04 LTS
   - **Instance Type:** t3.large
   - **Key Pair:** Create new → download .pem file → save safely
   - **Storage:** 60GB gp2

#### Step 2: Configure Security Group
Create a new security group with these inbound rules:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| Custom TCP | TCP | 22 | 0.0.0.0/0 |
| Custom TCP | TCP | 80 | 0.0.0.0/0 |
| Custom TCP | TCP | 443 | 0.0.0.0/0 |
| Custom TCP | TCP | 6443 | 0.0.0.0/0 |
| Custom TCP | TCP | 8000 | 0.0.0.0/0 |
| Custom TCP | TCP | 8001 | 0.0.0.0/0 |
| Custom TCP | TCP | 8002 | 0.0.0.0/0 |
| Custom TCP | TCP | 8003 | 0.0.0.0/0 |

#### Step 3: SSH Into EC2

**Windows:**
```cmd
# Fix .pem file permissions first
icacls "C:\path\to\your-key.pem" /reset
icacls "C:\path\to\your-key.pem" /grant "%USERNAME%:(R)"
icacls "C:\path\to\your-key.pem" /inheritance:r

# Connect
ssh -i "C:\path\to\your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

**Mac/Linux:**
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

#### Step 4: Update System
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip git curl wget
```

#### Step 5: Install K3s
```bash
curl -sfL https://get.k3s.io | sh -
sleep 30
sudo kubectl get nodes
```

#### Step 6: Configure kubectl
```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown ubuntu:ubuntu ~/.kube/config
echo "export KUBECONFIG=~/.kube/config" >> ~/.bashrc
source ~/.bashrc
kubectl get nodes
```

#### Step 7: Install Helm
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

#### Step 8: Add PATH for Python packages
```bash
echo 'export PATH=$PATH:/home/ubuntu/.local/bin' >> ~/.bashrc
source ~/.bashrc
```

---

### Phase 2: Build AI Model Microservices

#### Step 1: Create Project Structure
```bash
mkdir -p ~/mlops-agentic/{models/{sentiment,summarization,ner},orchestrator/{baseline,agentic},k8s/{models,orchestrator},experiments}
```

#### Step 2: Install Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
newgrp docker
sudo chmod 666 /var/run/docker.sock
docker --version
```

#### Step 3: Install Python Dependencies
```bash
sudo apt install python3-pip -y
pip3 install fastapi uvicorn transformers torch --break-system-packages
pip3 install requests matplotlib pandas numpy --break-system-packages
```

#### Step 4: Build Sentiment Analysis Model
```bash
cd ~/mlops-agentic/models/sentiment
```

Clone this repository and copy the files, or create them manually:

```bash
# Build Docker image
docker build -t sentiment-service:v1 .
```

#### Step 5: Build Summarization Model
```bash
cd ~/mlops-agentic/models/summarization
docker build -t summarization-service:v1 .
```

#### Step 6: Build NER Model
```bash
cd ~/mlops-agentic/models/ner
docker build -t ner-service:v1 .
```

#### Step 7: Build Baseline Orchestrator
```bash
cd ~/mlops-agentic/orchestrator/baseline
docker build -t baseline-orchestrator:v1 .
```

#### Step 8: Build Agentic Orchestrator
```bash
cd ~/mlops-agentic/orchestrator/agentic
docker build -t agentic-v4:latest .
```

---

### Phase 3: Deploy to Kubernetes

#### Step 1: Import Docker Images into K3s
```bash
docker save sentiment-service:v1 | sudo k3s ctr images import -
docker save summarization-service:v1 | sudo k3s ctr images import -
docker save ner-service:v1 | sudo k3s ctr images import -
docker save baseline-orchestrator:v1 | sudo k3s ctr images import -
docker save agentic-v4:latest | sudo k3s ctr images import -
```

#### Step 2: Verify Images
```bash
sudo k3s ctr images list | grep -E "sentiment|summarization|ner|orchestrator|agentic"
```

#### Step 3: Create OpenAI Secret
```bash
kubectl create secret generic openai-secret \
  --from-literal=OPENAI_API_KEY=your-openai-api-key-here
```

#### Step 4: Deploy Model Services
```bash
cd ~/mlops-agentic/k8s/models
kubectl apply -f sentiment-deployment.yaml
kubectl apply -f summarization-deployment.yaml
kubectl apply -f ner-deployment.yaml
```

#### Step 5: Deploy Orchestrators
```bash
cd ~/mlops-agentic/k8s/orchestrator
kubectl apply -f baseline-deployment.yaml
kubectl apply -f agentic-deployment.yaml
```

#### Step 6: Verify All Pods Running
```bash
kubectl get pods
```

Expected output:
NAME                                     READY   STATUS
agentic-orchestrator-xxx                 1/1     Running
baseline-orchestrator-xxx                1/1     Running
ner-service-xxx                          1/1     Running
sentiment-service-xxx                    1/1     Running
summarization-service-xxx                1/1     Running

#### Step 7: Verify Services
```bash
kubectl get services
```

---

### Phase 4: Run Experiments

#### Step 1: Setup Port Forwards
```bash
# Kill any existing port forwards
kill $(lsof -t -i:8010) 2>/dev/null
kill $(lsof -t -i:8011) 2>/dev/null
sleep 3

# Start baseline on port 8010
kubectl port-forward svc/baseline-orchestrator 8010:8003 &
sleep 5

# Start agentic on port 8011
kubectl port-forward svc/agentic-orchestrator 8011:8003 &
sleep 5
```

#### Step 2: Verify Both Orchestrators
```bash
# Test baseline
curl -s -X POST http://localhost:8010/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}' | grep orchestrator

# Test agentic
curl -s -X POST http://localhost:8011/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}' | grep orchestrator
```

Expected:
"orchestrator":"baseline-rule-based"
"orchestrator":"agentic-ai"

#### Step 3: Run All Experiments
```bash
cd ~/mlops-agentic/experiments
python3 run_experiments.py
```

**Important:** During Experiment 3, the script will pause and ask you to kill the NER service. Open a second terminal and run:
```bash
kubectl scale deployment ner-service --replicas=0
```
Press Enter to continue. After Experiment 3 completes, restore NER:
```bash
kubectl scale deployment ner-service --replicas=1
```

#### Step 4: Generate Graphs
```bash
python3 regenerate_graphs.py
```

---

### Phase 5: Download Results

Run this on your **local machine**:

```cmd
# Download graphs
scp -i your-key.pem -r ubuntu@YOUR_EC2_IP:~/mlops-agentic/experiments/graphs_final C:\Users\YourName\Desktop\graphs_final

# Download CSV results
scp -i your-key.pem -r ubuntu@YOUR_EC2_IP:~/mlops-agentic/experiments/*.csv C:\Users\YourName\Desktop\results
```

---

## Models Used

| Model | Task | Source |
|-------|------|--------|
| distilbert-base-uncased-finetuned-sst-2-english | Sentiment Analysis | HuggingFace |
| t5-small | Text Summarization | HuggingFace |
| dslim/bert-base-NER | Named Entity Recognition | HuggingFace |

---

## Experiment Results

### Experiment 1: Routing Accuracy
| Orchestrator | Accuracy |
|-------------|---------|
| Baseline (Rule-Based) | 95.0% |
| Agentic (LLM-Powered) | 100.0% |

### Experiment 2: Ambiguous Request Handling
- Both orchestrators handled all 10 requests successfully
- Disagreed on 5/10 routing decisions
- Agentic provided reasoning for every decision

### Experiment 3: Failure Recovery (NER Service Down)
| Orchestrator | Successful Responses |
|-------------|---------------------|
| Baseline | 0/5 |
| Agentic | 5/5 |

### Experiment 4: Latency Performance
| Metric | Baseline | Agentic |
|--------|---------|---------|
| Average | 1088.97ms | 2806.37ms |
| Minimum | 84.47ms | 1124.43ms |
| Maximum | 4761.42ms | 6299.94ms |
| Success Rate | 93.3% | 100% |

---

## Key Findings

- Agentic orchestrator achieves **100% routing accuracy** vs 95% for baseline
- Agentic orchestrator handles **all 5 requests** during service failure vs 0 for baseline
- Agentic orchestrator provides **human-readable reasoning** for every routing decision
- Agentic orchestrator has **+157.7% higher latency** due to LLM reasoning overhead

---

## Important Notes

### Security
- Never commit your `.pem` key file to GitHub
- Never commit your OpenAI API key
- Always use Kubernetes secrets for sensitive credentials

### Cost Estimate
- EC2 t3.large: ~$0.083/hour
- Full experiment run: ~$2-4 total
- OpenAI API calls: ~$0.50-1.00 total

### Known Issues
- K3s image caching: Always use a new image tag when updating containers
- Port forwarding: Always kill existing port forwards before starting new ones
- NER model cold start: First request may be slower than subsequent ones

---

## Citation

If you use this code or results in your research, please cite:
Danish Karim, Dr. Muhammad Mateen Yaqoob, Dr. Usama Imtiaz,
"Agentic Orchestration for Multi-Model AI Systems:
An Experimental Validation and Extension of
Kubernetes-Based MLOps Architecture,"

---

## License
MIT License — feel free to use and modify for your research.

---

## Contact
- Danish Karim — danishkarim97@gmail.com
- GitHub: https://github.com/karemdanish/mlops-agentic

## Results
All experiment results are in experiments/ directory
All graphs are in graphs/ directory

## Paper
[paper link]
