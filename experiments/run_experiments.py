import requests
import json
import time
import csv
from datetime import datetime

# Service URLs via port-forward
BASELINE_URL = "http://localhost:8010/orchestrate"
AGENTIC_URL = "http://localhost:8011/orchestrate"

# ============================================================
# TEST CASES
# ============================================================

# Experiment 1: Clear requests
CLEAR_REQUESTS = [
    # Sentiment (7)
    {"text": "I love this product, it is absolutely amazing!", "expected": "sentiment"},
    {"text": "This is the worst experience I have ever had.", "expected": "sentiment"},
    {"text": "The movie was fantastic and I enjoyed every moment.", "expected": "sentiment"},
    {"text": "I hate this service, it is terrible and slow.", "expected": "sentiment"},
    {"text": "The food was delicious and the staff was very friendly.", "expected": "sentiment"},
    {"text": "I am very happy with my purchase, great quality!", "expected": "sentiment"},
    {"text": "Worst product ever, completely disappointed.", "expected": "sentiment"},
    # Summarization (6)
    {"text": "Summarize this: Artificial intelligence is transforming industries across the world. Companies are investing billions of dollars in AI research and development. Machine learning models are being deployed in healthcare, finance, and education sectors to improve efficiency and accuracy of operations.", "expected": "summarization"},
    {"text": "Please provide a brief summary of the following: The global economy has been experiencing significant changes due to technological advancements. Digital transformation has affected every sector from manufacturing to services. Companies that adapt quickly to these changes tend to perform better in competitive markets.", "expected": "summarization"},
    {"text": "Give me a tldr of this text: Climate change is one of the most pressing issues facing humanity today. Rising temperatures are causing extreme weather events worldwide. Scientists warn that immediate action is needed to prevent catastrophic consequences for future generations.", "expected": "summarization"},
    {"text": "Condense this article: Space exploration has entered a new era with private companies joining government agencies. SpaceX, Blue Origin and Virgin Galactic are leading this commercial space race. These companies are developing reusable rockets to reduce the cost of space travel significantly.", "expected": "summarization"},
    {"text": "Shorten this passage: Quantum computing represents a fundamental shift in computational power. Unlike classical computers that use bits, quantum computers use qubits that can exist in multiple states simultaneously. This allows quantum computers to solve complex problems exponentially faster.", "expected": "summarization"},
    {"text": "Give me the main points of this: The human brain contains approximately 86 billion neurons. These neurons communicate through synapses to process information. Modern neuroscience is beginning to understand how consciousness and memory are formed through these neural connections.", "expected": "summarization"},
    # NER (7)
    {"text": "Extract entities: Apple CEO Tim Cook announced new products in California.", "expected": "ner"},
    {"text": "Find the named entities: Elon Musk founded SpaceX in Hawthorne California.", "expected": "ner"},
    {"text": "Who are the people and organizations mentioned: Google was founded by Larry Page and Sergey Brin at Stanford University.", "expected": "ner"},
    {"text": "Identify entities: Amazon CEO Andy Jassy visited their headquarters in Seattle Washington.", "expected": "ner"},
    {"text": "Extract all named entities from this: Microsoft and OpenAI are collaborating on AI research in San Francisco.", "expected": "ner"},
    {"text": "Find persons, organizations and locations: Mark Zuckerberg leads Meta which is based in Menlo Park California.", "expected": "ner"},
    {"text": "Identify who and where: Sundar Pichai is the CEO of Google which is headquartered in Mountain View.", "expected": "ner"},
]

# Experiment 2: Ambiguous requests
AMBIGUOUS_REQUESTS = [
    {"text": "Tell me about Elon Musk and whether people like him.", "expected": "ner_or_sentiment"},
    {"text": "Analyze this Apple customer review and find the key people mentioned.", "expected": "ner_or_sentiment"},
    {"text": "What do people think about Tesla and where is it located?", "expected": "ner_or_sentiment"},
    {"text": "Is the feedback about Google CEO Sundar Pichai positive?", "expected": "ner_or_sentiment"},
    {"text": "Summarize opinions about Microsoft and identify key executives.", "expected": "ner_or_sentiment"},
    {"text": "Give me a brief overview of what customers feel about Amazon.", "expected": "summarization_or_sentiment"},
    {"text": "Find out if people love or hate the new iPhone release.", "expected": "ner_or_sentiment"},
    {"text": "What are people saying about climate change and who are the key figures?", "expected": "ner_or_sentiment"},
    {"text": "Analyze this report about Jeff Bezos and Blue Origin.", "expected": "ner_or_sentiment"},
    {"text": "Tell me the main points and sentiment of this AI research paper.", "expected": "summarization_or_sentiment"},
]

# Experiment 3: Failure recovery requests (NER focused)
FAILURE_REQUESTS = [
    {"text": "Extract entities: Bill Gates founded Microsoft in Albuquerque.", "expected": "ner"},
    {"text": "Find named entities: Warren Buffett leads Berkshire Hathaway in Omaha.", "expected": "ner"},
    {"text": "Identify persons and organizations: Jack Ma founded Alibaba in Hangzhou China.", "expected": "ner"},
    {"text": "Who are the people mentioned: Jensen Huang is the CEO of NVIDIA in Santa Clara.", "expected": "ner"},
    {"text": "Extract entities: Sam Altman leads OpenAI which is based in San Francisco.", "expected": "ner"},
]

# Experiment 4: Latency test requests
LATENCY_REQUESTS = [
    {"text": "I really enjoyed this movie, it was brilliant!"},
    {"text": "Summarize this: Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
    {"text": "Find entities: Barack Obama served as president in Washington DC."},
    {"text": "This service is absolutely terrible and I want a refund."},
    {"text": "Extract named entities: Jeff Bezos founded Amazon in Seattle Washington."},
    {"text": "Brief summary: The stock market experienced significant volatility today due to inflation concerns."},
    {"text": "I am so happy with the quality of this product!"},
    {"text": "Who is mentioned: Satya Nadella leads Microsoft based in Redmond Washington."},
    {"text": "Condense this: Renewable energy sources like solar and wind are becoming increasingly cost competitive."},
    {"text": "The customer support was horrible and completely unhelpful."},
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def send_request(url: str, text: str) -> dict:
    """Send request to orchestrator and return result"""
    try:
        start = time.time()
        response = requests.post(
            url,
            json={"text": text},
            timeout=60
        )
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            data["actual_latency_ms"] = round(latency, 2)
            data["success"] = True
            return data
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "actual_latency_ms": round(latency, 2)
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "actual_latency_ms": 0
        }

def save_to_csv(results: list, filename: str):
    """Save results to CSV file"""
    if not results:
        return
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filename}")

# ============================================================
# EXPERIMENT 1: ROUTING ACCURACY
# ============================================================

def run_experiment_1():
    print("\n" + "="*60)
    print("EXPERIMENT 1: ROUTING ACCURACY")
    print("="*60)
    
    baseline_results = []
    agentic_results = []
    
    for i, req in enumerate(CLEAR_REQUESTS):
        print(f"Request {i+1}/{len(CLEAR_REQUESTS)}: {req['text'][:50]}...")
        
        # Test baseline
        baseline = send_request(BASELINE_URL, req["text"])
        baseline_correct = baseline.get("routing_decision") == req["expected"]
        baseline_results.append({
            "request_id": i+1,
            "text": req["text"][:100],
            "expected": req["expected"],
            "routing_decision": baseline.get("routing_decision", "error"),
            "correct": baseline_correct,
            "latency_ms": baseline.get("actual_latency_ms", 0),
            "success": baseline.get("success", False)
        })
        
        time.sleep(1)
        
        # Test agentic
        agentic = send_request(AGENTIC_URL, req["text"])
        agentic_correct = agentic.get("routing_decision") == req["expected"]
        agentic_results.append({
            "request_id": i+1,
            "text": req["text"][:100],
            "expected": req["expected"],
            "routing_decision": agentic.get("routing_decision", "error"),
            "correct": agentic_correct,
            "agent_reasoning": agentic.get("agent_reasoning", "")[:200],
            "latency_ms": agentic.get("actual_latency_ms", 0),
            "success": agentic.get("success", False)
        })
        
        time.sleep(2)
    
    # Calculate accuracy
    baseline_accuracy = sum(1 for r in baseline_results if r["correct"]) / len(baseline_results) * 100
    agentic_accuracy = sum(1 for r in agentic_results if r["correct"]) / len(agentic_results) * 100
    
    print(f"\nBaseline Accuracy: {baseline_accuracy:.1f}%")
    print(f"Agentic Accuracy:  {agentic_accuracy:.1f}%")
    
    save_to_csv(baseline_results, "exp1_baseline_routing.csv")
    save_to_csv(agentic_results, "exp1_agentic_routing.csv")
    
    return baseline_accuracy, agentic_accuracy

# ============================================================
# EXPERIMENT 2: AMBIGUOUS REQUEST HANDLING
# ============================================================

def run_experiment_2():
    print("\n" + "="*60)
    print("EXPERIMENT 2: AMBIGUOUS REQUEST HANDLING")
    print("="*60)
    
    results = []
    
    for i, req in enumerate(AMBIGUOUS_REQUESTS):
        print(f"Request {i+1}/{len(AMBIGUOUS_REQUESTS)}: {req['text'][:50]}...")
        
        # Test baseline
        baseline = send_request(BASELINE_URL, req["text"])
        time.sleep(1)
        
        # Test agentic
        agentic = send_request(AGENTIC_URL, req["text"])
        time.sleep(2)
        
        results.append({
            "request_id": i+1,
            "text": req["text"][:100],
            "expected_category": req["expected"],
            "baseline_routing": baseline.get("routing_decision", "error"),
            "baseline_success": baseline.get("success", False),
            "baseline_latency_ms": baseline.get("actual_latency_ms", 0),
            "agentic_routing": agentic.get("routing_decision", "error"),
            "agentic_reasoning": agentic.get("agent_reasoning", "")[:200],
            "agentic_success": agentic.get("success", False),
            "agentic_latency_ms": agentic.get("actual_latency_ms", 0),
        })
        
        print(f"  Baseline: {baseline.get('routing_decision', 'error')}")
        print(f"  Agentic:  {agentic.get('routing_decision', 'error')} | {agentic.get('agent_reasoning', '')[:80]}")
    
    save_to_csv(results, "exp2_ambiguous_requests.csv")
    return results

# ============================================================
# EXPERIMENT 3: FAILURE RECOVERY
# ============================================================

def run_experiment_3():
    print("\n" + "="*60)
    print("EXPERIMENT 3: FAILURE RECOVERY")
    print("="*60)
    print("NOTE: Please manually kill NER service pod now!")
    print("Run: kubectl scale deployment ner-service --replicas=0")
    print("Then press Enter to continue...")
    input()
    
    results = []
    
    for i, req in enumerate(FAILURE_REQUESTS):
        print(f"Request {i+1}/{len(FAILURE_REQUESTS)}: {req['text'][:50]}...")
        
        # Test baseline
        baseline = send_request(BASELINE_URL, req["text"])
        time.sleep(1)
        
        # Test agentic
        agentic = send_request(AGENTIC_URL, req["text"])
        time.sleep(2)
        
        results.append({
            "request_id": i+1,
            "text": req["text"][:100],
            "expected": req["expected"],
            "baseline_routing": baseline.get("routing_decision", "error"),
            "baseline_success": baseline.get("success", False),
            "baseline_error": baseline.get("error", ""),
            "baseline_latency_ms": baseline.get("actual_latency_ms", 0),
            "agentic_routing": agentic.get("routing_decision", "error"),
            "agentic_reasoning": agentic.get("agent_reasoning", "")[:200],
            "agentic_success": agentic.get("success", False),
            "agentic_error": agentic.get("error", ""),
            "agentic_latency_ms": agentic.get("actual_latency_ms", 0),
        })
        
        print(f"  Baseline: {baseline.get('routing_decision', 'error')} | Success: {baseline.get('success', False)}")
        print(f"  Agentic:  {agentic.get('routing_decision', 'error')} | Success: {agentic.get('success', False)}")
    
    print("\nRestoring NER service...")
    print("Run: kubectl scale deployment ner-service --replicas=1")
    input("Press Enter after NER service is restored...")
    
    save_to_csv(results, "exp3_failure_recovery.csv")
    return results

# ============================================================
# EXPERIMENT 4: LATENCY
# ============================================================

def run_experiment_4():
    print("\n" + "="*60)
    print("EXPERIMENT 4: LATENCY AND PERFORMANCE")
    print("="*60)
    
    baseline_latencies = []
    agentic_latencies = []
    results = []
    
    # Run 3 rounds
    for round_num in range(3):
        print(f"\nRound {round_num+1}/3")
        for i, req in enumerate(LATENCY_REQUESTS):
            print(f"  Request {i+1}/{len(LATENCY_REQUESTS)}: {req['text'][:40]}...")
            
            # Baseline
            baseline = send_request(BASELINE_URL, req["text"])
            if baseline.get("success"):
                baseline_latencies.append(baseline["actual_latency_ms"])
            time.sleep(1)
            
            # Agentic
            agentic = send_request(AGENTIC_URL, req["text"])
            if agentic.get("success"):
                agentic_latencies.append(agentic["actual_latency_ms"])
            time.sleep(2)
            
            results.append({
                "round": round_num+1,
                "request_id": i+1,
                "text": req["text"][:100],
                "baseline_latency_ms": baseline.get("actual_latency_ms", 0),
                "baseline_routing": baseline.get("routing_decision", "error"),
                "baseline_success": baseline.get("success", False),
                "agentic_latency_ms": agentic.get("actual_latency_ms", 0),
                "agentic_routing": agentic.get("routing_decision", "error"),
                "agentic_success": agentic.get("success", False),
            })
    
    # Summary statistics
    if baseline_latencies:
        print(f"\nBaseline Latency:")
        print(f"  Average: {sum(baseline_latencies)/len(baseline_latencies):.2f}ms")
        print(f"  Min:     {min(baseline_latencies):.2f}ms")
        print(f"  Max:     {max(baseline_latencies):.2f}ms")
    
    if agentic_latencies:
        print(f"\nAgentic Latency:")
        print(f"  Average: {sum(agentic_latencies)/len(agentic_latencies):.2f}ms")
        print(f"  Min:     {min(agentic_latencies):.2f}ms")
        print(f"  Max:     {max(agentic_latencies):.2f}ms")
    
    save_to_csv(results, "exp4_latency.csv")
    return baseline_latencies, agentic_latencies

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("MLOPS + AGENTIC AI EXPERIMENT SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Run all experiments
    b_acc, a_acc = run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    b_lat, a_lat = run_experiment_4()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Experiment 1 - Routing Accuracy:")
    print(f"  Baseline: {b_acc:.1f}%")
    print(f"  Agentic:  {a_acc:.1f}%")
    if b_lat and a_lat:
        print(f"Experiment 4 - Average Latency:")
        print(f"  Baseline: {sum(b_lat)/len(b_lat):.2f}ms")
        print(f"  Agentic:  {sum(a_lat)/len(a_lat):.2f}ms")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All results saved to CSV files!")
