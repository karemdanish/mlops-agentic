import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("graphs", exist_ok=True)

print("Loading CSV files...")
exp1_baseline = pd.read_csv("exp1_baseline_routing.csv")
exp1_agentic = pd.read_csv("exp1_agentic_routing.csv")
exp2 = pd.read_csv("exp2_ambiguous_requests.csv")
exp3 = pd.read_csv("exp3_failure_recovery.csv")
exp4 = pd.read_csv("exp4_latency.csv")

print("Generating graphs...")

# ============================================================
# GRAPH 1: Overall Routing Accuracy
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

orchestrators = ['Baseline\n(Rule-Based)', 'Agentic\n(LLM-Powered)']
accuracies = [95.0, 100.0]
colors = ['#FF6B6B', '#4ECDC4']

bars = ax.bar(orchestrators, accuracies, color=colors, width=0.4, edgecolor='black')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{acc}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylim(0, 115)
ax.set_ylabel('Routing Accuracy (%)', fontsize=12)
ax.set_title('Experiment 1: Overall Routing Accuracy\nBaseline vs Agentic Orchestrator', 
             fontsize=13, fontweight='bold')
ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect Accuracy')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/graph1_routing_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 1 saved!")

# ============================================================
# GRAPH 2: Routing Accuracy By Task Type
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

tasks = ['Sentiment', 'Summarization', 'NER']
task_keys = ['sentiment', 'summarization', 'ner']

baseline_by_task = []
agentic_by_task = []

for task in task_keys:
    b = exp1_baseline[exp1_baseline['expected'] == task]
    a = exp1_agentic[exp1_agentic['expected'] == task]
    baseline_by_task.append(round(b['correct'].mean() * 100, 1))
    agentic_by_task.append(round(a['correct'].mean() * 100, 1))

x = np.arange(len(tasks))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_by_task, width, label='Baseline', 
               color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x + width/2, agentic_by_task, width, label='Agentic', 
               color='#4ECDC4', edgecolor='black')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.0f}%', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.0f}%', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Experiment 1: Routing Accuracy by Task Type', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12)
ax.set_ylim(0, 120)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/graph2_accuracy_by_task.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 2 saved!")

# ============================================================
# GRAPH 3: Experiment 1 Latency By Task
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

baseline_latency_by_task = []
agentic_latency_by_task = []

for task in task_keys:
    b = exp1_baseline[exp1_baseline['expected'] == task]['latency_ms'].mean()
    a = exp1_agentic[exp1_agentic['expected'] == task]['latency_ms'].mean()
    baseline_latency_by_task.append(round(b, 2))
    agentic_latency_by_task.append(round(a, 2))

x = np.arange(len(tasks))
bars1 = ax.bar(x - width/2, baseline_latency_by_task, width, label='Baseline',
               color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x + width/2, agentic_latency_by_task, width, label='Agentic',
               color='#4ECDC4', edgecolor='black')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{bar.get_height():.0f}ms', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{bar.get_height():.0f}ms', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Average Latency (ms)', fontsize=12)
ax.set_title('Experiment 1: Average Latency by Task Type', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/graph3_latency_by_task.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 3 saved!")

# ============================================================
# GRAPH 4: Failure Recovery Results
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

categories = ['Successful\nResponses', 'Failed\nResponses']
baseline_counts = [
    int(exp3['baseline_success'].sum()),
    int(len(exp3) - exp3['baseline_success'].sum())
]
agentic_counts = [
    int(exp3['agentic_success'].sum()),
    int(len(exp3) - exp3['agentic_success'].sum())
]

x = np.arange(len(categories))
bars1 = ax.bar(x - width/2, baseline_counts, width, label='Baseline',
               color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x + width/2, agentic_counts, width, label='Agentic',
               color='#4ECDC4', edgecolor='black')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            str(int(bar.get_height())), ha='center', va='bottom', 
            fontsize=14, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            str(int(bar.get_height())), ha='center', va='bottom', 
            fontsize=14, fontweight='bold')

ax.set_ylabel('Number of Requests', fontsize=12)
ax.set_title('Experiment 3: Failure Recovery\nDuring NER Service Outage (5 Requests)', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 7)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/graph4_failure_recovery.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 4 saved!")

# ============================================================
# GRAPH 5: Latency Distribution Box Plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Only use successful requests
baseline_lat = exp4[exp4['baseline_success'] == True]['baseline_latency_ms'].tolist()
agentic_lat = exp4[exp4['agentic_success'] == True]['agentic_latency_ms'].tolist()

bp = ax.boxplot([baseline_lat, agentic_lat],
                labels=['Baseline\n(Rule-Based)', 'Agentic\n(LLM-Powered)'],
                patch_artist=True)

bp['boxes'][0].set_facecolor('#FF6B6B')
bp['boxes'][1].set_facecolor('#4ECDC4')
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('Experiment 4: Latency Distribution\nBaseline vs Agentic Orchestrator', 
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/graph5_latency_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 5 saved!")

# ============================================================
# GRAPH 6: Latency Over Time (Round 1)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

round1 = exp4[exp4['round'] == 1].reset_index(drop=True)

# Only plot successful requests
baseline_plot = round1['baseline_latency_ms'].where(
    round1['baseline_success'] == True)
agentic_plot = round1['agentic_latency_ms'].where(
    round1['agentic_success'] == True)

ax.plot(range(1, len(round1)+1), baseline_plot,
        marker='o', color='#FF6B6B', label='Baseline', 
        linewidth=2, markersize=8)
ax.plot(range(1, len(round1)+1), agentic_plot,
        marker='s', color='#4ECDC4', label='Agentic', 
        linewidth=2, markersize=8)

# Mark baseline failures
failures = round1[round1['baseline_success'] == False]
for idx in failures.index:
    ax.axvline(x=idx+1, color='red', linestyle='--', alpha=0.5)
    ax.text(idx+1, ax.get_ylim()[1]*0.9, 'Baseline\nFailed', 
            ha='center', color='red', fontsize=9)

ax.set_xlabel('Request Number', fontsize=12)
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('Experiment 4: Latency Over Sequential Requests (Round 1)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(range(1, len(round1)+1))
plt.tight_layout()
plt.savefig('graphs/graph6_latency_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 6 saved!")

# ============================================================
# GRAPH 7: Average Latency Comparison (Exp 4)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

metrics = ['Average', 'Minimum', 'Maximum']
baseline_stats = [
    round(exp4[exp4['baseline_success']==True]['baseline_latency_ms'].mean(), 2),
    round(exp4[exp4['baseline_success']==True]['baseline_latency_ms'].min(), 2),
    round(exp4[exp4['baseline_success']==True]['baseline_latency_ms'].max(), 2)
]
agentic_stats = [
    round(exp4[exp4['agentic_success']==True]['agentic_latency_ms'].mean(), 2),
    round(exp4[exp4['agentic_success']==True]['agentic_latency_ms'].min(), 2),
    round(exp4[exp4['agentic_success']==True]['agentic_latency_ms'].max(), 2)
]

x = np.arange(len(metrics))
bars1 = ax.bar(x - width/2, baseline_stats, width, label='Baseline',
               color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x + width/2, agentic_stats, width, label='Agentic',
               color='#4ECDC4', edgecolor='black')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
            f'{bar.get_height():.0f}ms', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
            f'{bar.get_height():.0f}ms', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('Experiment 4: Latency Statistics\nBaseline vs Agentic Orchestrator', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/graph7_latency_stats.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 7 saved!")

# ============================================================
# GRAPH 8: Ambiguous Request Routing Disagreements
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Count agreements and disagreements
agreements = sum(1 for _, row in exp2.iterrows() 
                if row['baseline_routing'] == row['agentic_routing'])
disagreements = len(exp2) - agreements

categories = ['Same Routing\nDecision', 'Different Routing\nDecision']
counts = [agreements, disagreements]
colors_pie = ['#95E1D3', '#F38181']

bars = ax.bar(categories, counts, color=colors_pie, edgecolor='black', width=0.4)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Number of Requests', fontsize=12)
ax.set_title('Experiment 2: Ambiguous Request Handling\nRouting Agreement between Baseline and Agentic', 
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/graph8_ambiguous_routing.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 8 saved!")

# ============================================================
# GRAPH 9: Overall Summary Dashboard
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top Left: Routing Accuracy
axes[0,0].bar(['Baseline', 'Agentic'], [95.0, 100.0],
              color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
axes[0,0].set_title('Routing Accuracy (Exp 1)', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('%')
axes[0,0].set_ylim(0, 115)
for i, v in enumerate([95.0, 100.0]):
    axes[0,0].text(i, v+1, f'{v}%', ha='center', fontsize=12, fontweight='bold')
axes[0,0].grid(axis='y', alpha=0.3)

# Top Right: Failure Recovery
axes[0,1].bar(['Baseline', 'Agentic'],
              [int(exp3['baseline_success'].sum()), 
               int(exp3['agentic_success'].sum())],
              color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
axes[0,1].set_title('Failure Recovery - Successful Responses (Exp 3)', 
                    fontsize=12, fontweight='bold')
axes[0,1].set_ylabel('Count out of 5')
axes[0,1].set_ylim(0, 7)
for i, v in enumerate([int(exp3['baseline_success'].sum()), 
                        int(exp3['agentic_success'].sum())]):
    axes[0,1].text(i, v+0.05, str(v), ha='center', fontsize=14, fontweight='bold')
axes[0,1].grid(axis='y', alpha=0.3)

# Bottom Left: Average Latency
avg_baseline = round(exp4[exp4['baseline_success']==True]['baseline_latency_ms'].mean(), 2)
avg_agentic = round(exp4[exp4['agentic_success']==True]['agentic_latency_ms'].mean(), 2)
axes[0,2] if False else None
axes[1,0].bar(['Baseline', 'Agentic'], [avg_baseline, avg_agentic],
              color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
axes[1,0].set_title('Average Latency (Exp 4)', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('ms')
for i, v in enumerate([avg_baseline, avg_agentic]):
    axes[1,0].text(i, v+30, f'{v:.0f}ms', ha='center', fontsize=11, fontweight='bold')
axes[1,0].grid(axis='y', alpha=0.3)

# Bottom Right: Ambiguous Routing Disagreements
axes[1,1].bar(['Same Decision', 'Different Decision'], 
              [agreements, disagreements],
              color=['#95E1D3', '#F38181'], edgecolor='black')
axes[1,1].set_title('Ambiguous Request Routing (Exp 2)', fontsize=12, fontweight='bold')
axes[1,1].set_ylabel('Count out of 10')
axes[1,1].set_ylim(0, 12)
for i, v in enumerate([agreements, disagreements]):
    axes[1,1].text(i, v+0.1, str(v), ha='center', fontsize=14, fontweight='bold')
axes[1,1].grid(axis='y', alpha=0.3)

plt.suptitle('Overall Comparison: Baseline vs Agentic Orchestrator\nAll Experiments Summary',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/graph9_overall_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graph 9 saved!")

print("\n" + "="*50)
print("ALL 9 GRAPHS GENERATED SUCCESSFULLY!")
print("="*50)
print("\nGraphs saved in: ~/mlops-agentic/experiments/graphs/")
for f in sorted(os.listdir('graphs')):
    print(f"  - {f}")
