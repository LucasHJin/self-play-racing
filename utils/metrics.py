import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def normalize(rewards):
    min_r = np.min(rewards)
    max_r = np.max(rewards)
    return (rewards - min_r) / (max_r - min_r)

def eval_training():
    ROOT = Path(__file__).resolve().parent.parent
    data = {
        "Baseline": json.load(open(ROOT / "data" / "single_agent.json")),
        "+ Speed": json.load(open(ROOT / "data" / "single_agent_speed.json")),
        "+ Time": json.load(open(ROOT / "data" / "single_agent_time.json")),
    }
    
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange']
    for (name, d), color in zip(data.items(), colors):
        normalized = normalize(d["rewards"])
        plt.plot(d["steps"], normalized, label=name, linewidth=2, color=color, alpha=0.6)
    plt.xlabel("Training Steps")
    plt.ylabel("Normalized Rewards")
    plt.title("Learning Speed Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_speed.png", dpi=300)
    plt.show()
