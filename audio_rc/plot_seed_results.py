import json
import numpy as np
import matplotlib.pyplot as plt

results = []

# JSONL 読み込み
with open("audio_rc/results/results.jsonl") as f:
    for line in f:
        results.append(json.loads(line))

# ソートしておくときれい
results = sorted(results, key=lambda x: x["seed"])

seeds = [r["seed"] for r in results]
acc_test = [r["acc_test"] for r in results]
acc_train = [r["acc_train"] for r in results]

# --- Test Accuracy プロット ---
plt.figure(figsize=(8, 4))
plt.plot(seeds, acc_test, marker="o", label="Test Accuracy")
plt.plot(seeds, acc_train, marker="o", label="Train Accuracy", alpha=0.5)
plt.xlabel("Seed")
plt.ylabel("Accuracy")
plt.title("Accuracy over different seeds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("audio_rc/figs/accuracy_over_seeds.png")
plt.show()

# --- 統計情報 ---
print("Mean test accuracy:", np.mean(acc_test))
print("Std  test accuracy:", np.std(acc_test))
