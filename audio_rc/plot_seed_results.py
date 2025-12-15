import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# =========================
# settings
# =========================
RESULTS_PATH = "audio_rc/results/results.jsonl"
TARGET_NUM_CELLS = 48  # 今回は 48 を可視化（将来 sweep 可能）

# =========================
# load jsonl
# =========================
records = []
with open(RESULTS_PATH, "r") as f:
    for line in f:
        records.append(json.loads(line))

# =========================
# aggregate
#   (mode, num_cells) -> list of acc_test
# =========================
acc_dict = defaultdict(list)

for r in records:
    key = (r["mode"], r["num_cells"])
    acc_dict[key].append(r["acc_test"])

# =========================
# prepare data
# =========================
labels = []
means = []
stds = []
scatter_data = []  # snn 用

# ---- snn / linear ----
for mode in ("snn", "linear"):
    key = (mode, TARGET_NUM_CELLS)
    if key not in acc_dict:
        raise ValueError(f"{mode} result not found")

    accs = acc_dict[key]
    labels.append(mode)
    means.append(np.mean(accs))
    stds.append(np.std(accs))
    scatter_data.append(accs if mode == "snn" else None)

# =========================
# plot
# =========================
x = np.arange(len(labels))

plt.figure(figsize=(5, 4))

# --- bar: mean ---
plt.bar(
    x,
    means,
    yerr=stds,
    capsize=6,
    alpha=0.6,
)

# --- scatter: snn only ---
for i, accs in enumerate(scatter_data):
    if accs is None:
        continue

    jitter = 0.1 * np.random.randn(len(accs))
    plt.plot(
        x[i] + jitter,
        accs,
        "o",
        markersize=5,
        alpha=0.9,
    )

# --- axes ---
plt.xticks(x, labels)
plt.xlabel("Model")
plt.ylabel("Test accuracy")
plt.ylim(0.0, 1.01)

plt.tight_layout()
plt.savefig("audio_rc/figs/accuracy_bar_plot.png")

results = []

# JSONL 読み込み
with open("audio_rc/results/results.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record.get("mode") == "linear":
            continue
        results.append(record)

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

# --- 統計情報 ---
print("Mean test accuracy:", np.mean(acc_test))
print("Std  test accuracy:", np.std(acc_test))
