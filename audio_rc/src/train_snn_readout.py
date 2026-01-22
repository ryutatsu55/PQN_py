import json
import numpy as np
import glob
import argparse
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import os
import shutil

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
import config
import src.PQN_RNN_onGPU as PQN_RNN_onGPU

# Load Config
cfg = config.Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["snn", "feature", "linear"],
    default="feature",
    help="snn: run SNN to compute features, feature: load saved feature .npy, linear: use cochleagram directly",
)
parser.add_argument(
    "--cells",
    "-c",
    type=int,
    default=cfg.N,
    help="number of reservoir cells (default: cfg)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=cfg.SEED,
    help="random seed (default: cfg)",
)
args = parser.parse_args()


# ================================
# 1. Data Loading Function
# ================================
def load_dataset_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_test_paths = []

    # --- Initialize Reservoir (only for SNN mode) ---
    sim = None
    reservoir_state = None
    
    if args.mode != "linear":
        # Initialize reservoir state using RNN_config
        # Note: RNN_config.init_reservoir does not require input_size argument in the current version
        reservoir_state = config.init_reservoir()
        
        # Initialize the GPU simulation class
        sim = PQN_RNN_onGPU.PQN_Reservoir_GPU(reservoir_state, cfg)

    # Define classes and paths
    # Assuming directory structure: audio_rc/reservoir_inputs/train/coch_female_zero/ etc.
    # Class mapping:
    
    classes = [
        {"name": "female_zero", "label": 0, "dir": "coch_female_zero"},
        {"name": "female_one",  "label": 1, "dir": "coch_female_one"},
        {"name": "male_zero",   "label": 2, "dir": "coch_male_zero"},
        {"name": "male_one",    "label": 3, "dir": "coch_male_one"},
    ]
    
    # Base directories (You might need to adjust "audio_rc" part based on your actual structure)
    train_base_dir = "audio_rc/reservoir_inputs/train"
    test_base_dir = "audio_rc/reservoir_inputs/test"
    output_base_dir = "audio_rc/reservoir_outputs" # For feature mode

    # Check if directories exist to avoid errors (Optional sanity check)
    if args.mode in ["snn", "linear"]:
        if not os.path.exists(train_base_dir):
             print(f"Warning: Train directory {train_base_dir} not found.")

    # ==========================
    # TRAIN LOOP
    # ==========================
    for cls in classes:
        label = cls["label"]
        name = cls["name"]
        subdir = cls["dir"]
        
        if args.mode == "snn":
            # Search for npy files
            search_path = os.path.join(train_base_dir, subdir, "*.npy")
            path_list = glob.glob(search_path)
            
            for path in tqdm(path_list, desc=f"TRAIN {name.upper()}"):
                coch = np.load(path)
                
                # Execute simulation
                # Updated to match recognition_test.py signature
                feat = PQN_RNN_onGPU.main(
                    input_data=coch,
                    reservoir_state=reservoir_state,
                    return_feature=True,
                    is_debug_print=False,
                    record=False,       # No need to record full voltage history for training
                    S_durt=cfg.INPUT_DT, # Audio typically uses INPUT_DT
                    cfg=cfg,
                    sim=sim
                )
                X_train.append(feat)
                y_train.append(label)

        elif args.mode == "feature":
            # Load pre-calculated features
            # Assuming features are saved in a similar structure under reservoir_outputs
            # e.g., audio_rc/reservoir_outputs/train/features_female_zero/*.npy
            feat_subdir = f"features_{name}"
            search_path = os.path.join(output_base_dir, "train", feat_subdir, "*.npy")
            
            for path in tqdm(glob.glob(search_path), desc=f"TRAIN {name.upper()}"):
                feat = np.load(path)
                X_train.append(feat)
                y_train.append(label)
                
        elif args.mode == "linear":
            search_path = os.path.join(train_base_dir, subdir, "*.npy")
            for path in tqdm(glob.glob(search_path), desc=f"TRAIN {name.upper()} (linear)"):
                coch = np.load(path)
                X_train.append(coch)
                y_train.append(label)

    # ==========================
    # TEST LOOP
    # ==========================
    for cls in classes:
        label = cls["label"]
        name = cls["name"]
        subdir = cls["dir"]

        if args.mode == "snn":
            search_path = os.path.join(test_base_dir, subdir, "*.npy")
            path_list = glob.glob(search_path)
            
            for path in tqdm(path_list, desc=f"TEST {name.upper()}"):
                coch = np.load(path)
                
                feat = PQN_RNN_onGPU.main(
                    input_data=coch,
                    reservoir_state=reservoir_state,
                    return_feature=True,
                    is_debug_print=False,
                    record=False,
                    S_durt=cfg.INPUT_DT,
                    cfg=cfg,
                    sim=sim
                )
                X_test.append(feat)
                y_test.append(label)
                X_test_paths.append(path)

        elif args.mode == "feature":
            feat_subdir = f"features_{name}"
            search_path = os.path.join(output_base_dir, "test", feat_subdir, "*.npy")
            
            for path in tqdm(glob.glob(search_path), desc=f"TEST {name.upper()}"):
                feat = np.load(path)
                X_test.append(feat)
                y_test.append(label)
                X_test_paths.append(path)

        elif args.mode == "linear":
            search_path = os.path.join(test_base_dir, subdir, "*.npy")
            for path in tqdm(glob.glob(search_path), desc=f"TEST {name.upper()} (linear)"):
                coch = np.load(path)
                X_test.append(coch)
                y_test.append(label)
                X_test_paths.append(path)

    # Shuffle datasets
    perm_train = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in perm_train]
    y_train = [y_train[i] for i in perm_train]

    perm_test = np.random.permutation(len(X_test))
    X_test = [X_test[i] for i in perm_test]
    y_test = [y_test[i] for i in perm_test]

    if len(X_train) == 0:
        print("Error: No training data found. Please check input directories.")
        return [], np.array([]), [], np.array([]), []

    # --- Data Flattening Logic (Kept as is) ---
    # Determine fixed time length (median of training samples)


    return (
        X_train,
        np.array(y_train),
        X_test,
        np.array(y_test),
        X_test_paths,
    )

# 修正予定
def pad_and_flatten(X_source):
    Ts = [x.shape[0] for x in X_source]
    T_fixed = int(np.median(Ts))
    print(f"Fixed time length determined: {T_fixed}")
    x = x[:T_fixed]
    if x.shape[0] < T_fixed:
        pad = np.zeros((T_fixed - x.shape[0], x.shape[1]), dtype=np.float32)
        x = np.vstack([x, pad])
    return x.reshape(-1)


def analyze_trajectories(X_list: list[np.ndarray], y_list: list[int], save_dir: str, dt: float) -> None:
    print("\nStarting Trajectory Analysis...")
    
    # データの前処理: 全トライアルで最小のデータ長に合わせる（時系列平均のため）
    min_len = min([x.shape[0] for x in X_list])
    X_truncated = [x[:min_len, :] for x in X_list]
    
    # 解析用にデータを結合 (Total_Time_Steps, Neurons)
    X_concat = np.vstack(X_truncated)
    
    # (ニューロンごとのばらつきを正規化)
    scaler = StandardScaler()
    X_standardized_concat = scaler.fit_transform(X_concat)
    
    # トライアルごとの形に戻す (Num_Trials, Time, Neurons)
    n_trials = len(X_truncated)
    time_steps = min_len
    n_neurons = X_truncated[0].shape[1]
    X_reshaped = X_standardized_concat.reshape(n_trials, time_steps, n_neurons)
    y_arr = np.array(y_list)

    # ==========================================
    # 1. PCA Analysis 
    # ==========================================
    pca = PCA(n_components=3)
    X_pca_concat = pca.fit_transform(X_standardized_concat)
    # (Num_Trials, Time, 3) に変形
    X_pca = X_pca_concat.reshape(n_trials, time_steps, 3)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # クラスごとに色を変えてプロット
    # クラス0:赤, 1:緑, 2:青 (必要に応じて変更してください)
    colors = ['r', 'g', 'b'] 
    labels = ['Top', 'Middle', 'Bottom']
    
    # 凡例用に一度だけラベル付きでプロットするためのフラグ
    plotted_labels = set()
    
    for i in range(n_trials):
        label_idx = int(y_arr[i])
        c = colors[label_idx % len(colors)]
        l = labels[label_idx % len(labels)]
        
        if label_idx not in plotted_labels:
            ax.plot(X_pca[i, :, 0], X_pca[i, :, 1], X_pca[i, :, 2], color=c, alpha=0.6, label=l)
            plotted_labels.add(label_idx)
        else:
            ax.plot(X_pca[i, :, 0], X_pca[i, :, 1], X_pca[i, :, 2], color=c, alpha=0.6)
            
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Trajectories in PC Subspace')
    ax.legend()
    
    save_path_pca = os.path.join(save_dir, "pca_trajectories.png")
    plt.savefig(save_path_pca)
    plt.close()
    print(f"Saved PCA plot to {save_path_pca}")

    # ==========================================
    # 2. Distance Analysis (Fig 2D 相当)
    # ==========================================
    # Instantaneous normalized distance の計算
    # d_pq(t) = (1 / sqrt(N)) * || p(t) - q(t) ||_2
    
    dist_same = []
    dist_diff = []
    
    # 全ペアについて距離を計算
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            # 時刻ごとのユークリッド距離を計算 (Time,)
            diff = X_reshaped[i] - X_reshaped[j] 
            # 論文式 (4) に基づき sqrt(N) で割って正規化
            dist_t = np.linalg.norm(diff, axis=1) / np.sqrt(n_neurons)
            
            if y_arr[i] == y_arr[j]:
                dist_same.append(dist_t)
            else:
                dist_diff.append(dist_t)
                
    dist_same = np.array(dist_same) # (Num_Same_Pairs, Time)
    dist_diff = np.array(dist_diff) # (Num_Diff_Pairs, Time)
    
    # 平均と標準偏差を計算
    mean_same = dist_same.mean(axis=0)
    std_same = dist_same.std(axis=0)
    
    mean_diff = dist_diff.mean(axis=0)
    std_diff = dist_diff.std(axis=0)

    lower_diff = mean_diff - std_diff
    lower_same = mean_same - std_same

    lower_diff = np.maximum(lower_diff, 0)
    lower_same = np.maximum(lower_same, 0)
    
    # 時間軸の作成 (秒単位)
    t_axis = np.arange(time_steps) * dt
    
    plt.figure(figsize=(8, 6))
    
    # Different inputs (Blue)
    plt.plot(t_axis, mean_diff, label='Different inputs', color='blue')
    plt.fill_between(t_axis, lower_diff, mean_diff + std_diff, color='blue', alpha=0.2)
    
    # Same inputs (Red)
    plt.plot(t_axis, mean_same, label='Same inputs', color='red')
    plt.fill_between(t_axis, lower_same, mean_same + std_same, color='red', alpha=0.2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized distance')
    plt.title('Instantaneous normalized distance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_dist = os.path.join(save_dir, "distance_analysis.png")
    plt.savefig(save_path_dist)
    plt.close()
    print(f"Saved Distance plot to {save_path_dist}")

# ================================
# 2. Linear Readout Training (Ridge Regression)
# ================================
def train_readout(X: np.ndarray, y: np.ndarray, lambda_reg: float = 1e-2) -> np.ndarray:
    num_samples, D = X.shape
    classes = np.unique(y)
    C = len(classes)

    Y = np.zeros((num_samples, C), dtype=np.float32)
    for i, label in enumerate(y):
        Y[i, label] = 1.0

    # K = X X^T : (num_samples, num_samples)
    K = X @ X.T
    I = np.eye(num_samples, dtype=np.float32)

    alpha = np.linalg.inv(K + lambda_reg * I) @ Y  # (num_samples, C)
    W_out = X.T @ alpha  # (D, C)

    return W_out


# ================================
# 3. Prediction
# ================================
def predict(W_out: np.ndarray, feat: np.ndarray) -> np.intp:
    logits = feat @ W_out  # shape = (C,)
    return np.argmax(logits)


# ================================
# 4. Evaluation
# ================================
def evaluate(W_out: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    for feat, label in zip(X, y):
        pred = predict(W_out, feat)
        if pred == label:
            correct += 1
    if len(y) == 0:
        return 0.0
    return correct / len(y)


# ================================
# 5. Main Process
# ================================
def main_train(num_of_cells: int, seed: int) -> None:
    print(f"Mode: {args.mode}")
    print(f"Number of reservoir cells: {num_of_cells}")
    print(f"Random seed: {seed}")
    print("Loading dataset...")
    
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split()

    
    analyze_trajectories(
        X_train, 
        y_train, 
        save_dir="figs",        # 修正予定
        dt=cfg.DT  # または cfg.DT (シミュレーションの時間刻みに合わせてください)
    )

    X_train_flat = np.stack([pad_and_flatten(x) for x in X_train])
    X_test_flat = np.stack([pad_and_flatten(x) for x in X_test])
    del X_train
    del X_test
    gc.collect()
    
    if len(X_train) == 0:
        return

    print("Train shape:", X_train_flat.shape)
    print("Test  shape:", X_test_flat.shape)

    # Basic stats (checking just first class vs second class for sanity)
    # Note: indices might need adjustment if class 0 or 1 is missing
    if np.any(y_train == 0):
        print("Female Zero feat mean (first 10):", X_train[y_train == 0].mean(axis=0)[:10])
    
    # 修正予定---------------------------------
    print("Training readout...")
    W_out = train_readout(X_train, y_train, lambda_reg=1e-2)

    # Evaluate
    acc_train = evaluate(W_out, X_train, y_train)
    acc_test = evaluate(W_out, X_test, y_test)
    #--------------------------------------------------

    # Label shuffle test
    y_train_shuffled = np.random.permutation(y_train)
    W_out_shuffled = train_readout(X_train, y_train_shuffled)
    acc_test_shuffled = evaluate(W_out_shuffled, X_test, y_test)
    print(f"label shuffled test accuracy: {acc_test_shuffled}")

    # --- Confusion Matrix (4x4) ---
    num_classes = 4
    misclassified = []
    conf = np.zeros((num_classes, num_classes), dtype=int)
    
    for feat, true_label, path in zip(X_test, y_test, X_test_paths):
        pred_label = predict(W_out, feat)
        conf[true_label, pred_label] += 1
        if true_label != pred_label:
            misclassified.append((path, true_label, pred_label))

    print("\nConfusion Matrix (rows=True, cols=Pred):")
    print(conf)
    
    class_names = ["F-Zero", "F-One", "M-Zero", "M-One"]
    for i in range(num_classes):
        total = conf[i].sum()
        if total > 0:
            print(f"True {class_names[i]} predicted correctly: {conf[i,i]} / {total}")

    # --- Save confusion matrix as image ---
    plt.figure(figsize=(6, 6))
    plt.imshow(conf, cmap="Blues")
    
    title_suffix = "(Linear)" if args.mode == "linear" else f", N = {num_of_cells}"
    plt.title(f"Confusion Matrix {title_suffix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # set axis ticks
    plt.xticks(np.arange(num_classes), class_names)
    plt.yticks(np.arange(num_classes), class_names)

    # annotate cells
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                str(conf[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold",
            )

    plt.colorbar()
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"confusion_matrix_{timestamp}.png"
    
    save_dir = "audio_rc/figs/confusion_matrix"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Saved {filename}")

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # Save W_out
    save_W_dir = "audio_rc/reservoir_outputs"
    os.makedirs(save_W_dir, exist_ok=True)
    np.save(os.path.join(save_W_dir, "W_out.npy"), W_out)
    print("Saved W_out.npy")

    print("\nMisclassified files:")
    if len(misclassified) == 0:
        print("  None! Perfect classification.")
    else:
        print(len(misclassified), "files misclassified.")

    # Save Results JSON
    results_dir = "audio_rc/results"
    os.makedirs(results_dir, exist_ok=True)
    
    result = {
        "mode": args.mode,
        "seed": seed,
        "num_cells": num_of_cells,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

    with open(os.path.join(results_dir, "results_mf.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    num_of_cells = args.cells
    seed = args.seed
    (num_of_cells, seed)