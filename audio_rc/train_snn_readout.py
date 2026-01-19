import json
import os
import numpy as np
import glob
import argparse
from datetime import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)  # HACK: 親ディレクトリをパスに追加
import GPU_SNN_simulation

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
    default=100,
    help="number of reservoir cells (default: 100)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="random seed (default: 123)",
)
args = parser.parse_args()


# ================================
# 1. データ読み込み関数
# ================================
def load_dataset_split(
    num_of_cells: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_test_paths = []

    # determine input dimension M (number of cochleagram channels)
    sample_paths = glob.glob("audio_rc/reservoir_inputs/train/coch_zero/*.npy")
    if len(sample_paths) == 0:
        sample_paths = glob.glob("audio_rc/reservoir_inputs/train/coch_one/*.npy")
    if len(sample_paths) == 0:
        raise RuntimeError("No cochleagram .npy files found in train directories.")
    sample_coch = np.load(sample_paths[0])
    input_size = sample_coch.shape[1]
    reservoir_state = GPU_SNN_simulation.init_reservoir(
        N=num_of_cells, seed=seed, input_size=input_size
    )
    # If linear mode, we do not initialize or use the reservoir
    if args.mode == "linear":
        reservoir_state = None

    # ----- TRAIN -----
    if args.mode == "snn":
        # ZERO
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/train/coch_zero/*.npy"),
            desc="TRAIN ZERO",
        ):
            coch = np.load(path)
            feat = GPU_SNN_simulation.main(
                input_data=coch,
                reservoir_state=reservoir_state,
                label="zero",
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
            )
            X_train.append(feat)
            y_train.append(0)

        # ONE
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/train/coch_one/*.npy"),
            desc="TRAIN ONE",
        ):
            coch = np.load(path)
            feat = GPU_SNN_simulation.main(
                input_data=coch,
                reservoir_state=reservoir_state,
                label="one",
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
            )
            X_train.append(feat)
            y_train.append(1)
    elif args.mode == "feature":
        # ZERO features
        for path in tqdm(
            glob.glob("audio_rc/reservoir_outputs/train/features_zero/*.npy"),
            desc="TRAIN ZERO",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(0)
        # ONE features
        for path in tqdm(
            glob.glob("audio_rc/reservoir_outputs/train/features_one/*.npy"),
            desc="TRAIN ONE",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(1)
    elif args.mode == "linear":
        # ZERO
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/train/coch_zero/*.npy"),
            desc="TRAIN ZERO (linear)",
        ):
            coch = np.load(path)
            X_train.append(coch)
            y_train.append(0)

        # ONE
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/train/coch_one/*.npy"),
            desc="TRAIN ONE (linear)",
        ):
            coch = np.load(path)
            X_train.append(coch)
            y_train.append(1)

    # ----- TEST -----
    if args.mode == "snn":
        # ZERO
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/test/coch_zero/*.npy"),
            desc="TEST ZERO",
        ):
            coch = np.load(path)
            feat = GPU_SNN_simulation.main(
                input_data=coch,
                reservoir_state=reservoir_state,
                label="zero",
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
            )
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)

        # ONE
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/test/coch_one/*.npy"), desc="TEST ONE"
        ):
            coch = np.load(path)
            feat = GPU_SNN_simulation.main(
                input_data=coch,
                reservoir_state=reservoir_state,
                label="one",
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
            )
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)
    elif args.mode == "feature":
        # ZERO features
        for path in tqdm(
            glob.glob("audio_rc/reservoir_outputs/test/features_zero/*.npy"),
            desc="TEST ZERO",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)
        # ONE features
        for path in tqdm(
            glob.glob("audio_rc/reservoir_outputs/test/features_one/*.npy"),
            desc="TEST ONE",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)
    elif args.mode == "linear":
        # ZERO
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/test/coch_zero/*.npy"),
            desc="TEST ZERO (linear)",
        ):
            coch = np.load(path)
            X_test.append(coch)
            y_test.append(0)
            X_test_paths.append(path)

        # ONE
        for path in tqdm(
            glob.glob("audio_rc/reservoir_inputs/test/coch_one/*.npy"),
            desc="TEST ONE (linear)",
        ):
            coch = np.load(path)
            X_test.append(coch)
            y_test.append(1)
            X_test_paths.append(path)

    # shuffle
    perm_train = np.random.permutation(len(X_train))

    X_train = [X_train[i] for i in perm_train]
    y_train = [y_train[i] for i in perm_train]

    perm_test = np.random.permutation(len(X_test))

    X_test = [X_test[i] for i in perm_test]
    y_test = [y_test[i] for i in perm_test]

    # after loading X_train (list of (T, M))
    Ts = [x.shape[0] for x in X_train]
    T_fixed = int(np.median(Ts))  # ← まずこれ

    def pad_and_flatten(x):
        x = x[:T_fixed]
        if x.shape[0] < T_fixed:
            pad = np.zeros((T_fixed - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.vstack([x, pad])
        return x.reshape(-1)

    X_train_flat = np.stack([pad_and_flatten(x) for x in X_train])
    X_test_flat = np.stack([pad_and_flatten(x) for x in X_test])

    return (
        X_train_flat,
        np.array(y_train),
        X_test_flat,
        np.array(y_test),
        X_test_paths,
    )


def analyze_trajectories(
    X_list: list[np.ndarray], y_list: list[int], save_dir: str, dt: float
) -> None:
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
    ax = fig.add_subplot(111, projection="3d")

    # クラスごとに色を変えてプロット
    # クラス0:赤, 1:緑 (必要に応じて変更してください)
    colors = ["r", "g"]
    labels = ["Zero", "One"]

    # 凡例用に一度だけラベル付きでプロットするためのフラグ
    plotted_labels = set()

    for i in range(n_trials):
        label_idx = int(y_arr[i])
        c = colors[label_idx % len(colors)]
        l = labels[label_idx % len(labels)]

        if label_idx not in plotted_labels:
            ax.plot(
                X_pca[i, :, 0],
                X_pca[i, :, 1],
                X_pca[i, :, 2],
                color=c,
                alpha=0.6,
                label=l,
            )
            plotted_labels.add(label_idx)
        else:
            ax.plot(X_pca[i, :, 0], X_pca[i, :, 1], X_pca[i, :, 2], color=c, alpha=0.6)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Trajectories in PC Subspace")
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

    dist_same = np.array(dist_same)  # (Num_Same_Pairs, Time)
    dist_diff = np.array(dist_diff)  # (Num_Diff_Pairs, Time)

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
    plt.plot(t_axis, mean_diff, label="Different inputs", color="blue")
    plt.fill_between(t_axis, lower_diff, mean_diff + std_diff, color="blue", alpha=0.2)

    # Same inputs (Red)
    plt.plot(t_axis, mean_same, label="Same inputs", color="red")
    plt.fill_between(t_axis, lower_same, mean_same + std_same, color="red", alpha=0.2)

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized distance")
    plt.title("Instantaneous normalized distance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path_dist = os.path.join(save_dir, "distance_analysis.png")
    plt.savefig(save_path_dist)
    plt.close()
    print(f"Saved Distance plot to {save_path_dist}")


# ================================
# 2. 線形 readout の学習 (ridge regression)
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
# 3. 推論
# ================================
def predict(W_out: np.ndarray, feat: np.ndarray) -> np.intp:
    logits = feat @ W_out  # shape = (C,)
    return np.argmax(logits)


# ================================
# 4. テストの精度測定
# ================================
def evaluate(W_out: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    for feat, label in zip(X, y):
        pred = predict(W_out, feat)
        if pred == label:
            correct += 1
    return correct / len(y)


# ================================
# 5. メイン処理
# ================================
def main_train(num_of_cells: int, seed: int) -> None:
    print(f"Mode: {args.mode}")
    print(f"Number of reservoir cells: {num_of_cells}")
    print(f"Random seed: {seed}")
    print("Loading dataset...")
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split(
        num_of_cells, seed
    )
    print("Train shape:", X_train.shape)
    print("Test  shape:", X_test.shape)

    analyze_trajectories(
        X_train,
        y_train,
        save_dir=f"audio_rc/figs",
        dt=0.0001,
    )

    # load_dataset_split の戻り値を受け取った直後あたりに追加
    print("zero feat mean:", X_train[y_train == 0].mean(axis=0)[:10])
    print("one  feat mean:", X_train[y_train == 1].mean(axis=0)[:10])
    print(
        "difference norm:",
        np.linalg.norm(
            X_train[y_train == 0].mean(axis=0) - X_train[y_train == 1].mean(axis=0)
        ),
    )

    print("Training readout...")
    W_out = train_readout(X_train, y_train, lambda_reg=1e-2)

    # 精度評価
    acc_train = evaluate(W_out, X_train, y_train)
    acc_test = evaluate(W_out, X_test, y_test)

    y_train_shuffled = np.random.permutation(y_train)
    W_out_shuffled = train_readout(X_train, y_train_shuffled)
    acc_test_shuffled = evaluate(W_out_shuffled, X_test, y_test)
    print(f"label shuffled test accuracy: {acc_test_shuffled}")

    # --- Confusion Matrix (2x2) ---
    num_classes = 2
    misclassified = []
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for feat, true_label, path in zip(X_test, y_test, X_test_paths):
        pred_label = predict(W_out, feat)
        conf[true_label, pred_label] += 1
        if true_label != pred_label:
            misclassified.append((path, true_label, pred_label))

    print("\nConfusion Matrix (rows=True, cols=Pred):")
    print(conf)
    print(f"\nTrue ZERO predicted as ZERO: {conf[0,0]} / {conf[0].sum()}")
    print(f"True ONE predicted as ONE:   {conf[1,1]} / {conf[1].sum()}\n")

    # --- Save confusion matrix as image ---

    plt.figure(figsize=(4, 4))
    plt.imshow(conf, cmap="Blues")
    if args.mode == "linear":
        plt.title(f"Confusion Matrix (Linear)")
    else:
        plt.title(f"Confusion Matrix, N = {num_of_cells}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # set axis ticks (only 0 and 1)
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

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
                fontsize=28,
                fontweight="bold",
            )

    plt.colorbar()
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(f"audio_rc/figs/confusion_matrix/{filename}")
    plt.close()
    print(f"Saved {filename}")

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # 保存
    np.save("audio_rc/reservoir_outputs/W_out.npy", W_out)
    print("Saved W_out.npy")

    print("\nMisclassified files:")
    if len(misclassified) == 0:
        print("  None! Perfect classification.")
    else:
        print(len(misclassified), "files misclassified.")
        # for path, true_label, pred_label in misclassified:
        #     print(f"  {path}  true={true_label}, pred={pred_label}")

    Path("audio_rc/results").mkdir(exist_ok=True)

    result = {
        "mode": args.mode,
        "seed": seed,
        "num_cells": num_of_cells,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

    # with open("audio_rc/results/results_LTS.jsonl", "a") as f:
    #     f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    num_of_cells = args.cells
    seed = args.seed
    main_train(num_of_cells, seed)
