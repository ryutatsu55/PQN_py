import numpy as np
import glob
import argparse
from datetime import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)  # HACK: 親ディレクトリをパスに追加
import GPU_SNN_simulation

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["snn", "feature"],
    default="feature",
    help="snn: run SNN to compute features, feature: load saved feature .npy",
)
parser.add_argument(
    "--cells",
    "-c",
    type=int,
    default=100,
    help="number of reservoir cells (default: 100)",
)
args = parser.parse_args()


# ================================
# 1. データ読み込み関数
# ================================
def load_dataset_split(
    num_of_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_test_paths = []

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
                label="zero",
                return_feature=True,
                isDebugPrint=False,
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
                label="one",
                return_feature=True,
                isDebugPrint=False,
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
                label="zero",
                return_feature=True,
                isDebugPrint=False,
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
                label="one",
                return_feature=True,
                isDebugPrint=False,
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
        # ONE features
        for path in tqdm(
            glob.glob("audio_rc/reservoir_outputs/test/features_one/*.npy"),
            desc="TEST ONE",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(1)

    # shuffle
    perm_train = np.random.permutation(len(X_train))

    X_train = [X_train[i] for i in perm_train]
    y_train = [y_train[i] for i in perm_train]

    perm_test = np.random.permutation(len(X_test))

    X_test = [X_test[i] for i in perm_test]
    y_test = [y_test[i] for i in perm_test]

    return (
        np.stack(X_train, axis=0),
        np.array(y_train),
        np.stack(X_test, axis=0),
        np.array(y_test),
        X_test_paths,
    )


# ================================
# 2. 線形 readout の学習 (ridge regression)
# ================================
def train_readout(X: np.ndarray, y: np.ndarray, lambda_reg: float = 1e-2) -> np.ndarray:
    num_samples, N = X.shape
    classes = np.unique(y)
    C = len(classes)

    # one-hot 行列 Y (num_samples, C)
    Y = np.zeros((num_samples, C), dtype=np.float32)
    for i, label in enumerate(y):
        Y[i, label] = 1.0

    I = np.eye(N, dtype=np.float32)

    # ridge regression closed form解
    W_out = np.linalg.inv(X.T @ X + lambda_reg * I) @ (X.T @ Y)
    # shape = (100, C)

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
def main_train(num_of_cells: int) -> None:
    print(f"Mode: {args.mode}")
    print("Loading dataset...")
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split(num_of_cells)
    print("Train shape:", X_train.shape)
    print("Test  shape:", X_test.shape)

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
        for path, true_label, pred_label in misclassified:
            print(f"  {path}  true={true_label}, pred={pred_label}")


if __name__ == "__main__":
    num_of_cells = args.cells
    main_train(num_of_cells)
