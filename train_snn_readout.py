import numpy as np
import glob
import argparse

from tqdm import tqdm
from GPU_SNN_simulation import main  # あなたのSNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["snn", "feature"],
    default="feature",
    help="snn: run SNN to compute features, feature: load saved feature .npy",
)
args = parser.parse_args()


# ================================
# 1. データ読み込み関数
# ================================
def load_dataset_split():
    X_train, y_train = [], []
    X_test, y_test = [], []

    # ----- TRAIN -----
    if args.mode == "snn":
        # ZERO
        for path in tqdm(glob.glob("inputs/train/coch_zero/*.npy"), desc="TRAIN ZERO"):
            coch = np.load(path)
            feat = main(
                input_data=coch, label="zero", return_feature=True, isDebugPrint=False
            )
            X_train.append(feat)
            y_train.append(0)

        # ONE
        for path in tqdm(glob.glob("inputs/train/coch_one/*.npy"), desc="TRAIN ONE"):
            coch = np.load(path)
            feat = main(
                input_data=coch, label="one", return_feature=True, isDebugPrint=False
            )
            X_train.append(feat)
            y_train.append(1)
    elif args.mode == "feature":
        # ZERO features
        for path in tqdm(
            glob.glob("outputs/train/features_zero/*.npy"), desc="TRAIN ZERO"
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(0)
        # ONE features
        for path in tqdm(
            glob.glob("outputs/train/features_one/*.npy"), desc="TRAIN ONE"
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(1)

    # ----- TEST -----
    if args.mode == "snn":
        # ZERO
        for path in tqdm(glob.glob("inputs/test/coch_zero/*.npy"), desc="TEST ZERO"):
            coch = np.load(path)
            feat = main(
                input_data=coch, label="zero", return_feature=True, isDebugPrint=False
            )
            X_test.append(feat)
            y_test.append(0)

        # ONE
        for path in tqdm(glob.glob("inputs/test/coch_one/*.npy"), desc="TEST ONE"):
            coch = np.load(path)
            feat = main(
                input_data=coch, label="one", return_feature=True, isDebugPrint=False
            )
            X_test.append(feat)
            y_test.append(1)
    elif args.mode == "feature":
        # ZERO features
        for path in tqdm(
            glob.glob("outputs/test/features_zero/*.npy"), desc="TEST ZERO"
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(0)
        # ONE features
        for path in tqdm(glob.glob("outputs/test/features_one/*.npy"), desc="TEST ONE"):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(1)

    return (
        np.stack(X_train, axis=0),
        np.array(y_train),
        np.stack(X_test, axis=0),
        np.array(y_test),
    )


# ================================
# 2. 線形 readout の学習 (ridge regression)
# ================================
def train_readout(X, y, lambda_reg=1e-2):
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
def predict(W_out, feat):
    logits = feat @ W_out  # shape = (C,)
    return np.argmax(logits)


# ================================
# 4. テストの精度測定
# ================================
def evaluate(W_out, X, y):
    correct = 0
    for feat, label in zip(X, y):
        pred = predict(W_out, feat)
        if pred == label:
            correct += 1
    return correct / len(y)


# ================================
# 5. メイン処理
# ================================
def main_train():
    print(f"Mode: {args.mode}")
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset_split()
    print("Train shape:", X_train.shape)
    print("Test  shape:", X_test.shape)

    print("Training readout...")
    W_out = train_readout(X_train, y_train, lambda_reg=1e-2)

    # 精度評価
    acc_train = evaluate(W_out, X_train, y_train)
    acc_test = evaluate(W_out, X_test, y_test)

    # --- Confusion Matrix (2x2) ---
    num_classes = 2
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for feat, true_label in zip(X_test, y_test):
        pred_label = predict(W_out, feat)
        conf[true_label, pred_label] += 1

    print("\nConfusion Matrix (rows=True, cols=Pred):")
    print(conf)
    print(f"\nTrue ZERO predicted as ZERO: {conf[0,0]} / {conf[0].sum()}")
    print(f"True ONE predicted as ONE:   {conf[1,1]} / {conf[1].sum()}\n")

    # --- Save confusion matrix as image ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.imshow(conf, cmap="Blues")
    plt.title("Confusion Matrix")
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
                fontsize=24,
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig("graphs/confusion_matrix.png")
    plt.close()
    print("Saved confusion_matrix.png")

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # 保存
    np.save("W_out.npy", W_out)
    print("Saved W_out.npy")


if __name__ == "__main__":
    main_train()
