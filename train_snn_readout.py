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
def load_dataset():
    X_list = []
    y_list = []

    if args.mode == "snn":
        # ZERO
        for path in tqdm(glob.glob("inputs/coch_zero/*.npy"), desc="ZERO"):
            coch = np.load(path)
            feat = main(
                input_data=coch, label="zero", return_feature=True, isDebugPrint=False
            )
            X_list.append(feat)
            y_list.append(0)

        # ONE
        for path in tqdm(glob.glob("inputs/coch_one/*.npy"), desc="ONE"):
            coch = np.load(path)
            feat = main(
                input_data=coch, label="one", return_feature=True, isDebugPrint=False
            )
            X_list.append(feat)
            y_list.append(1)

    elif args.mode == "feature":
        # ZERO features
        for path in tqdm(glob.glob("outputs/features_zero/*.npy"), desc="ZERO"):
            feat = np.load(path)
            X_list.append(feat)
            y_list.append(0)

        # ONE features
        for path in tqdm(glob.glob("outputs/features_one/*.npy"), desc="ONE"):
            feat = np.load(path)
            X_list.append(feat)
            y_list.append(1)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


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
    X, y = load_dataset()  # 特徴生成＋ラベル読み込み
    print("Shape of X:", X.shape)  # (num_samples, 100)

    # シャッフル
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # train/test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

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
