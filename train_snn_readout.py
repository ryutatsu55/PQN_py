import numpy as np
import glob
from GPU_SNN_simulation import main  # あなたのSNN


# ================================
# 1. データ読み込み関数
# ================================
def load_dataset():
    X_list = []
    y_list = []

    # ZERO
    for path in glob.glob("inputs/coch_zero/*.npy"):
        coch = np.load(path)
        feat = main(input_data=coch, label="zero", return_feature=True, isTQDM=False)
        X_list.append(feat)
        y_list.append(0)

    # ONE
    for path in glob.glob("inputs/coch_one/*.npy"):
        coch = np.load(path)
        feat = main(input_data=coch, label="one", return_feature=True, isTQDM=False)
        X_list.append(feat)
        y_list.append(1)

    X = np.stack(X_list, axis=0)  # shape = (num_samples, 100)
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

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # 保存
    np.save("W_out.npy", W_out)
    print("Saved W_out.npy")


if __name__ == "__main__":
    main_train()
