import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# パス設定（ご自身の環境に合わせてください）
BASE_DIR = "audio_rc/reservoir_inputs"
TRAIN_ZERO = os.path.join(BASE_DIR, "train/coch_zero/*.npy")
TRAIN_ONE  = os.path.join(BASE_DIR, "train/coch_one/*.npy")
TEST_ZERO  = os.path.join(BASE_DIR, "test/coch_zero/*.npy")
TEST_ONE   = os.path.join(BASE_DIR, "test/coch_one/*.npy")

def get_lengths(glob_pattern):
    """指定されたパターンのファイルの長さをリストで返す"""
    paths = glob.glob(glob_pattern)
    lengths = []
    for p in paths:
        # load時に mmap_mode='r' を使うと中身を全部読まないので高速です
        try:
            arr = np.load(p, mmap_mode='r')
            lengths.append(arr.shape[0]) # 時間方向の長さ
        except:
            pass
    return np.array(lengths)

print("Loading data lengths...")
len_train_0 = get_lengths(TRAIN_ZERO)
len_train_1 = get_lengths(TRAIN_ONE)
len_test_0  = get_lengths(TEST_ZERO)
len_test_1  = get_lengths(TEST_ONE)

print(f"Train Zero: {len(len_train_0)} files, Mean len: {len_train_0.mean():.1f}")
print(f"Train One : {len(len_train_1)} files, Mean len: {len_train_1.mean():.1f}")

# --- 1. ヒストグラムの描画 ---
plt.figure(figsize=(10, 6))
plt.hist(len_train_0, bins=30, alpha=0.5, label='Zero (Train)', color='blue')
plt.hist(len_train_1, bins=30, alpha=0.5, label='One (Train)', color='orange')
plt.title("Distribution of Data Lengths (Time Steps)")
plt.xlabel("Length (Time steps)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)
save_path = "length_distribution.png"
plt.savefig(save_path)
print(f"\nHistogram saved to {save_path}")

# --- 2. 「長さだけ」での分類精度シミュレーション ---
# 最適な閾値(Threshold)をTrainデータから探す
# 「長さ > 閾値 なら Zero」という単純ルールを作ってみる

def calculate_accuracy(threshold, len_0, len_1):
    # Zeroは閾値より長いと仮定 (逆なら符号反転)
    correct_0 = np.sum(len_0 > threshold)
    correct_1 = np.sum(len_1 <= threshold)
    return (correct_0 + correct_1) / (len(len_0) + len(len_1))

# 総当たりでベストな閾値を探す
all_lengths = np.concatenate([len_train_0, len_train_1])
best_acc = 0
best_thresh = 0

# 閾値の候補（データの最小値から最大値まで）
thresholds = np.unique(all_lengths)
thresholds = (thresholds[:-1] + thresholds[1:]) / 2 # 中間点を閾値にする

for th in thresholds:
    acc = calculate_accuracy(th, len_train_0, len_train_1)
    if acc > best_acc:
        best_acc = acc
        best_thresh = th

# 逆パターンの確認（Zeroの方が短い場合）
for th in thresholds:
    # Zero < Threshold
    correct_0 = np.sum(len_train_0 <= th)
    correct_1 = np.sum(len_train_1 > th)
    acc = (correct_0 + correct_1) / (len(len_train_0) + len(len_train_1))
    if acc > best_acc:
        best_acc = acc
        best_thresh = th
        mode = "Zero is SHORTER"
    else:
        mode = "Zero is LONGER"

print("\n" + "="*40)
print("【検証結果】長さだけの分類能力")
print("="*40)
print(f"学習データから求めた最適閾値: {best_thresh:.1f}")
print(f"判定ルール: Length {'>' if mode == 'Zero is LONGER' else '<='} {best_thresh:.1f} --> ZERO")
print(f"Train Accuracy (Length only): {best_acc*100:.2f}%")

# Testデータで検証
if mode == "Zero is LONGER":
    test_correct = np.sum(len_test_0 > best_thresh) + np.sum(len_test_1 <= best_thresh)
else:
    test_correct = np.sum(len_test_0 <= best_thresh) + np.sum(len_test_1 > best_thresh)
    
test_acc = test_correct / (len(len_test_0) + len(len_test_1))

print(f"Test Accuracy  (Length only): {test_acc*100:.2f}%")
print("="*40)

if test_acc > 0.9:
    print("★警告: 「長さ」だけでほぼ分類できてしまっています。")
    print("SNNは音声の特徴ではなく、単に信号の長さを学習している可能性が高いです。")
else:
    print("〇 良好: 「長さ」だけでは分類できません。")
    print("SNNはちゃんと音声の中身（特徴）を見ていると言えます。")