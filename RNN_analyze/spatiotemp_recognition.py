import json
import numpy as np
import glob
import argparse
from datetime import datetime
from collections import Counter
import gc

from tqdm import tqdm
import matplotlib.pyplot as plt

import os
from pathlib import Path

import src.PQN_RNN_onGPU as PQN_RNN_onGPU
import RNN_config

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
    default=240,
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
    sample_paths = glob.glob("RNN_analyze/reservoir_inputs/train/top/*.npy")
    if len(sample_paths) == 0:
        sample_paths = glob.glob("RNN_analyze/reservoir_inputs/train/top/*.npy")
    if len(sample_paths) == 0:
        raise RuntimeError("No cochleagram .npy files found in train directories.")
    sample_coch = np.load(sample_paths[0])
    input_size = sample_coch.shape[1]
    # print(sample_coch.shape)
    # If linear mode, we do not initialize or use the reservoir
    if args.mode != "linear":
        reservoir_state = RNN_config.init_reservoir(seed=seed, input_size=input_size)
    else:
        reservoir_state = None

    # ----- TRAIN -----
    if args.mode == "snn":
        # TOP
        path_list = glob.glob("RNN_analyze/reservoir_inputs/train/top/*.npy")
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TRAIN TOP")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
                record = is_last_loop,
            )
            X_train.append(feat)
            y_train.append(0)

        # MIDDLE
        path_list = glob.glob("RNN_analyze/reservoir_inputs/train/middle/*.npy")
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TRAIN MIDDLE")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
                record = is_last_loop,
            )
            X_train.append(feat)
            y_train.append(1)

        # BOTTOM
        path_list = glob.glob("RNN_analyze/reservoir_inputs/train/bottom/*.npy")
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TRAIN BOTTOM")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
                record = is_last_loop,
            )
            X_train.append(feat)
            y_train.append(2)
    elif args.mode == "feature":
        # TOP features
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_outputs/train/top/*.npy"),
            desc="TRAIN TOP",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(0)

        # MIDDLE features
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_outputs/train/middle/*.npy"),
            desc="TRAIN MIDDLE",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(1)

        # BOTTOM features
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_outputs/train/bottom/*.npy"),
            desc="TRAIN BOTTOM",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(2)
    elif args.mode == "linear":
        # TOP
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_inputs/train/top/*.npy"),
            desc="TRAIN TOP (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(0)

        # MIDDLE
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_inputs/train/middle/*.npy"),
            desc="TRAIN MIDDLE (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(1)

        # BOTTOM
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_inputs/train/bottom/*.npy"),
            desc="TRAIN BOTTOM (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(2)

    # ----- TEST -----
    if args.mode == "snn":
        # TOP
        path_list = glob.glob("RNN_analyze/reservoir_inputs/test/top/*.npy")
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TEST TOP")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
                record=is_last_loop,
            )
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)

        # MIDDLE
        path_list = glob.glob("RNN_analyze/reservoir_inputs/test/middle/*.npy")
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TEST MIDDLE")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
                record=is_last_loop,
            )
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)

        # BOTTOM
        path_list = glob.glob("RNN_analyze/reservoir_inputs/test/bottom/*.npy")
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TEST BOTTOM")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                N=num_of_cells,
                record=is_last_loop,
            )
            X_test.append(feat)
            y_test.append(2)
            X_test_paths.append(path)
    elif args.mode == "feature":
        # TOP features
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_outputs/test/top/*.npy"),
            desc="TEST TOP",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)
        # MIDDLE features
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_outputs/test/middle/*.npy"),
            desc="TEST MIDDLE",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)
        # BOTTOM features
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_outputs/test/bottom/*.npy"),
            desc="TEST BOTTOM",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(2)
            X_test_paths.append(path)
    elif args.mode == "linear":
        # TOP
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_inputs/test/top/*.npy"),
            desc="TEST TOP (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(0)
            X_test_paths.append(path)

        # MIDLE
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_inputs/test/middle/*.npy"),
            desc="TEST MIDLE (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(1)
            X_test_paths.append(path)

        # BOTTOM
        for path in tqdm(
            glob.glob("RNN_analyze/reservoir_inputs/test/bottom/*.npy"),
            desc="TEST BOTTOM (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(2)
            X_test_paths.append(path)

    # shuffle
    perm_train = np.random.permutation(len(X_train))

    X_train = [X_train[i] for i in perm_train]
    y_train = [y_train[i] for i in perm_train]

    perm_test = np.random.permutation(len(X_test))

    X_test = [X_test[i] for i in perm_test]
    y_test = [y_test[i] for i in perm_test]

    if len(X_train) == 0:
        raise RuntimeError("No training data loaded.")


    # make delayed answer
    X_train_array = np.vstack([x for x in X_train])
    X_test_array = np.vstack([x for x in X_test])
    del X_train
    del X_test
    gc.collect()

    return (
        X_train_array,
        np.array(y_train),
        X_test_array,
        np.array(y_test),
        X_test_paths,
    )

# --- Pad sequences to T_max and flatten ---
def delay_answer(y, T_max, delay):
    C = 3
    Y = np.zeros((T_max,C), dtype=np.float32)
    delay_steps = int(delay / 0.0001)  # assuming dt=0.0001
    Y[delay_steps:delay_steps+1000,y] = 0.8
    return Y

# ================================
# 2. 線形 readout の学習 (ridge regression)
# ================================
def train_readout(X: np.ndarray, y: np.ndarray, lambda_reg: float = 1e-2, batch_size: int = 5000) -> np.ndarray:
    """
    元のコードの機能を維持しつつ、メモリを節約するためにデータを小分けにして計算する関数
    """
    num_samples, D = X.shape
    
    # yの形状を見て、クラス分類（ラベル）か回帰（ターゲット）かを判断
    if y.ndim == 1:
        # 分類タスク（元のコードと同じ処理）
        classes = np.unique(y)
        C = len(classes)
        is_classification = True
    else:
        # 回帰タスク（タイマータスクなどですでにYができている場合への対応）
        C = y.shape[1]
        is_classification = False

    # 1. 巨大な K (NxN) は作らず、小さな XtX (DxD) と XtY (DxC) を累積する箱を用意
    #    float64 にすることで精度落ち（結果が違う現象）を防ぎます
    XtX = np.zeros((D, D), dtype=np.float64)
    XtY = np.zeros((D, C), dtype=np.float64)
    
    # 2. データをバッチサイズ（例: 5000行）ごとに区切って処理
    # print(f"Training readout in batches (N={num_samples}, Batch={batch_size})...")
    
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        
        # 必要な分だけメモリに載せる
        X_batch = X[i:end].astype(np.float64)
        y_batch_part = y[i:end]
        
        # ラベル(1D)なら、このバッチの中だけでOne-hot行列(2D)を作る
        # (巨大なY行列を一度に作らないのでメモリに優しい)
        if is_classification:
            Y_batch = np.zeros((len(y_batch_part), C), dtype=np.float64)
            for j, label in enumerate(y_batch_part):
                Y_batch[j, label] = 1.0
        else:
            Y_batch = y_batch_part.astype(np.float64)
        
        # 累積加算 (X^T X と X^T Y)
        XtX += X_batch.T @ X_batch
        XtY += X_batch.T @ Y_batch
        
        # 使い終わった変数を削除してメモリ掃除
        del X_batch, y_batch_part, Y_batch
        # gc.collect() # 動作が重すぎる場合はコメントアウトでも可

    # 3. 最後にまとめて計算 (D x D なので一瞬で終わります)
    #    数学的に元のコード (inv(K + lambda I) @ Y) と等価です
    I = np.eye(D, dtype=np.float64)
    
    # inv ではなく solve を使うことで、さらに精度と安定性を高めています
    W_out = np.linalg.solve(XtX + lambda_reg * I, XtY)
    
    return W_out.astype(np.float32)


# ================================
# 4. テストの精度測定
# ================================
def evaluate(W_out: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    Y = X @ W_out  # shape = (C,)
    # print(Y.shape)
    # print(y.shape)
    Y = Y.reshape(-1)
    y = y.reshape(-1)    
    # 【追加】分散が0（すべての値が同じ）なら、相関は計算できないので 0 を返す
    if np.std(Y) == 0 or np.std(y) == 0:
        return 0.0

    # サイズ不一致チェック（念のため）
    if Y.shape[0] != y.shape[0]:
        print(f"Error: Shape mismatch in evaluate. Pred: {Y.shape}, True: {y.shape}")
        return 0.0

    # 正常なら計算
    corr_matrix = np.corrcoef(Y, y)
    
    # NaNチェック（念のため）
    if np.isnan(corr_matrix[0, 1]):
        return 0.0
        
    r = corr_matrix[0, 1]
    r2 = r ** 2
    return r2


# ================================
# 5. メイン処理
# ================================
def main_train(num_of_cells: int, seed: int) -> None:
    print(f"Mode: {args.mode}")
    print(f"Number of reservoir cells: {num_of_cells}")
    print(f"Random seed: {seed}")
    print("Loading dataset...")
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split(num_of_cells, seed)
    print("Train shape:", X_train.shape)
    print("Test  shape:", X_test.shape)
    del X_test_paths

    num_trials = len(y_train)
    total_rows = X_train.shape[0]
    steps_per_trial = total_rows // num_trials
    T_max = int(steps_per_trial * 0.0001)

    print("Training readout...")
    print()
    duration = 0.01
    steps = int(T_max // duration)
    t = np.zeros(steps)
    r_train = np.zeros(steps)
    r_test = np.zeros(steps)
    for i in tqdm(np.arange(steps), desc="short term memory"):
        delay = duration * i
        Y_train_delayed = np.vstack([delay_answer(y, steps_per_trial, delay) for y in y_train])
        Y_test_delayed = np.vstack([delay_answer(y, steps_per_trial, delay) for y in y_test])
        W_out = train_readout(X_train, Y_train_delayed, lambda_reg=1e-2)
        # 精度評価
        t[i] = delay
        r_train[i] = evaluate(W_out, X_train, Y_train_delayed)
        r_test[i] = evaluate(W_out, X_test, Y_test_delayed)
        del Y_train_delayed
        del Y_test_delayed
        if i != steps - 1:
            del W_out
        gc.collect()

    # show graph
    plt.figure(figsize=(8, 6)) 
    plt.plot(t, r_train, label='Dataset 1', linestyle='-', color='blue')
    plt.plot(t, r_test, label='Dataset 2', linestyle='-', color='orange')
    plt.title("short term memory")
    plt.xlabel(" τ [s] ")
    plt.ylabel("r^2")
    plt.legend()
    plt.grid(True)

    folder = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%H%M")
    filename = "short-term-memory.png"
    os.makedirs(f"RNN_analyze/data/{folder}/{timestamp}", exist_ok=True)
    plt.savefig(f"RNN_analyze/data/{folder}/{timestamp}/{filename}")
    plt.close()
    print(f"Saved {filename}")
    data = np.column_stack([t, r_train, r_test])
    filename = "short-term-memory.npy"
    np.save(f"RNN_analyze/data/{folder}/{timestamp}/{filename}", data)


    # 保存
    np.save("RNN_analyze/reservoir_outputs/W_out.npy", W_out)
    print("Saved W_out.npy")


if __name__ == "__main__":
    num_of_cells = args.cells
    seed = args.seed
    main_train(num_of_cells, seed)
