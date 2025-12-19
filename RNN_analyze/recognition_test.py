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
import shutil

import src.PQN_RNN_onGPU as PQN_RNN_onGPU
import RNN_config

cfg = RNN_config.Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--classifier",
    choices=["space", "delayed_space", "both"],
    default="both",
    help="space: classify with space, delayed_space: classify with delayed space",
)
parser.add_argument(
    "--mode",
    choices=["snn", "feature", "linear"],
    default="feature",
    help="snn: run SNN to compute features, feature: load saved feature .npy, linear: use cochleagram directly",
)
args = parser.parse_args()


def main(num_of_cells: int = cfg.N, seed: int = cfg.SEED) -> None:
    print(f"Mode: {args.mode}")
    print(f"Number of reservoir cells: {num_of_cells}")
    print(f"Random seed: {seed}")
    print("Loading dataset...")
    if args.classifier == "space":
        spatial_recognition(mode = args.mode)
    elif args.classifier == "delayed_space":
        delayed_space(mode = args.mode)
    elif args.classifier == "both":
        spatial_recognition(mode = args.mode)
        delayed_space(mode = args.mode if args.mode == "linear" else "feature")

# ================================
# 1. データ読み込み関数
# ================================
def load_dataset_split(
    mode: str, num_of_cells: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_test_paths = []

    # determine input dimension M (number of cochleagram channels)
    train_top_dir = os.path.join(cfg.INPUT_DIR, "train", "top", "*.npy")
    sample_paths = glob.glob(train_top_dir)
    if len(sample_paths) == 0:
        raise RuntimeError("No cochleagram .npy files found in train directories.")
    sample_data = np.load(sample_paths[0])
    input_size = sample_data.shape[1]
    # print(sample_data.shape)
    # If linear mode, we do not initialize or use the reservoir
    if mode != "linear":
        reservoir_state = RNN_config.init_reservoir()
    else:
        reservoir_state = None

    # ----- TRAIN -----
    if mode == "snn":
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
        # TOP
        output_dir = os.path.join(cfg.OUTPUT_DIR, "train/top")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(cfg.INPUT_DIR, "train", "top", "*.npy"))
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
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
            )
            X_train.append(feat)
            y_train.append(0)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # MIDDLE
        output_dir = os.path.join(cfg.OUTPUT_DIR, "train/middle")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(cfg.INPUT_DIR, "train", "middle", "*.npy"))
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
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
            )
            X_train.append(feat)
            y_train.append(1)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # BOTTOM
        output_dir = os.path.join(cfg.OUTPUT_DIR, "train/bottom")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(cfg.INPUT_DIR, "train", "bottom", "*.npy"))
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
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
            )
            X_train.append(feat)
            y_train.append(2)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)
    elif mode == "feature":
        # TOP features
        for path in tqdm(
            glob.glob(os.path.join(cfg.OUTPUT_DIR, "train", "top", "*.npy")),
            desc="TRAIN TOP",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(0)

        # MIDDLE features
        for path in tqdm(
            glob.glob(os.path.join(cfg.OUTPUT_DIR, "train", "middle", "*.npy")),
            desc="TRAIN MIDDLE",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(1)

        # BOTTOM features
        for path in tqdm(
            glob.glob(os.path.join(cfg.OUTPUT_DIR, "train", "bottom", "*.npy")),
            desc="TRAIN BOTTOM",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(2)
    elif mode == "linear":
        # TOP
        for path in tqdm(
            glob.glob(os.path.join(cfg.INPUT_DIR, "train", "top", "*.npy")),
            desc="TRAIN TOP (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(0)

        # MIDDLE
        for path in tqdm(
            glob.glob(os.path.join(cfg.INPUT_DIR, "train", "middle", "*.npy")),
            desc="TRAIN MIDDLE (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(1)

        # BOTTOM
        for path in tqdm(
            glob.glob(os.path.join(cfg.INPUT_DIR, "train", "bottom", "*.npy")),
            desc="TRAIN BOTTOM (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(2)

    # ----- TEST -----
    if mode == "snn":
        # TOP
        output_dir = os.path.join(cfg.OUTPUT_DIR, "test/top")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(cfg.INPUT_DIR, "test", "top", "*.npy"))
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
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
            )
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # MIDDLE
        output_dir = os.path.join(cfg.OUTPUT_DIR, "test/middle")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(cfg.INPUT_DIR, "test", "middle", "*.npy"))
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
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
            )
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # BOTTOM
        output_dir = os.path.join(cfg.OUTPUT_DIR, "test/bottom")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(cfg.INPUT_DIR, "test", "bottom", "*.npy"))
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
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
            )
            X_test.append(feat)
            y_test.append(2)
            X_test_paths.append(path)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)
    elif mode == "feature":
        # TOP features
        for path in tqdm(
            glob.glob(os.path.join(cfg.OUTPUT_DIR, "test", "top", "*.npy")),
            desc="TEST TOP",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)
        # MIDDLE features
        for path in tqdm(
            glob.glob(os.path.join(cfg.OUTPUT_DIR, "test", "middle", "*.npy")),
            desc="TEST MIDDLE",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)
        # BOTTOM features
        for path in tqdm(
            glob.glob(os.path.join(cfg.OUTPUT_DIR, "test", "bottom", "*.npy")),
            desc="TEST BOTTOM",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(2)
            X_test_paths.append(path)
    elif mode == "linear":
        # TOP
        for path in tqdm(
            glob.glob(os.path.join(cfg.INPUT_DIR, "test", "top", "*.npy")),
            desc="TEST TOP (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(0)
            X_test_paths.append(path)

        # MIDLE
        for path in tqdm(
            glob.glob(os.path.join(cfg.INPUT_DIR, "test", "middle", "*.npy")),
            desc="TEST MIDLE (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(1)
            X_test_paths.append(path)

        # BOTTOM
        for path in tqdm(
            glob.glob(os.path.join(cfg.INPUT_DIR, "test", "bottom", "*.npy")),
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

    return (
        X_train,
        np.array(y_train),
        X_test,
        np.array(y_test),
        X_test_paths,
    )



# --- Pad sequences to T_max and flatten ---
def pad_and_integrate(x, steps_per_trial):
    T, M = x.shape
    if T < steps_per_trial:
        pad = np.zeros((steps_per_trial - T, M), dtype=np.float32)
        x = np.vstack([x, pad])
    return x.mean(axis=0)


def delay_answer(y, T_max, delay, mode):
    C = 3
    Y = np.zeros((T_max,C), dtype=np.float32)
    if mode == "linear":
        delay_steps = int(delay / cfg.INPUT_DT)
        Y[delay_steps:delay_steps+int(0.1//cfg.INPUT_DT),y] = 0.8
    else:
        delay_steps = int(delay / cfg.DT)
        Y[delay_steps:delay_steps+int(0.1//cfg.DT),y] = 0.8
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

def predict(W_out: np.ndarray, feat: np.ndarray) -> np.intp:
    logits = feat @ W_out  # shape = (C,)
    return np.argmax(logits)

def evaluate(W_out: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    for feat, label in zip(X, y):
        pred = predict(W_out, feat)
        if pred == label:
            correct += 1
    return correct / len(y)

def calc_r2(W_out: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
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
        print(f"Error: Shape mismatch in calc_r2. Pred: {Y.shape}, True: {y.shape}")
        return 0.0

    # 正常なら計算
    corr_matrix = np.corrcoef(Y, y)
    
    # NaNチェック（念のため）
    if np.isnan(corr_matrix[0, 1]):
        return 0.0
        
    r = corr_matrix[0, 1]
    r2 = r ** 2
    return r2

def spatial_recognition(mode: str, num_of_cells: int = cfg.N, seed: int = cfg.SEED) -> None:
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split(mode, num_of_cells, seed)
    steps_per_trial = X_train[0].shape[0]
    X_train_flat = np.stack([pad_and_integrate(x, steps_per_trial) for x in X_train])
    X_test_flat = np.stack([pad_and_integrate(x, steps_per_trial) for x in X_test])
    del X_train
    del X_test
    gc.collect()

    print("Train shape:", X_train_flat.shape)
    print("Test  shape:", X_test_flat.shape)

    # load_dataset_split の戻り値を受け取った直後あたりに追加
    print("top feat mean:", X_train_flat[y_train == 0].mean(axis=0)[:10])
    print("middle feat mean:", X_train_flat[y_train == 1].mean(axis=0)[40:50])
    print("bottom feat mean:", X_train_flat[y_train == 2].mean(axis=0)[80:90])
    print(
        "difference norm:",
        np.linalg.norm(
            X_train_flat[y_train == 0].mean(axis=0) - X_train_flat[y_train == 1].mean(axis=0)
        ),
    )

    print("Training readout...")
    W_out = train_readout(X_train_flat, y_train, lambda_reg=1e-2)

    # 精度評価
    acc_train = evaluate(W_out, X_train_flat, y_train)
    acc_test = evaluate(W_out, X_test_flat, y_test)
    y_train_shuffled = np.random.permutation(y_train)
    W_out_shuffled = train_readout(X_train_flat, y_train_shuffled)
    acc_test_shuffled = evaluate(W_out_shuffled, X_test_flat, y_test)
    print(f"label shuffled test accuracy: {acc_test_shuffled}")

    # --- Confusion Matrix ---
    classes = np.unique(y_train)
    num_classes = len(classes)
    misclassified = []
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for feat, true_label, path in zip(X_test_flat, y_test, X_test_paths):
        pred_label = predict(W_out, feat)
        conf[true_label, pred_label] += 1
        if true_label != pred_label:
            misclassified.append((path, true_label, pred_label))

    print("\nConfusion Matrix (rows=True, cols=Pred):")
    print(conf)
    print(f"\nTrue TOP predicted as TOP: {conf[0,0]} / {conf[0].sum()}")
    print(f"True MIDDLE predicted as MIDDLE:   {conf[1,1]} / {conf[1].sum()}")
    print(f"True BOTTOM predicted as BOTTOM:   {conf[2,2]} / {conf[2].sum()}\n")

    # --- Save confusion matrix as image ---

    plt.figure(figsize=(4, 4))
    plt.imshow(conf, cmap="Blues")
    if args.mode == "linear":
        plt.title(f"Confusion Matrix (Linear)")
    else:
        plt.title(f"Confusion Matrix, N = {num_of_cells}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # set axis ticks
    # plt.xticks([0, 1], ["0", "1"])
    # plt.yticks([0, 1], ["0", "1"])
    plt.xticks([0, 1, 2], ["Top", "Middle", "Bottom"])
    plt.yticks([0, 1, 2], ["Top", "Middle", "Bottom"])

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

    # ================================================
    # Save results
    # ================================================
    filename = "confusion_matrix_.png"
    plt.savefig(f"{cfg.RESULT_DIR}/figs/{filename}")
    plt.close()
    print(f"Saved {filename}")

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # 保存
    np.save(f"{cfg.OUTPUT_DIR}/W_out_space.npy", W_out)
    np.save(f"{cfg.RESULT_DIR}/data/W_out_space.npy", W_out)
    print("Saved W_out.npy")

    print("\nMisclassified files:")
    if len(misclassified) == 0:
        print("  None! Perfect classification.")
    else:
        print(len(misclassified), "files misclassified.")
        # for path, true_label, pred_label in misclassified:
        #     print(f"  {path}  true={true_label}, pred={pred_label}")

    result = {
        "mode": args.mode,
        "seed": seed,
        "num_cells": num_of_cells,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

    with open(f"{cfg.BASE_DIR}/archive/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

def delayed_space(mode: str, num_of_cells: int = cfg.N, seed: int = cfg.SEED) -> None:
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split(mode, num_of_cells, seed)
    
    steps_per_trial = X_train[0].shape[0]
    if mode == "linear":
        T_max = steps_per_trial * cfg.INPUT_DT
    else:
        T_max = steps_per_trial * cfg.DT

    X_train_array = np.vstack([x for x in X_train])
    X_test_array = np.vstack([x for x in X_test])
    del X_train
    del X_test
    del X_test_paths
    gc.collect()

    print("Train shape:", X_train_array.shape)
    print("Test  shape:", X_test_array.shape)

    print("Training readout...")
    duration = cfg.SPATIO_TEMP_DT
    steps = int(T_max // duration)
    t = np.zeros(steps)
    r_train = np.zeros(steps)
    r_test = np.zeros(steps)
    for i in tqdm(np.arange(steps), desc="short term memory"):
        delay = duration * i
        Y_train_delayed = np.vstack([delay_answer(y, steps_per_trial, delay, mode) for y in y_train])
        Y_test_delayed = np.vstack([delay_answer(y, steps_per_trial, delay, mode) for y in y_test])
        W_out = train_readout(X_train_array, Y_train_delayed, lambda_reg=1e-2)
        # 精度評価
        t[i] = delay
        r_train[i] = calc_r2(W_out, X_train_array, Y_train_delayed)
        r_test[i] = calc_r2(W_out, X_test_array, Y_test_delayed)
        del Y_train_delayed
        del Y_test_delayed
        if i != steps - 1:
            del W_out
        gc.collect()

    # show graph
    plt.figure(figsize=(8, 6)) 
    plt.plot(t, r_train, label='train', linestyle='-', color='blue')
    plt.plot(t, r_test, label='test', linestyle='-', color='orange')
    plt.title("short term memory")
    plt.xlabel(" τ [s] ")
    plt.ylabel("r^2")
    plt.legend()
    plt.grid(True)


    # ================================================
    # Save results
    # ================================================
    filename = "short-term-memory"
    plt.savefig(f"{cfg.RESULT_DIR}/figs/{filename}.png")
    plt.close()
    print(f"Saved {filename}")
    data = np.column_stack([t, r_train, r_test])
    np.save(f"{cfg.RESULT_DIR}/data/{filename}.npy", data)


    # 保存
    filename = "W_out_spatiotemp"
    np.save(f"{cfg.RESULT_DIR}/data/{filename}.npy", W_out)
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
