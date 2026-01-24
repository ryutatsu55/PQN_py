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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import lfilter

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
import config
import src.PQN_RNN_onGPU as PQN_RNN_onGPU

cfg = config.Config

# --- ディレクトリパス設定 ---
BASE_DIR = "RNN_analyze"
INPUT_DIR = os.path.join(BASE_DIR, "reservoir_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "reservoir_outputs")
RESULT_DIR = os.path.join(BASE_DIR, "result")

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


def main() -> None:
    print(f"Mode: {args.mode}")
    print(f"Number of reservoir cells: {cfg.N}")
    print(f"Random seed: {cfg.SEED}")
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
    mode: str, rng
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_test_paths = []

    # determine input dimension M (number of cochleagram channels)
    train_top_dir = os.path.join(INPUT_DIR, "train", "top", "*.npy")
    sample_paths = glob.glob(train_top_dir)
    if len(sample_paths) == 0:
        raise RuntimeError("No cochleagram .npy files found in train directories.")
    sample_data = np.load(sample_paths[0])
    input_size = sample_data.shape[1]
    # print(sample_data.shape)
    # If linear mode, we do not initialize or use the reservoir
    if mode != "linear":
        reservoir_state = config.init_reservoir()
        sim = PQN_RNN_onGPU.PQN_Reservoir_GPU(reservoir_state, cfg) 
    else:
        reservoir_state = None
    
    
    # ----- TRAIN -----
    if mode == "snn":
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        # TOP
        record={"result_dir": RESULT_DIR, "filename": "top"}
        output_dir = os.path.join(OUTPUT_DIR, "train/top")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(INPUT_DIR, "train", "top", "*.npy"))
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TRAIN TOP")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                record = record if is_last_loop else None,
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
                sim=sim,
            )
            X_train.append(feat)
            y_train.append(0)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # MIDDLE
        record={"result_dir": RESULT_DIR, "filename": "middle"}
        output_dir = os.path.join(OUTPUT_DIR, "train/middle")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(INPUT_DIR, "train", "middle", "*.npy"))
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TRAIN MIDDLE")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                record = record if is_last_loop else None,
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
                sim=sim,
            )
            X_train.append(feat)
            y_train.append(1)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # BOTTOM
        record={"result_dir": RESULT_DIR, "filename": "bottom"}
        output_dir = os.path.join(OUTPUT_DIR, "train/bottom")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(INPUT_DIR, "train", "bottom", "*.npy"))
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TRAIN BOTTOM")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                record = record if is_last_loop else None,
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
                sim=sim,
            )
            X_train.append(feat)
            y_train.append(2)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)
    elif mode == "feature":
        # TOP features
        for path in tqdm(
            glob.glob(os.path.join(OUTPUT_DIR, "train", "top", "*.npy")),
            desc="TRAIN TOP",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(0)

        # MIDDLE features
        for path in tqdm(
            glob.glob(os.path.join(OUTPUT_DIR, "train", "middle", "*.npy")),
            desc="TRAIN MIDDLE",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(1)

        # BOTTOM features
        for path in tqdm(
            glob.glob(os.path.join(OUTPUT_DIR, "train", "bottom", "*.npy")),
            desc="TRAIN BOTTOM",
        ):
            feat = np.load(path)
            X_train.append(feat)
            y_train.append(2)
    elif mode == "linear":
        # TOP
        for path in tqdm(
            glob.glob(os.path.join(INPUT_DIR, "train", "top", "*.npy")),
            desc="TRAIN TOP (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(0)

        # MIDDLE
        for path in tqdm(
            glob.glob(os.path.join(INPUT_DIR, "train", "middle", "*.npy")),
            desc="TRAIN MIDDLE (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(1)

        # BOTTOM
        for path in tqdm(
            glob.glob(os.path.join(INPUT_DIR, "train", "bottom", "*.npy")),
            desc="TRAIN BOTTOM (linear)",
        ):
            input = np.load(path)
            X_train.append(input)
            y_train.append(2)

    # ----- TEST -----
    if mode == "snn":
        # TOP
        record={"result_dir": RESULT_DIR, "filename": "top"}
        output_dir = os.path.join(OUTPUT_DIR, "test/top")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(INPUT_DIR, "test", "top", "*.npy"))
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TEST TOP")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                record = record if is_last_loop else None,
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
                sim=sim,
            )
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # MIDDLE
        record={"result_dir": RESULT_DIR, "filename": "middle"}
        output_dir = os.path.join(OUTPUT_DIR, "test/middle")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(INPUT_DIR, "test", "middle", "*.npy"))
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TEST MIDDLE")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                record = record if is_last_loop else None,
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
                sim=sim,
            )
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)
            
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, feat)

        # BOTTOM
        record={"result_dir": RESULT_DIR, "filename": "bottom"}
        output_dir = os.path.join(OUTPUT_DIR, "test/bottom")
        os.makedirs(output_dir, exist_ok=True)
        path_list = glob.glob(os.path.join(INPUT_DIR, "test", "bottom", "*.npy"))
        total_files = len(path_list)
        for i, path in enumerate(tqdm(path_list, desc="TEST BOTTOM")):
            is_last_loop = (i == total_files - 1)
            input = np.load(path)
            feat = PQN_RNN_onGPU.main(
                input_data=input,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                record = record if is_last_loop else None,
                S_durt=cfg.INPUT_DT,
                cfg=cfg,
                sim=sim,
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
            glob.glob(os.path.join(OUTPUT_DIR, "test", "top", "*.npy")),
            desc="TEST TOP",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(0)
            X_test_paths.append(path)
        # MIDDLE features
        for path in tqdm(
            glob.glob(os.path.join(OUTPUT_DIR, "test", "middle", "*.npy")),
            desc="TEST MIDDLE",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(1)
            X_test_paths.append(path)
        # BOTTOM features
        for path in tqdm(
            glob.glob(os.path.join(OUTPUT_DIR, "test", "bottom", "*.npy")),
            desc="TEST BOTTOM",
        ):
            feat = np.load(path)
            X_test.append(feat)
            y_test.append(2)
            X_test_paths.append(path)
    elif mode == "linear":
        # TOP
        for path in tqdm(
            glob.glob(os.path.join(INPUT_DIR, "test", "top", "*.npy")),
            desc="TEST TOP (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(0)
            X_test_paths.append(path)

        # MIDLE
        for path in tqdm(
            glob.glob(os.path.join(INPUT_DIR, "test", "middle", "*.npy")),
            desc="TEST MIDLE (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(1)
            X_test_paths.append(path)

        # BOTTOM
        for path in tqdm(
            glob.glob(os.path.join(INPUT_DIR, "test", "bottom", "*.npy")),
            desc="TEST BOTTOM (linear)",
        ):
            input = np.load(path)
            X_test.append(input)
            y_test.append(2)
            X_test_paths.append(path)

    # shuffle
    perm_train = rng.permutation(len(X_train))

    X_train = [X_train[i] for i in perm_train]
    y_train = [y_train[i] for i in perm_train]

    perm_test = rng.permutation(len(X_test))

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

def spatial_recognition(mode: str) -> None:
    rng = np.random.RandomState(cfg.SEED)
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split(mode, rng)
    # print("Applying calcium response filter...")
    # tau_calcium = 0.8 
    
    # X_train = [apply_calcium_filter(x, cfg.DT, tau=tau_calcium) for x in X_train]
    # X_test  = [apply_calcium_filter(x, cfg.DT, tau=tau_calcium) for x in X_test]

    analyze_trajectories(
        X_train, 
        y_train, 
        save_dir=f"{RESULT_DIR}", 
        dt=cfg.DT  # または cfg.DT (シミュレーションの時間刻みに合わせてください)
    )

    steps_per_trial = X_train[0].shape[0]
    X_train_flat = np.stack([pad_and_integrate(x, steps_per_trial) for x in X_train])
    X_test_flat = np.stack([pad_and_integrate(x, steps_per_trial) for x in X_test])
    del X_train
    del X_test
    gc.collect()

    print("\nTrain shape:", X_train_flat.shape)
    print("Test  shape:", X_test_flat.shape)

    # load_dataset_split の戻り値を受け取った直後あたりに追加
    print("top feat mean:", X_train_flat[y_train == 0].mean(axis=0)[:10])
    print("middle feat mean:", X_train_flat[y_train == 1].mean(axis=0)[40:50])
    print("bottom feat mean:", X_train_flat[y_train == 2].mean(axis=0)[80:90])
    print(
        "roughly calclated difference norm between top - middle:",
        np.linalg.norm(
            X_train_flat[y_train == 0].mean(axis=0) - X_train_flat[y_train == 1].mean(axis=0)
        ),
    )

    print("\nTraining readout...")
    W_out = train_readout(X_train_flat, y_train, lambda_reg=1e-2)

    # 精度評価
    acc_train = evaluate(W_out, X_train_flat, y_train)
    acc_test = evaluate(W_out, X_test_flat, y_test)
    y_train_shuffled = rng.permutation(y_train)
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
        plt.title(f"Confusion Matrix, N = {cfg.N}")
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
    plt.savefig(f"{RESULT_DIR}/figs/{filename}")
    plt.close()
    print(f"Saved {filename}")

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # 保存
    np.save(f"{OUTPUT_DIR}/W_out_space.npy", W_out)
    np.save(f"{RESULT_DIR}/data/W_out_space.npy", W_out)
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
        "seed": cfg.SEED,
        "num_cells": cfg.N,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

    with open(f"{BASE_DIR}/archive/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

def delayed_space(mode: str) -> None:
    rng = np.random.RandomState(cfg.SEED)
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test, X_test_paths = load_dataset_split(mode, rng)
    # print("Applying calcium response filter...")
    # tau_calcium = 0.8 
    
    # X_train = [apply_calcium_filter(x, cfg.DT, tau=tau_calcium) for x in X_train]
    # X_test  = [apply_calcium_filter(x, cfg.DT, tau=tau_calcium) for x in X_test]
    
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

    print("\nTraining readout...")
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
    plt.savefig(f"{RESULT_DIR}/figs/{filename}.png")
    plt.close()
    print(f"Saved {filename}")
    data = np.column_stack([t, r_train, r_test])
    np.save(f"{RESULT_DIR}/data/{filename}.npy", data)

def apply_calcium_filter(neural_data: np.ndarray, dt: float, tau: float = 0.8) -> np.ndarray:
    """
    ニューロン活動にカルシウム蛍光の減衰ダイナミクスを適用する
    
    Args:
        neural_data (np.ndarray): 形状 (Time, Neurons) の時系列データ
        dt (float): サンプリング間隔 [s] (例: cfg.INPUT_DT)
        tau (float): カルシウム減衰時定数 [s] (論文再現なら 0.6 ~ 1.0 程度)
        
    Returns:
        np.ndarray: フィルタ適用後のデータ
    """
    # 減衰係数の計算 ( alpha = exp(-dt/tau) )
    alpha = np.exp(-dt / tau)
    
    # フィルタ係数の設定
    # 数式: y[t] = alpha * y[t-1] + x[t]
    # (入力 x があると急上昇し、ない間は alpha の倍率で減衰していく)
    b = [1.0]           # 入力側の係数
    a = [1.0, -alpha]   # 出力(自己回帰)側の係数
    
    # フィルタ適用 (axis=0 は時間方向)
    filtered_data = lfilter(b, a, neural_data, axis=0)
    
    return filtered_data

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
    
    save_path_pca = os.path.join(save_dir, "figs", "pca_trajectories.png")
    plt.savefig(save_path_pca)
    plt.close()
    print(f"Saved PCA plot to {save_path_pca}")
    
    save_data_dir = os.path.join(save_dir, "data")        
    os.makedirs(save_data_dir, exist_ok=True)
    
    # データを保存用に成形 (2次元配列化)
    # X_pca shape: (n_trials, time_steps, 3)
    # 1. Trial ID [0, 0, ..., 1, 1, ...]
    trial_ids = np.repeat(np.arange(n_trials), time_steps).reshape(-1, 1)
    # 2. Time [0, dt, 2dt, ..., 0, dt, ...]
    times = np.tile(np.arange(time_steps) * dt, n_trials).reshape(-1, 1)
    # 3. PC1, PC2, PC3 (フラット化)
    pcs = X_pca.reshape(-1, 3)
    # 4. Label [0, 0, ..., 1, 1, ...]
    labels_expanded = np.repeat(y_arr, time_steps).reshape(-1, 1)
    # 全て結合 (N*T, 6)
    pca_data_to_save = np.hstack((trial_ids, times, pcs, labels_expanded))
    filename = f"pca_data.npy"
    
    save_path_data = os.path.join(save_data_dir, filename)
    np.save(save_path_data, pca_data_to_save)
    print(f"Saved PCA data to {save_path_data} (Shape: {pca_data_to_save.shape})")

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
    
    save_path_dist = os.path.join(save_dir, "figs", "distance_analysis.png")
    plt.savefig(save_path_dist)
    plt.close()
    print(f"Saved Distance plot to {save_path_dist}")

    # --- 距離解析データの保存 ---
    # データを結合して 2次元配列 (TimeSteps, 5) を作成
    # Col 0: Time [s]
    # Col 1: Mean (Different Class)
    # Col 2: Std  (Different Class)
    # Col 3: Mean (Same Class)
    # Col 4: Std  (Same Class)
    dist_data_to_save = np.column_stack((t_axis, mean_diff, std_diff, mean_same, std_same))
    filename = f"distance_analysis.npy"
    save_path_npy = os.path.join(save_data_dir, filename)
    np.save(save_path_npy, dist_data_to_save)
    print(f"Saved distance analysis data to {save_path_npy} (Shape: {dist_data_to_save.shape})")

# --- Pad sequences to T_max and flatten ---
def pad_and_integrate(x, steps_per_trial):
    # print(x.shape)
    # T, M = x.shape
    # if T < steps_per_trial:
    #     pad = np.zeros((steps_per_trial - T, M), dtype=np.float32)
    #     x = np.vstack([x, pad])
    return x.sum(axis=0)


def delay_answer(y, T_max, delay, mode):
    C = 3
    Y = np.zeros((T_max,C), dtype=np.float32)
    if mode == "linear":
        delay_steps = int(delay / cfg.INPUT_DT)
        Y[delay_steps:delay_steps+int(0.1//cfg.INPUT_DT),y] = cfg.INPUT_STRENGTH
    else:
        delay_steps = int(delay / cfg.DT)
        Y[delay_steps:delay_steps+int(0.1//cfg.DT),y] = cfg.INPUT_STRENGTH
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


if __name__ == "__main__":
    # RNN_config.set_global_seed(cfg.SEED)
    main()
