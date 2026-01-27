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
from scipy.optimize import curve_fit

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
    default="linear",
    help="snn: run SNN to compute features, feature: load saved feature .npy, linear: use cochleagram directly",
)
args = parser.parse_args()


def main() -> None:
    print(f"Mode: {args.mode}")
    print(f"Number of reservoir cells: {cfg.N}")
    print(f"Random seed: {cfg.SEED}")
    print()

    reservoir_state = None
    sim = None
    if args.mode == "snn":
        print("Initializing Reservoir...")
        reservoir_state = config.init_reservoir()
        sim = PQN_RNN_onGPU.PQN_Reservoir_GPU(reservoir_state, cfg)

    if args.mode == "snn" and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"{OUTPUT_DIR} is removed for overwrite.")

    print()
    if args.classifier == "space":
        spatial_recognition(sim, reservoir_state)
    elif args.classifier == "delayed_space":
        delayed_space(sim, reservoir_state)
    elif args.classifier == "both":
        spatial_recognition(sim, reservoir_state)
        delayed_space(sim, reservoir_state)

def create_target_signal_classification(n_steps, label, num_classes, dt):
    """
    論文準拠のターゲット信号を作成する
    - 形状: (n_steps, num_classes)
    - 刺激開始から 2.5秒間 は 1.0、それ以外は 0.0
    """
    target = np.zeros((n_steps, num_classes), dtype=np.float32)
    
    # 2.5秒分のステップ数
    duration_sec = cfg.TEACHING_DURATION
    active_steps = int(duration_sec / dt)
    
    # 配列の長さを超えないようにクリップ
    end_idx = min(active_steps, n_steps)
    
    # 正解クラスの列を 1.0 に設定
    target[:end_idx, label] = 1.0
    
    # print(f"{end_idx*dt} [s]")
    # print(f"{n_steps*dt} [s]")
    
    return target

def create_target_signal_timer(n_steps, dt, delay_sec, input_duration_sec=0.1):
    """
    タイマータスク用ターゲット信号
    入力パルス(0.1s)を delay_sec だけ遅らせた波形を作成
    Shape: (n_steps, 1)
    """
    target = np.zeros((n_steps, 1), dtype=np.float32)
    start_step = int(delay_sec / dt)
    # パルス幅ステップ
    width_step = int(input_duration_sec / dt)
    
    end_step = min(start_step + width_step, n_steps)
    
    if start_step < n_steps:
        target[start_step:end_step, 0] = 1.0
        
    return target

def collect_file_metadata():
    base_dirs = {
        "train": f"{INPUT_DIR}/train",
        "test": f"{INPUT_DIR}/test"
    }
    # カテゴリ定義 (ディレクトリ名 -> 属性)
    categories = [
        {"dir": "top",      "area": "top",    "label": 0},
        {"dir": "middle",   "area": "middle", "label": 1},
        {"dir": "bottom",   "area": "bottom", "label": 2},
    ]

    metadata_list = []

    for split, base_dir in base_dirs.items():
        for cat in categories:
            subdir = cat["dir"]
            # 入力ファイル (.npy) を探す
            search_path = os.path.join(base_dir, subdir, "*.npy")
            files = sorted(glob.glob(search_path))
            
            for path in files:
                metadata_list.append({
                    "input_path": path,
                    "area": cat["area"],
                    "label": cat["label"],
                    "split": split, # original split ('train' or 'test')
                    "subdir_name": subdir # 保存先ディレクトリ作成用
                })
    return metadata_list

def get_feature_save_path(input_path, original_split, subdir_name):
    """
    入力パスに対応する特徴量の保存先パスを生成する。
    構造: {BASE_DIR}/reservoir_outputs/{train|test}/{subdir}/{filename}
    """
    filename = os.path.basename(input_path)
    
    save_dir = os.path.join(OUTPUT_DIR, original_split, subdir_name)
    save_path = os.path.join(save_dir, filename)
    return save_path, save_dir

def load_and_process_data(items, sim, reservoir_state, dt, recorded_areas, dsec="Processing data"):
    """
    Returns:
        X_flat: (N, Neurons) - Integrated features for readout
        y: (N,) - Labels
        paths: List[str] - File paths
        X_time: List[np.ndarray] - Raw time-series features for analysis
    """
    X_list = []
    Y_list = [] # ターゲット信号リスト
    y_labels = []

    num_classes = 3

    for item in tqdm(items, desc=dsec):
        input_path = item["input_path"]         # full path to .npy
        label = item["label"]                   # 0, 1, 2
        original_split = item["split"]          # train/test
        subdir_name = item["subdir_name"]       # top/middle/bottom
        area = item["area"]                     # area name

        # 特徴量の保存先パスを決定
        save_path, save_dir = get_feature_save_path(input_path, original_split, subdir_name)
        feat = None
        
        # n_steps: entire steps for each trial
        if original_split == "train":
            tmax = cfg.DURATION_INTERVAL
        else:
            tmax = cfg.TEACHING_DURATION

        n_steps = int(tmax / dt)
        
        if args.mode == "linear":
            feat = np.load(input_path)
            current_len = feat.shape[0]
            if current_len < n_steps:
                padding = np.zeros((n_steps - current_len, feat.shape[1]))
                feat = np.vstack([feat, padding])
            else:
                feat = feat[:n_steps]
        elif args.mode == "feature":
            if os.path.exists(save_path):
                feat = np.load(save_path)
            else:
                print(f"Warning: Feature file not found: {save_path}")
                continue
        elif args.mode == "snn":
            # キャッシュチェック
            if os.path.exists(save_path):   #overwrite の場合はキャッシュをすでに消去していると仮定
                # キャッシュがあればロード
                feat = np.load(save_path)
            else:
                os.makedirs(save_dir, exist_ok=True)
                
                input_data = np.load(input_path)
                if area not in recorded_areas:
                    # まだ記録していないエリアなら記録用辞書を作成
                    current_record = {
                        "result_dir": RESULT_DIR,
                        "filename": area,
                    }
                    recorded_areas.add(area)
                else:
                    current_record = None
                feat = PQN_RNN_onGPU.main(
                    input_data=input_data,
                    coch=False,
                    reservoir_state=reservoir_state,
                    return_feature=True,
                    is_debug_print=False,
                    # record=True if i+1 == len(items) else False,
                    record=current_record,
                    tmax=tmax,
                    S_durt=cfg.INPUT_DT,
                    cfg=cfg,
                    sim=sim
                )
                # 保存（生の時系列特徴量を保存しておく）
                np.save(save_path, feat)
            
        if feat is not None:
            # 切り詰め/パディング (念のため)
            if feat.shape[0] > n_steps: 
                feat = feat[:n_steps]
            
            X_list.append(feat)
            y_labels.append(label)
            target = create_target_signal_classification(n_steps, label, num_classes, dt)
            Y_list.append(target)

    # 結合 (学習用)
    if len(X_list) > 0:
        X_concat = np.vstack(X_list)
        Y_concat = np.vstack(Y_list)
    else:
        X_concat, Y_concat = np.array([]), np.array([])
        
    return X_concat, Y_concat, X_list, np.array(y_labels)

def spatial_recognition(sim, reservoir_state) -> None:
    rng = np.random.RandomState(cfg.SEED)
    dt = cfg.INPUT_DT if args.mode == "linear" else cfg.DT
    print("======== Spatial Recognition Task ========")

    print("\nLoading dataset...")

    all_meta = collect_file_metadata()
    train_meta = [m for m in all_meta if m["split"] == "train"]
    test_meta = [m for m in all_meta if m["split"] == "test"]

    # Shuffle & Limit
    rng.shuffle(train_meta)
    rng.shuffle(test_meta)
    train_meta = train_meta[:getattr(cfg, "N_TRAIN", len(train_meta))]
    test_meta = test_meta[:getattr(cfg, "N_TEST", len(test_meta))]

    print(f"selected Train files: {len(train_meta)}")
    print(f"selected Test files:  {len(test_meta)}")

    # process Data
    print()
    print("Processing data...")
    recorded_areas = set()
    X_train_all, Y_train_all, X_train_list, y_train_labels = load_and_process_data(
        train_meta, sim, reservoir_state, dt, recorded_areas, dsec="TRAIN"
        )
    X_test_all, Y_test_all, X_test_list, y_test_labels = load_and_process_data(
        test_meta, sim, reservoir_state, dt, recorded_areas, dsec="TEST"
        )

    # print(X_train_all.shape)
    # print(X_train_list[0].shape)
    # print(Y_train_all.shape)
    # print(len(y_train_labels))

    if len(Y_train_all) == 0:
        print("Error: No training data.")
        return

    # Train
    print("Training Readout...")
    W_out = train_readout(X_train_all, Y_train_all, lambda_reg=1.0)

    # Test
    print("Testing...")
    train_correct = 0
    test_correct = 0
    conf_matrix = np.zeros((3, 3), dtype=int)

    for i, feat in enumerate(X_train_list):
        pred = predict(W_out, feat, dt)
        true = y_train_labels[i]
        if pred == true:
            train_correct += 1
    for i, feat in enumerate(X_test_list):
        pred = predict(W_out, feat, dt)
        true = y_test_labels[i]
        conf_matrix[true, pred] += 1
        if pred == true:
            test_correct += 1

    train_acc = train_correct / len(y_train_labels)
    test_acc = test_correct / len(y_test_labels)
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(conf_matrix)
    print(f"\nTrue TOP predicted as TOP: {conf_matrix[0,0]} / {conf_matrix[0].sum()}")
    print(f"True MIDDLE predicted as MIDDLE:   {conf_matrix[1,1]} / {conf_matrix[1].sum()}")
    print(f"True BOTTOM predicted as BOTTOM:   {conf_matrix[2,2]} / {conf_matrix[2].sum()}\n")

    class_names = ["Top", "Middle", "Bottom"]
    num_classes = len(class_names)
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title(f"Confusion Matrix (spatial recognition)\nAcc: {test_acc*100:.1f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(num_classes), class_names)
    plt.yticks(np.arange(num_classes), class_names)
    
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black", fontsize=16)
            
    plt.colorbar()
    plt.tight_layout()
    
    save_fig_dir = f"{RESULT_DIR}/figs"
    os.makedirs(save_fig_dir, exist_ok=True)
    filename = f"conf.png"
    plt.savefig(os.path.join(save_fig_dir, filename))
    plt.close()
    print(f"Saved confusion matrix to {os.path.join(save_fig_dir, filename)}")

    # --- Trajectory Analysis ---
    os.makedirs(os.path.join(RESULT_DIR, "figs"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "data"), exist_ok=True)
    analyze_trajectories(X_train_list, y_train_labels, W_out, dt)

    print()

def delayed_space(sim, reservoir_state) -> None:
    rng = np.random.RandomState(cfg.SEED)
    dt = cfg.INPUT_DT if args.mode == "linear" else cfg.DT
    print("======== Delayed Space Task ========")

    print("\nLoading dataset...")

    all_meta = collect_file_metadata()
    train_meta = [m for m in all_meta if m["split"] == "train"]
    test_meta = [m for m in all_meta if m["split"] == "test"]

    rng.shuffle(train_meta)
    rng.shuffle(test_meta)
    train_meta = train_meta[:getattr(cfg, "N_TRAIN", len(train_meta))]
    test_meta = test_meta[:getattr(cfg, "N_TEST", len(test_meta))]
    print(f"selected Train files: {len(train_meta)}")
    print(f"selected Test files:  {len(test_meta)}")

    # process Data
    print()
    print("Processing data...")
    X_train_all, _, X_train_list, _ = load_and_process_data(
        train_meta, sim, reservoir_state, dt, dsec="TRAIN"
    )
    X_test_all, _, X_test_list, _ = load_and_process_data(
        test_meta, sim, reservoir_state, dt, dsec="TEST"
    )

    if len(X_train_all) == 0:
        print("Error: No training data.")
        return
    
    max_delay = 2.0
    delay_step = 0.05        #論文のカルシウムイメージングのサンプリングレートは20Hz(0.05s)
    
    delays = np.arange(0, max_delay + delay_step, delay_step)
    r2_scores = []
    
    print()

    for tau in tqdm(delays, desc="Delay Loop"):
        Y_train_tau_list = []
        for feat in X_train_list:
            n_steps = feat.shape[0]
            # 入力パルス(0.1s)を tau 遅らせたもの
            y = create_target_signal_timer(n_steps, dt, delay_sec=tau)
            Y_train_tau_list.append(y)
        Y_train_tau = np.vstack(Y_train_tau_list)
        
        # Test用ターゲット
        Y_test_tau_list = []
        for feat in X_test_list:
            n_steps = feat.shape[0]
            y = create_target_signal_timer(n_steps, dt, delay_sec=tau)
            Y_test_tau_list.append(y)
        Y_test_tau = np.vstack(Y_test_tau_list)

        # 2. 学習
        W_out = train_readout(X_train_all, Y_train_tau)
        
        # 3. 評価 (TestデータでのR2スコア)
        Y_pred = X_test_all @ W_out
        r2 = calc_r2_score(Y_test_tau, Y_pred)
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)

    # 記憶容量 (Memory Capacity) = Sum of R2
    mc = np.sum(r2_scores)
    print(f"\nMemory Capacity: {mc:.4f}")
    fit_decay_constant(delays, r2_scores, mc)

def analyze_trajectories(X_list: list[np.ndarray], y_list: list[int], W_out: np.ndarray, dt: float) -> None:
    """
    PCAによる軌道可視化と、クラス間・クラス内距離の計算
    """
    if not X_list:
        print("No data for analysis.")
        return

    print(f"\n--- Starting Trajectory Analysis ---")
    
    # 全トライアルで最小のデータ長に合わせる
    target_duration = 5.0
    target_steps = int(target_duration / dt)
    min_len = min([x.shape[0] for x in X_list])
    if min_len < target_steps:
        print(f"Warning: Data length ({min_len*dt:.2f}s) is shorter than target 5.0s. Using full length.")
        calc_steps = min_len
    else:
        calc_steps = target_steps
        print(f"Analyzing first {target_duration} seconds ({calc_steps} steps) of trajectories.")
    
    X_truncated = [x[:calc_steps, :] for x in X_list]
    
    Y_list = [x @ W_out for x in X_truncated]
    y_arr = np.array(y_list)

    # データを結合して正規化
    pca = PCA(n_components=3)
    scaler = StandardScaler()

    X_concat = np.vstack(X_truncated)
    X_standardized_concat = scaler.fit_transform(X_concat)
    
    n_trials = len(X_truncated)
    X_pca_concat = pca.fit_transform(X_standardized_concat)
    X_pca = X_pca_concat.reshape(n_trials, calc_steps, 3)

    # --- 1. PCA Analysis ---
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # クラスごとの色設定
    class_names = ["Top", "Middle", "Bottom"]
    colors = ['r', 'b', 'g']
    plotted_labels = set()
    
    for i in range(n_trials):
        label_idx = int(y_arr[i])
        c = colors[label_idx % len(colors)]
        l = class_names[label_idx % len(class_names)]
        
        if label_idx not in plotted_labels:
            ax.plot(X_pca[i, :, 0], X_pca[i, :, 1], X_pca[i, :, 2], color=c, alpha=0.6, label=l)
            plotted_labels.add(label_idx)
        else:
            ax.plot(X_pca[i, :, 0], X_pca[i, :, 1], X_pca[i, :, 2], color=c, alpha=0.6)
            
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'Trajectories in PC Subspace')
    ax.legend()
    
    save_path_pca = os.path.join(RESULT_DIR, "figs", "pca.png")
    plt.savefig(save_path_pca)
    plt.close()
    print(f"Saved PCA plot to {save_path_pca}")
    
    # データを保存用に成形 (2次元配列化)
    # X_pca shape: (n_trials, time_steps, 3)
    # 1. Trial ID [0, 0, ..., 1, 1, ...]
    trial_ids = np.repeat(np.arange(n_trials), calc_steps).reshape(-1, 1)
    # 2. Time [0, dt, 2dt, ..., 0, dt, ...]
    times = np.tile(np.arange(calc_steps) * dt, n_trials).reshape(-1, 1)
    # 3. PC1, PC2, PC3 (フラット化)
    pcs = X_pca.reshape(-1, 3)
    # 4. Label [0, 0, ..., 1, 1, ...]
    labels_expanded = np.repeat(y_arr, calc_steps).reshape(-1, 1)
    # 全て結合 (N*T, 6)
    pca_data_to_save = np.hstack((trial_ids, times, pcs, labels_expanded))
    filename = f"pca.npy"
    
    save_path_data = os.path.join(RESULT_DIR, "data", filename)
    np.save(save_path_data, pca_data_to_save)
    print(f"Saved PCA data to {save_path_data} (Shape: {pca_data_to_save.shape})")

    # --- 2. Distance Analysis ---
    def calc_mean_distances(data_list, labels, name=""):
        n_trials = len(data_list)
        n_dim = data_list[0].shape[1]

        data_concat = np.vstack(data_list)
        scaler = StandardScaler()
        data_std_concat = scaler.fit_transform(data_concat)
        
        data_std = data_std_concat.reshape(n_trials, calc_steps, n_dim)
        
        D_same_list = []
        D_diff_list = []
        D_same_list_t = []
        D_diff_list_t = []
        
        # 全ペアの組み合わせで距離を計算
        for i in range(n_trials):
            for j in range(i + 1, n_trials):
                # d(t) = ||p(t) - q(t)|| / sqrt(L)
                diff = data_std[i] - data_std[j]
                dist_t = np.linalg.norm(diff, axis=1) / np.sqrt(n_dim)
                
                D_pq = np.mean(dist_t)
                
                if labels[i] == labels[j]:
                    D_same_list.append(D_pq)
                    D_same_list_t.append(dist_t)
                else:
                    D_diff_list.append(D_pq)
                    D_diff_list_t.append(dist_t)
        
        # 平均値を計算 (ペアが存在しない場合はNaN)
        mean_same = np.mean(D_same_list) if D_same_list else np.nan
        mean_diff = np.mean(D_diff_list) if D_diff_list else np.nan
        
        print(f"{name}: D_same={mean_same:.4f}, D_diff={mean_diff:.4f}")
        return D_same_list_t, D_diff_list_t

    calc_mean_distances(X_truncated, y_arr, name="Reservoir x(t)")
    dist_same, dist_diff = calc_mean_distances(Y_list, y_arr, name="Output y(t)")

    if dist_same and dist_diff:
        dist_same = np.array(dist_same)
        dist_diff = np.array(dist_diff)
        
        mean_same = dist_same.mean(axis=0)
        std_same = dist_same.std(axis=0)
        mean_diff = dist_diff.mean(axis=0)
        std_diff = dist_diff.std(axis=0)

        lower_diff = np.maximum(mean_diff - std_diff, 0)
        lower_same = np.maximum(mean_same - std_same, 0)
        
        t_axis = np.arange(target_steps) * dt
        
        plt.figure(figsize=(8, 6))
        plt.plot(t_axis, mean_diff, label='Different Class', color='blue')
        plt.fill_between(t_axis, lower_diff, mean_diff + std_diff, color='blue', alpha=0.2)
        
        plt.plot(t_axis, mean_same, label='Same Class', color='red')
        plt.fill_between(t_axis, lower_same, mean_same + std_same, color='red', alpha=0.2)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized distance')
        plt.title(f'Instantaneous normalized distance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path_dist = os.path.join(RESULT_DIR, "figs", "dist.png")
        plt.savefig(save_path_dist)
        plt.close()
        print(f"Saved Dist plot to {save_path_dist}")

        # --- 距離解析データの保存 ---
        # データを結合して 2次元配列 (TimeSteps, 5) を作成
        # Col 0: Time [s]
        # Col 1: Mean (Different Class)
        # Col 2: Std  (Different Class)
        # Col 3: Mean (Same Class)
        # Col 4: Std  (Same Class)
        dist_data_to_save = np.column_stack((t_axis, mean_diff, std_diff, mean_same, std_same))
        filename = f"dist.npy"
        save_path_npy = os.path.join(RESULT_DIR, "data", filename)
        np.save(save_path_npy, dist_data_to_save)
        print(f"Saved Dist data to {save_path_npy} (Shape: {dist_data_to_save.shape})")
    else:
        print("Skipping distance analysis: Not enough pairs.")

def fit_decay_constant(delays, scores, mc):
    """
    決定係数の減衰カーブに指数関数をフィッティングし、減衰定数(tau)を求める
    """
    print("\n--- Fitting Decay Constant ---")
    
    # 欠損値(NaN)や無限大を除去
    valid_mask = np.isfinite(scores)
    t_data = delays[valid_mask]
    y_data = scores[valid_mask]
    
    if len(t_data) < 3:
        print("Not enough data points for fitting.")
        return

    # 初期値の推定 [A, tau, C]
    # A: 最大値, tau: データ範囲の半分くらい, C: 最小値
    p0 = [np.max(y_data), np.max(t_data)/2.0, 0.0]
    
    # bounds: A>0, tau>0
    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(exponential_decay, t_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
        A_opt, tau_opt, C_opt = popt
        
        print(f"Fitted Parameters:")
        print(f"  A (Amplitude) = {A_opt:.4f}")
        print(f"  tau (Decay Constant) = {tau_opt:.4f} [s]")
        print(f"  C (Offset) = {C_opt:.4f}")
        
        # フィッティング結果のプロット
        plt.figure(figsize=(8, 6))
        # 生データ
        plt.plot(t_data, y_data, 'o-', color='blue', alpha=0.5, label=f"Measured (MC={mc:.2f})")
        
        # 近似曲線
        t_fit = np.linspace(min(t_data), max(t_data), 200)
        y_fit = exponential_decay(t_fit, *popt)
        plt.plot(t_fit, y_fit, 'r-', linewidth=2, label=f'Fit: $\\tau_{{decay}}={tau_opt:.3f}s$')
        
        plt.title(f"Memory Capacity & Decay Fit")
        plt.xlabel("Delay $\\tau$ [s]")
        plt.ylabel("$R^2$ Score")
        plt.legend()
        plt.grid(True)
        
        filename = "short-term-memory.png"
        save_path = os.path.join(RESULT_DIR, "figs", filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {filename}")
        data = np.column_stack([delays, scores])
        np.save(f"{RESULT_DIR}/data/{filename}.npy", data)
        
        return tau_opt

    except Exception as e:
        print(f"Fitting failed: {e}")

        plt.figure(figsize=(8, 6))
        plt.plot(t_data, y_data, 'o-', color='blue', label=f"Measured (MC={mc:.2f})")
        plt.title("Memory Capacity (Fitting Failed)")
        plt.xlabel("Delay $\\tau$ [s]")
        plt.ylabel("$R^2$ Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULT_DIR, "figs", "short-term-memory.png"))
        plt.close()
        return None

def train_readout(X, Y, lambda_reg=1.0):
    # Ridge Regression (Linear Readout)# Ridge Regression: W = (X^T X + lambda I)^-1 X^T Y
    num_features = X.shape[1]
    I = np.eye(num_features)
    XtX = X.T @ X
    XtY = X.T @ Y
    W_out = np.linalg.solve(XtX + lambda_reg * I, XtY)
    return W_out

def predict(W_out, feat, dt) -> np.intp:
    """
    空間認識用: 出力を時間積分(2.5s)して最大値判定
    """
    y_seq = feat @ W_out
    
    # 積分
    y_integrated = y_seq.sum(axis=0)
    return np.argmax(y_integrated)

def calc_r2_score(y_true, y_pred):
    """
    決定係数 R^2 を計算
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if np.std(y_true) == 0:
        return 0.0
    corr_matrix = np.corrcoef(y_true, y_pred)
    if np.isnan(corr_matrix[0, 1]):
        return 0.0
    r = corr_matrix[0, 1]
    return r ** 2

def exponential_decay(t, A, tau, C):
    """
    指数関数モデル: y = A * exp(-t / tau) + C
    """
    return A * np.exp(-t / tau) + C

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


if __name__ == "__main__":
    # RNN_config.set_global_seed(cfg.SEED)
    main()
