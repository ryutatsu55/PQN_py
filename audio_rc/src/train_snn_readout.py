import json
import numpy as np
import glob
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import os
import shutil

import sys
from pathlib import Path
# プロジェクトルートへのパス設定
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
import config
import src.PQN_RNN_onGPU as PQN_RNN_onGPU

# Load Config
cfg = config.Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["snn", "feature", "linear"],
    default="feature",
    help="snn: run SNN to compute features, feature: load saved feature .npy, linear: use cochleagram directly",
)
parser.add_argument(
    "--task",
    choices=["digit", "gender", "m2f_digit", "f2m_digit", "z2o_gender", "o2z_gender"],
    default="digit",
    help="Select the recognition task.\n"
         "digit: 0 vs 1 (mixed gender)\n"
         "gender: Female vs Male (mixed digit)\n"
         "m2f_digit: Train digit on Male, Test on Female\n"
         "f2m_digit: Train digit on Female, Test on Male\n"
         "z2o_gender: Train gender on Zero, Test on One\n"
         "o2z_gender: Train gender on One, Test on Zero"
)
parser.add_argument(
    "--seed", type=int, default=cfg.SEED,
    help="random seed"
)
parser.add_argument(
    "--overwrite", action="store_true",
    help="If True, re-calculate SNN even if feature file exists."
)
args = parser.parse_args()


# ================================
# 1. Helper Functions
# ================================

def pad_and_integrate(x):
    """
    時系列特徴量を時間方向に積分（合計）して、固定長の空間パターンに変換する。
    Args:
        x (np.ndarray): Shape (Time, Neurons) or (Time, Channels)
    Returns:
        np.ndarray: Shape (Neurons,)
    """
    return x.sum(axis=0)

def get_class_names(task):
    if "digit" in task:
        return ["Zero", "One"]
    elif "gender" in task:
        return ["Female", "Male"]
    return ["Class 0", "Class 1"]

# ================================
# 2. Data Loading & Labeling Logic
# ================================
def collect_file_metadata():
    base_dirs = {
        "train": "audio_rc/reservoir_inputs/train",
        "test": "audio_rc/reservoir_inputs/test"
    }
    # カテゴリ定義 (ディレクトリ名 -> 属性)
    categories = [
        {"dir": "coch_female_zero", "gender": "female", "digit": 0},
        {"dir": "coch_female_one",  "gender": "female", "digit": 1},
        {"dir": "coch_male_zero",   "gender": "male",   "digit": 0},
        {"dir": "coch_male_one",    "gender": "male",   "digit": 1},
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
                    "gender": cat["gender"],
                    "digit": cat["digit"],
                    "split": split, # original split ('train' or 'test')
                    "subdir_name": subdir # 保存先ディレクトリ作成用
                })
    
    return metadata_list

def select_files_for_task(metadata_list, task, rng):
    train_items = []
    test_items = []

    for item in metadata_list:
        gender = item["gender"]
        digit = item["digit"]
        original_split = item["split"]
        
        is_train = False
        is_test = False
        label = -1

        # --- Task Logic (振り分けルール) ---
        if task == "digit":
            label = digit
            if original_split == "train": is_train = True
            elif original_split == "test": is_test = True

        elif task == "gender":
            label = 0 if gender == "female" else 1
            if original_split == "train": is_train = True
            elif original_split == "test": is_test = True

        elif task == "m2f_digit":
            if gender == "male" and original_split == "train":
                is_train = True
                label = digit
            elif gender == "female" and original_split == "test":
                is_test = True
                label = digit

        elif task == "f2m_digit":
            if gender == "female" and original_split == "train":
                is_train = True
                label = digit
            elif gender == "male" and original_split == "test":
                is_test = True
                label = digit

        elif task == "z2o_gender":
            label = 0 if gender == "female" else 1
            if digit == 0 and original_split == "train": is_train = True
            elif digit == 1 and original_split == "test": is_test = True

        elif task == "o2z_gender":
            label = 0 if gender == "female" else 1
            if digit == 1 and original_split == "train": is_train = True
            elif digit == 0 and original_split == "test": is_test = True

        # ラベル情報を付与してリストに追加
        if is_train:
            item["label"] = label
            train_items.append(item)
        elif is_test:
            item["label"] = label
            test_items.append(item)

    # --- Shuffling ---
    rng.shuffle(train_items)
    rng.shuffle(test_items)

    # --- Apply Limits (N_TRAIN, N_TEST) ---
    # config.py に N_TRAIN, N_TEST が定義されている前提
    # 定義されていない場合は全数使用
    n_train_limit = getattr(cfg, "N_TRAIN_COCH", len(train_items))
    n_test_limit = getattr(cfg, "N_TEST_COCH", len(test_items))

    train_items = train_items[:n_train_limit]
    test_items = test_items[:n_test_limit]

    return train_items, test_items

# ================================
# 3. Execution (Simulate or Load)
# ================================

def get_feature_save_path(input_path, original_split, subdir_name):
    """
    入力パスに対応する特徴量の保存先パスを生成する。
    構造: audio_rc/reservoir_outputs/{train|test}/features_{subdir}/{filename}
    """
    filename = os.path.basename(input_path)
    # coch_female_zero -> features_female_zero
    feat_subdir = subdir_name.replace("coch_", "features_")
    
    save_dir = os.path.join("audio_rc", "reservoir_outputs", original_split, feat_subdir)
    save_path = os.path.join(save_dir, filename)
    return save_path, save_dir

def process_and_load_data(items, sim, reservoir_state):
    """
    Returns:
        X_flat: (N, Neurons) - Integrated features for readout
        y: (N,) - Labels
        paths: List[str] - File paths
        X_time: List[np.ndarray] - Raw time-series features for analysis
    """
    X_flat = []
    X_time = []
    y = []
    loaded_paths = []

    for i, item in enumerate(tqdm(items, desc="Processing")):
        input_path = item["input_path"]
        label = item["label"]
        original_split = item["split"]
        subdir_name = item["subdir_name"]

        # 特徴量の保存先パスを決定
        save_path, save_dir = get_feature_save_path(input_path, original_split, subdir_name)
        feat = None
        
        # --- Mode: Linear (SNNを使わない) ---
        if args.mode == "linear":
            feat = np.load(input_path)
            # 線形の場合も pad_and_integrate をするかはタスクによるが、
            # 形式を合わせるためここでは適用する（必要に応じて変更してください）
            # feat = pad_and_integrate(feat)

        # --- Mode: Feature (既存ファイルのみロード) ---
        elif args.mode == "feature":
            if os.path.exists(save_path):
                feat = np.load(save_path)
                # feat = pad_and_integrate(feat)
            else:
                # featureモードなのにファイルがない場合はスキップするかエラーにする
                # ここではスキップ
                print(f"Warning: Feature file not found: {save_path}")
                continue

        # --- Mode: SNN (シミュレーション + キャッシング) ---
        elif args.mode == "snn":
            # キャッシュチェック
            if os.path.exists(save_path) and not args.overwrite:
                # キャッシュがあればロード
                feat = np.load(save_path)
            else:
                # キャッシュがない、または上書き指定なら計算
                os.makedirs(save_dir, exist_ok=True)
                
                coch = np.load(input_path)
                # SNN実行
                feat = PQN_RNN_onGPU.main(
                    input_data=coch,
                    coch=True,
                    reservoir_state=reservoir_state,
                    return_feature=True,
                    is_debug_print=False,
                    record=True if i+1 == len(items) else False,
                    S_durt=cfg.INPUT_DT_COCH if hasattr(cfg, "INPUT_DT_COCH") else 0.01,
                    cfg=cfg,
                    sim=sim
                )
                # 保存（生の時系列特徴量を保存しておく）
                np.save(save_path, feat)
            
            # ロード/計算後に積分
            # feat = pad_and_integrate(feat)

        if feat is not None:
            X_time.append(feat)
            X_flat.append(pad_and_integrate(feat))
            y.append(label)
            loaded_paths.append(input_path)

    return np.array(X_flat), np.array(y), loaded_paths, X_time

def analyze_trajectories(X_list: list[np.ndarray], y_list: list[int], save_dir: str, dt: float, task_name: str) -> None:
    """
    PCAによる軌道可視化と、クラス間・クラス内距離の計算
    """
    if not X_list:
        print("No data for analysis.")
        return

    print(f"\nStarting Trajectory Analysis for task: {task_name}...")
    
    # 全トライアルで最小のデータ長に合わせる
    min_len = min([x.shape[0] for x in X_list])
    X_truncated = [x[:min_len, :] for x in X_list]
    
    # データを結合して正規化
    X_concat = np.vstack(X_truncated)
    scaler = StandardScaler()
    X_standardized_concat = scaler.fit_transform(X_concat)
    
    n_trials = len(X_truncated)
    time_steps = min_len
    n_neurons = X_truncated[0].shape[1]
    X_reshaped = X_standardized_concat.reshape(n_trials, time_steps, n_neurons)
    y_arr = np.array(y_list)

    # --- 1. PCA Analysis ---
    pca = PCA(n_components=3)
    X_pca_concat = pca.fit_transform(X_standardized_concat)
    X_pca = X_pca_concat.reshape(n_trials, time_steps, 3)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # クラスごとの色設定
    class_names = get_class_names(task_name)
    colors = ['r', 'b', 'g', 'c', 'm', 'y']
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
    ax.set_title(f'Trajectories in PC Subspace ({task_name})')
    ax.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    save_path_pca = os.path.join(save_dir, f"pca.png")
    plt.savefig(save_path_pca)
    plt.close()
    print(f"Saved PCA plot to {save_path_pca}")

    # --- 2. Distance Analysis ---
    dist_same = []
    dist_diff = []
    
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            diff = X_reshaped[i] - X_reshaped[j] 
            dist_t = np.linalg.norm(diff, axis=1) / np.sqrt(n_neurons)
            
            if y_arr[i] == y_arr[j]:
                dist_same.append(dist_t)
            else:
                dist_diff.append(dist_t)
                
    if dist_same and dist_diff:
        dist_same = np.array(dist_same)
        dist_diff = np.array(dist_diff)
        
        mean_same = dist_same.mean(axis=0)
        std_same = dist_same.std(axis=0)
        mean_diff = dist_diff.mean(axis=0)
        std_diff = dist_diff.std(axis=0)

        lower_diff = np.maximum(mean_diff - std_diff, 0)
        lower_same = np.maximum(mean_same - std_same, 0)
        
        t_axis = np.arange(time_steps) * dt
        
        plt.figure(figsize=(8, 6))
        plt.plot(t_axis, mean_diff, label='Different Class', color='blue')
        plt.fill_between(t_axis, lower_diff, mean_diff + std_diff, color='blue', alpha=0.2)
        
        plt.plot(t_axis, mean_same, label='Same Class', color='red')
        plt.fill_between(t_axis, lower_same, mean_same + std_same, color='red', alpha=0.2)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized distance')
        plt.title(f'Instantaneous normalized distance ({task_name})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path_dist = os.path.join(save_dir, f"dist.png")
        plt.savefig(save_path_dist)
        plt.close()
        print(f"Saved Distance plot to {save_path_dist}")
    else:
        print("Skipping distance analysis: Not enough pairs.")

# ================================
# 4. Training & Evaluation
# ================================

def train_readout(X, y, lambda_reg=1.0):
    # Ridge Regression (Linear Readout)
    # W_out = (X^T X + lambda I)^-1 X^T y
    # X shape: (Samples, Neurons)
    # y (one-hot) shape: (Samples, Classes)
    
    num_samples, num_features = X.shape
    classes = np.unique(y)
    num_classes = len(classes)

    # Make One-hot targets
    Y_onehot = np.zeros((num_samples, num_classes))
    for i, label in enumerate(y):
        Y_onehot[i, label] = 1.0

    # Normal Equation
    I = np.eye(num_features)
    XtX = X.T @ X
    XtY = X.T @ Y_onehot
    
    # Solve
    W_out = np.linalg.solve(XtX + lambda_reg * I, XtY)
    return W_out

def predict(W_out, X):
    logits = X @ W_out
    return np.argmax(logits, axis=1)

def evaluate(W_out, X, y):
    preds = predict(W_out, X)
    acc = np.mean(preds == y)
    return acc

# ================================
# 5. Main Process
# ================================
def main():
    rng = np.random.RandomState(cfg.SEED)

    print(f"Mode: {args.mode}")
    print(f"Task: {args.task}")
    print(f"Reservoir Cells: {cfg.N}")

    # 1. Initialize Simulator (SNNモードの場合のみ)
    sim = None
    reservoir_state = None
    if args.mode == "snn":
        print("Initializing Reservoir...")
        reservoir_state = config.init_reservoir()
        sim = PQN_RNN_onGPU.PQN_Reservoir_GPU(reservoir_state, cfg)

    # 2. Collect Metadata (No heavy loading yet)
    print("\n--- Scanning Files ---")
    all_metadata = collect_file_metadata()
    print(f"Total files found: {len(all_metadata)}")

    # 3. Select Files (Filter by Task & Limit N)
    print(f"\n--- Selecting Files for Task: {args.task} ---")
    # N_TRAIN/N_TEST はここで config から読み込まれ適用される
    train_meta, test_meta = select_files_for_task(all_metadata, args.task, rng)

    print(f"Selected Train: {len(train_meta)}")
    print(f"Selected Test:  {len(test_meta)}")

    # 4. Process (Load Cache or Simulate)
    print("\n--- Processing Training Data ---")
    X_train, y_train, _, X_train_ts = process_and_load_data(train_meta, sim, reservoir_state)

    print("\n--- Processing Test Data ---")
    X_test, y_test, test_paths, X_test_ts = process_and_load_data(test_meta, sim, reservoir_state)

    if len(X_train) == 0:
        print("Error: No training data.")
        return
    
    save_dir_figs = "audio_rc/result/figs"
    os.makedirs(save_dir_figs, exist_ok=True)
    analyze_trajectories(X_train_ts, y_train, save_dir_figs, cfg.DT, args.task)
    del X_train_ts, X_test_ts
    gc.collect()
    
    # 5. Train & Evaluate
    print("\n--- Training Readout ---")
    W_out = train_readout(X_train, y_train, lambda_reg=1e-2)

    acc_train = evaluate(W_out, X_train, y_train)
    acc_test = evaluate(W_out, X_test, y_test)

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # 6. Confusion Matrix & Save (Brief version)
    class_names = get_class_names(args.task)
    num_classes = len(class_names)
    preds_test = predict(W_out, X_test)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_test)):
        true_l = y_test[i]
        pred_l = preds_test[i]
        conf_matrix[true_l, pred_l] += 1

    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title(f"Confusion Matrix ({args.task})\nAcc: {acc_test*100:.1f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(num_classes), class_names)
    plt.yticks(np.arange(num_classes), class_names)
    
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black", fontsize=16)
            
    plt.colorbar()
    plt.tight_layout()
    
    save_fig_dir = "audio_rc/result/figs"
    os.makedirs(save_fig_dir, exist_ok=True)
    filename = f"conf.png"
    plt.savefig(os.path.join(save_fig_dir, filename))
    plt.close()
    print(f"Saved confusion matrix to {os.path.join(save_fig_dir, filename)}")

   # Save JSON log
    res_dir = "audio_rc/result"
    os.makedirs(res_dir, exist_ok=True)
    res = {
        # "timestamp": ts, 
        "task": args.task, 
        "seed": args.seed,
        "n_train_limit": getattr(cfg, "N_TRAIN", "all"),
        "acc_test": acc_test
    }
    with open(os.path.join(res_dir, "results_snn.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")

    # Save Weights
    weight_dir = "audio_rc/reservoir_outputs"
    os.makedirs(weight_dir, exist_ok=True)
    np.save(os.path.join(weight_dir, f"W_out.npy"), W_out)
    print(f"Saved weights to W_out_{args.task}.npy")

    # if misclassified:
    #     print(f"\n{len(misclassified)} Misclassified Samples (First 5):")
        # for m in misclassified[:5]:
        #     print(f"  Path: {os.path.basename(m[0])}, True: {class_names[m[1]]}, Pred: {class_names[m[2]]}")

if __name__ == "__main__":
    main()