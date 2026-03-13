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

# --- ディレクトリパス設定 ---
BASE_DIR = "audio_rc"
INPUT_DIR = os.path.join(BASE_DIR, "reservoir_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "reservoir_outputs")
RESULT_DIR = os.path.join(BASE_DIR, "result")

# Load Config
cfg = config.Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["snn", "feature", "linear"],
    default="linear",
    help="snn: run SNN to compute features, feature: load saved feature .npy, linear: use cochleagram directly",
)
parser.add_argument(
    "--task",
    choices=[
        "f2f_digit", 
        "m2m_digit", 
        "z2z_gender", 
        "o2o_gender", 
        "m2f_digit", 
        "f2m_digit", 
        "z2o_gender", 
        "o2z_gender", 
        "reverse_f_z",
        ]
        ,
    default="f2f_digit",
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
    elif "reverse_f_z" in task:
        return ["Forward", "Reverse"]
    return ["Class 0", "Class 1"]

def create_target_signal(n_steps, label, num_classes, dt):
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
        {"dir": "coch_female_zero", "gender": "female", "digit": 0, "reverse": False},
        {"dir": "coch_female_one",  "gender": "female", "digit": 1, "reverse": False},
        {"dir": "coch_male_zero",   "gender": "male",   "digit": 0, "reverse": False},
        {"dir": "coch_male_one",    "gender": "male",   "digit": 1, "reverse": False},
        {"dir": "rev_coch_female_zero", "gender": "female", "digit": 0, "reverse": True},
        {"dir": "rev_coch_female_one",  "gender": "female", "digit": 1, "reverse": True},
        {"dir": "rev_coch_male_zero",   "gender": "male",   "digit": 0, "reverse": True},
        {"dir": "rev_coch_male_one",    "gender": "male",   "digit": 1, "reverse": True},
    ]

    metadata_list = []

    # --- Mode: Feature (抽出済みファイルのみ対象) ---
    if args.mode == "feature":
        base_dirs = {
            "train": "audio_rc/reservoir_outputs/train",
            "test": "audio_rc/reservoir_outputs/test"
        }
        for split, base_dir in base_dirs.items():
            for cat in categories:
                # 特徴量ディレクトリ名は "coch_" -> "features_" に置換されている
                feat_subdir = cat["dir"].replace("coch_", "features_")
                search_path = os.path.join(base_dir, feat_subdir, "*.npy")
                files = sorted(glob.glob(search_path))
                
                for path in files:
                    # path は特徴量ファイルのパス
                    # input_path として保持するが、後で feature モードならそのままロードされる
                    metadata_list.append({
                        "input_path": path, # ここが特徴量パスになる
                        "gender": cat["gender"],
                        "digit": cat["digit"],
                        "reverse": cat["reverse"],
                        "split": split,
                        "subdir_name": cat["dir"] # 元のsubdir名も保持
                    })

    # --- Mode: SNN / Linear (元データを対象) ---
    else:
        base_dirs = {
            "train": "audio_rc/reservoir_inputs/train",
            "test": "audio_rc/reservoir_inputs/test"
        }
        for split, base_dir in base_dirs.items():
            for cat in categories:
                subdir = cat["dir"]
                search_path = os.path.join(base_dir, subdir, "*.npy")
                files = sorted(glob.glob(search_path))
                
                for path in files:
                    metadata_list.append({
                        "input_path": path,
                        "gender": cat["gender"],
                        "digit": cat["digit"],
                        "reverse": cat["reverse"],
                        "split": split,
                        "subdir_name": subdir
                    })
    
    return metadata_list

def select_files_for_task(metadata_list, task, rng):
    candidates_train = []
    candidates_test = []

    for item in metadata_list:
        gender = item["gender"]
        digit = item["digit"]
        reverse = item["reverse"]
        original_split = item["split"]
        
        is_train = False
        is_test = False
        label = -1

        # --- Task Logic (振り分けルール) ---
        if task == "f2f_digit":
            if reverse: continue
            label = digit
            if gender == "female" and original_split == "train": is_train = True
            elif gender == "female" and original_split == "test": is_test = True
        
        elif task == "m2m_digit":
            if reverse: continue
            label = digit
            if gender == "male" and original_split == "train": is_train = True
            elif gender == "male" and original_split == "test": is_test = True

        elif task == "z2z_gender":
            if reverse: continue
            label = 0 if gender == "female" else 1
            if digit == 0 and original_split == "train": is_train = True
            elif digit == 0 and original_split == "test": is_test = True

        elif task == "o2o_gender":
            if reverse: continue
            label = 0 if gender == "female" else 1
            if digit == 1 and original_split == "train": is_train = True
            elif digit == 1 and original_split == "test": is_test = True

        elif task == "m2f_digit":
            if reverse: continue
            label = digit
            if gender == "male" and original_split == "train":
                is_train = True
            elif gender == "female" and original_split == "test":
                is_test = True

        elif task == "f2m_digit":
            if reverse: continue
            label = digit
            if gender == "female" and original_split == "train":
                is_train = True
            elif gender == "male" and original_split == "test":
                is_test = True

        elif task == "z2o_gender":
            if reverse: continue
            label = 0 if gender == "female" else 1
            if digit == 0 and original_split == "train": is_train = True
            elif digit == 1 and original_split == "test": is_test = True

        elif task == "o2z_gender":
            if reverse: continue
            label = 0 if gender == "female" else 1
            if digit == 1 and original_split == "train": is_train = True
            elif digit == 0 and original_split == "test": is_test = True

        elif task == "reverse_f_z":
            label = 1 if reverse else 0
            if digit == 0 and gender == "female":
                if original_split == "train": is_train = True
                elif original_split == "test": is_test = True

        # ラベル情報を付与してリストに追加
        if is_train:
            item["label"] = label
            candidates_train.append(item)
        elif is_test:
            item["label"] = label
            candidates_test.append(item)


    # Configから回数を取得
    n_train_target = getattr(cfg, "N_TRAIN_COCH", len(candidates_train))
    n_test_target = getattr(cfg, "N_TEST_COCH", len(candidates_test))

    if args.mode != "feature":
        
        def expand_representative_files(items, n_limit):
            """クラスごとに1つ選び、n_limit回複製する"""
            grouped = {}
            # クラスごとにグループ化 (gender, digit, reverse)
            for it in items:
                key = (it["gender"], it["digit"], it["reverse"])
                if key not in grouped: grouped[key] = []
                grouped[key].append(it)
            
            expanded = []
            for key, group in grouped.items():
                if not group: continue
                # 各クラスの先頭のファイルを代表として選ぶ
                rep = group[0]
                
                # 指定回数だけ複製してリストに追加
                for i in range(n_limit):
                    new_item = rep.copy()
                    new_item["rep_idx"] = i  # 繰り返し番号 (0, 1, ..., 19)
                    expanded.append(new_item)
            return expanded

        # train_items = expand_representative_files(candidates_train, n_train_target)
        # test_items = expand_representative_files(candidates_test, n_test_target)
        train_items = candidates_train
        test_items = candidates_test
        # --- Shuffling ---
        rng.shuffle(train_items)
        rng.shuffle(test_items)
        train_items = train_items[:n_train_target]
        test_items = test_items[:n_test_target]
        # print(train_items)
        # print(test_items)
        
    else:
        # Featureモードの時は、保存されているファイルをそのまま使う (既に増殖済み)
        train_items = candidates_train
        test_items = candidates_test
        # --- Shuffling ---
        rng.shuffle(train_items)
        rng.shuffle(test_items)
        train_items = train_items[:n_train_target]
        test_items = test_items[:n_test_target]
        # train_items = train_items[:getattr(cfg, "N_TRAIN_COCH", 20)]

    return train_items, test_items

# ================================
# 3. Execution (Simulate or Load)
# ================================

def get_feature_save_path(input_path, original_split, subdir_name, rep_idx=None):
    """
    入力パスに対応する特徴量の保存先パスを生成する。
    構造: audio_rc/reservoir_outputs/{train|test}/features_{subdir}/{filename}_rep{rep_idx}
    """
    filename = os.path.basename(input_path)
    if rep_idx is not None:
        name, ext = os.path.splitext(filename)
        filename = f"{name}_rep{rep_idx}{ext}"
    # coch_female_zero -> features_female_zero
    feat_subdir = subdir_name.replace("coch_", "features_")
    
    save_dir = os.path.join(OUTPUT_DIR, original_split, feat_subdir)
    save_path = os.path.join(save_dir, filename)
    return save_path, save_dir

def process_and_load_data(items, sim, reservoir_state, dt, recorded_voice, desc="Processing data"):
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
    class_names = []

    num_classes = 2

    for i, item in enumerate(tqdm(items, desc=desc)):
        input_path = item["input_path"]
        label = item["label"]
        digit = item["digit"]
        gender = item["gender"]
        reverse = item["reverse"]
        original_split = item["split"]
        subdir_name = item["subdir_name"]
        class_name = f"{gender}{digit}"
        class_name = f"rev_{class_name}" if reverse else class_name

        # 特徴量の保存先パスを決定
        if args.mode == "feature":
            save_path = input_path
            save_dir = os.path.dirname(input_path)
        else:
            # rep_idx = item["rep_idx"]
            rep_idx = None
            save_path, save_dir = get_feature_save_path(input_path, original_split, subdir_name, rep_idx)
        feat = None
        
        tmax = None
        n_steps = None
        if original_split == "train":
            if hasattr(cfg, "DURATION_INTERVAL_COCH"):
                tmax = cfg.DURATION_INTERVAL_COCH
                n_steps = int(tmax / dt) 
        else:
            if hasattr(cfg, "TEACHING_DURATION"):
                tmax = cfg.TEACHING_DURATION
                n_steps = int(tmax / dt)
        
        # --- Mode: Linear (SNNを使わない) ---
        if args.mode == "linear":
            feat = np.load(input_path)
            current_len = feat.shape[0]
            n_steps = current_len if n_steps is None else n_steps
            if current_len < n_steps:
                padding = np.zeros((n_steps - current_len, feat.shape[1]))
                feat = np.vstack([feat, padding])
            else:
                feat = feat[:n_steps]

        # --- Mode: Feature (既存ファイルのみロード) ---
        elif args.mode == "feature":
            feat = np.load(save_path)

        # --- Mode: SNN (シミュレーション + キャッシング) ---
        elif args.mode == "snn":
            os.makedirs(save_dir, exist_ok=True)
            
            coch = np.load(input_path)
            if recorded_voice is None:
                current_record = None
            elif class_name not in recorded_voice:
                current_record = {
                    "result_dir": RESULT_DIR,
                    "filename": class_name,
                }
                # print("Recording class:", class_name)
                recorded_voice.add(class_name)
            else:
                current_record = None
            feat = PQN_RNN_onGPU.main(
                input_data=coch,
                coch=True,
                reservoir_state=reservoir_state,
                return_feature=True,
                is_debug_print=False,
                # record=True if i+1 == len(items) else False,
                record=current_record,
                tmax=tmax if tmax is not None else None,
                S_durt=cfg.INPUT_DT_COCH if hasattr(cfg, "INPUT_DT_COCH") else 0.01,
                cfg=cfg,
                sim=sim
            )
            # 保存（生の時系列特徴量を保存しておく）
            np.save(save_path, feat)
            
        if feat is not None:
            n_steps = feat.shape[0]
            
            # リストに追加
            X_list.append(feat)
            y_labels.append(label)
            class_names.append(class_name)

            # ターゲット信号の生成 (Time, Classes)
            target = create_target_signal(n_steps, label, num_classes, dt)
            Y_list.append(target)
    # 結合 (学習用)
    if len(X_list) > 0:
        X_concat = np.vstack(X_list)
        Y_concat = np.vstack(Y_list)
    else:
        X_concat, Y_concat = np.array([]), np.array([])
        
    return X_concat, Y_concat, X_list, y_labels, class_names

def analyze_trajectories(X_list: list[np.ndarray], y_list: list[int], class_names: list[str], save_dir: str, dt: float, task_name: str) -> None:
    """
    PCAによる軌道可視化と、クラス間・クラス内距離の計算
    """
    if not X_list:
        print("No data for analysis.")
        return

    print(f"\n--- Starting Trajectory Analysis for task: {task_name} ---")
    
    # 全トライアルで最小のデータ長に合わせる
    min_len = min([x.shape[0] for x in X_list])
    print(f"Truncating all trials to length: {min_len*dt:.3f} seconds")
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
    colors = {
        "female0": "r",
        "female1": "b",
        "male0": "g",
        "male1": "m",
        "rev_female0": "y",
    }
    plotted_labels = set()
    
    for i in range(n_trials):
        c = colors[class_names[i]] if class_names[i] in colors else "k"
        label = class_names[i]
        
        if label not in plotted_labels:
            ax.plot(X_pca[i, :, 0], X_pca[i, :, 1], X_pca[i, :, 2], color=c, alpha=0.6, label=label)
            plotted_labels.add(label)
        else:
            ax.plot(X_pca[i, :, 0], X_pca[i, :, 1], X_pca[i, :, 2], color=c, alpha=0.6)
            
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'Trajectories in PC Subspace ({task_name})')
    ax.legend()
    
    save_path_pca = os.path.join(save_dir, "figs", "pca.png")
    plt.savefig(save_path_pca)
    plt.close()
    print(f"Saved PCA plot to {save_path_pca}")
    
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
    
    save_path_data = os.path.join(save_dir, "data", filename)
    np.save(save_path_data, pca_data_to_save)
    print(f"Saved PCA data to {save_path_data} (Shape: X_pca shape: (n_trials, time_steps, 3))")

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
        
        filename = "dist.png"
        save_path_dist = os.path.join(save_dir, "figs", filename)
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
        filename = "dist.npy"
        save_path_npy = os.path.join(save_dir, "data", filename)
        np.save(save_path_npy, dist_data_to_save)
        print(f"Saved distance analysis data to {save_path_npy} (Shape: (time[s], mean_diff, std_diff, mean_same, std_same))")
    else:
        print("Skipping distance analysis: Not enough pairs.")

# ================================
# 4. Training & Evaluation
# ================================

def train_readout(X, Y, lambda_reg=1.0):
    # Ridge Regression (Linear Readout)
    # W_out = (X^T X + lambda I)^-1 X^T y
    # X shape: (Samples, Neurons)
    # y (one-hot) shape: (Samples, Classes)
    
    num_samples, num_features = X.shape
    classes = np.unique(Y)
    num_classes = len(classes)

    # # Make One-hot targets
    # Y_onehot = np.zeros((num_samples, num_classes))
    # for i, label in enumerate(y):
    #     Y_onehot[i, label] = 1.0

    # Normal Equation
    I = np.eye(num_features)
    XtX = X.T @ X
    XtY = X.T @ Y
    
    # Solve
    W_out = np.linalg.solve(XtX + lambda_reg * I, XtY)
    return W_out

def predict(W_out, X):
    y_seq = X @ W_out
    y_integrated = y_seq.sum(axis=0)
    return np.argmax(y_integrated, axis=0)

# def evaluate(W_out, X, y):
#     preds = predict(W_out, X)
#     acc = np.mean(preds == y)
#     return acc

# ================================
# 5. Main Process
# ================================
def main():
    rng = np.random.RandomState(args.seed)

    print(f"Mode: {args.mode}")
    print(f"Task: {args.task}")
    print(f"SEED: {args.seed}")
    print(f"Reservoir Cells: {cfg.N}")

    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
        print(f"deleted following directory: {RESULT_DIR} ( to make new input dataset )")
    os.makedirs(os.path.join(RESULT_DIR, "figs"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "data"), exist_ok=True)

    # 1. Initialize Simulator (SNNモードの場合のみ)
    sim = None
    reservoir_state = None
    if args.mode == "snn":
        print("Initializing Reservoir...")
        reservoir_state = config.init_reservoir(args.seed)
        sim = PQN_RNN_onGPU.PQN_Reservoir_GPU(reservoir_state, cfg)
    dt = cfg.INPUT_DT_COCH if args.mode == "linear" else cfg.DT

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

    rmv_dir = "audio_rc/reservoir_outputs"
    if os.path.exists(rmv_dir) and args.mode == "snn":
        shutil.rmtree(rmv_dir)
        print(f"\ndeleted following directory: {rmv_dir} ( to make new featured dataset )")

    # 4. Process (Load Cache or Simulate)
    recorded_voice = set()
    print("\n--- Processing Train Data ---")
    X_train_all, Y_train_all, X_train_list, y_train_labels, class_names_train = process_and_load_data(
        train_meta, sim, reservoir_state, dt, recorded_voice, desc="TRAIN"
        )

    print("\n--- Processing Test Data ---")
    _, _, X_test_list, y_test_labels, class_names_test = process_and_load_data(
        test_meta, sim, reservoir_state, dt, recorded_voice, desc="TEST"
        )

    if len(Y_train_all) == 0:
        print("Error: No training data.")
        return
    

    # 5. Train & Evaluate
    # print("\n--- Testing with Temporal Integration ---")
    train_correct = 0
    test_correct = 0
    conf_matrix = np.zeros((2, 2), dtype=int)
    print("\n--- Training Readout ---")
    W_out = train_readout(X_train_all, Y_train_all, lambda_reg=1.0)


    for i, x_trial in enumerate(X_train_list):
        pred = predict(W_out, x_trial)
        true_label = y_train_labels[i]
        if pred == true_label:
            train_correct += 1
    for i, x_trial in enumerate(X_test_list):
        pred = predict(W_out, x_trial)
        true_label = y_test_labels[i]
        
        conf_matrix[true_label, pred] += 1
        if pred == true_label:
            test_correct += 1

    acc_train = train_correct / len(y_train_labels)
    acc_test = test_correct / len(y_test_labels)
    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy: {acc_test * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    class_names = get_class_names(args.task)
    num_classes = len(class_names)

    print("--- Training Readout (Shuffle Control) ---")
    y_train_labels_shuffle = list(y_train_labels)
    rng.shuffle(y_train_labels_shuffle)
    Y_train_shuffle_list = []

    for i, x_feat in enumerate(X_train_list):
        n_steps = x_feat.shape[0]
        shuffled_label = y_train_labels_shuffle[i]
        target_shuffled = create_target_signal(n_steps, shuffled_label, num_classes, dt)
        Y_train_shuffle_list.append(target_shuffled)
        
    Y_train_shuffle_all = np.vstack(Y_train_shuffle_list)
    
    W_out_shuffle = train_readout(X_train_all, Y_train_shuffle_all, lambda_reg=1.0)

    test_correct_shuffle = 0
    for i, x_trial in enumerate(X_test_list):
        pred = predict(W_out_shuffle, x_trial)
        true_label = y_test_labels[i]
        if pred == true_label:
            test_correct_shuffle += 1
            
    acc_test_shuffle = test_correct_shuffle / len(y_test_labels)
    print(f"Test Accuracy (Shuffle): {acc_test_shuffle * 100:.2f}%")
    

    # ====================================
    # 6. Save Results & Analysis
    # ====================================

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
    
    filename = f"conf.png"
    plt.savefig(os.path.join(RESULT_DIR, "figs", filename))
    plt.close()
    print(f"Saved confusion matrix to {os.path.join(RESULT_DIR, 'figs', filename)}")

    dt = cfg.INPUT_DT_COCH if args.mode == "linear" else cfg.DT
    X_full_list = X_train_list + X_test_list
    y_full_labels = y_train_labels + y_test_labels
    full_class_names = class_names_train + class_names_test
    analyze_trajectories(X_full_list, y_full_labels, full_class_names, RESULT_DIR, dt, args.task)
    # del X_train_list, X_test_list
    # gc.collect()


    # Save Weights
    weight_dir = "audio_rc/reservoir_outputs"
    os.makedirs(weight_dir, exist_ok=True)
    np.save(os.path.join(weight_dir, f"W_out.npy"), W_out)
    np.save(os.path.join("audio_rc/result/data", f"W_out.npy"), W_out)
    print(f"Saved weights to W_out.npy")

    # if misclassified:
    #     print(f"\n{len(misclassified)} Misclassified Samples (First 5):")
        # for m in misclassified[:5]:
        #     print(f"  Path: {os.path.basename(m[0])}, True: {class_names[m[1]]}, Pred: {class_names[m[2]]}")

if __name__ == "__main__":
    main()