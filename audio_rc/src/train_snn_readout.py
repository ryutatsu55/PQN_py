import json
import numpy as np
import glob
import argparse
from datetime import datetime
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

def load_all_data():
    """
    すべてのカテゴリのデータを読み込み、メタデータ付きのリストとして返す。
    Returns:
        all_data (list of dict): Each dict contains 'feat', 'gender', 'digit', 'path', 'split'
    """
    base_dirs = {
        "train": "audio_rc/reservoir_inputs/train",
        "test": "audio_rc/reservoir_inputs/test"
    }
    output_base_dir = "audio_rc/reservoir_outputs"
    
    # カテゴリ定義 (ディレクトリ名 -> 属性)
    categories = [
        {"dir": "coch_female_zero", "gender": "female", "digit": 0},
        {"dir": "coch_female_one",  "gender": "female", "digit": 1},
        {"dir": "coch_male_zero",   "gender": "male",   "digit": 0},
        {"dir": "coch_male_one",    "gender": "male",   "digit": 1},
    ]

    all_data = []

    # Initialize Reservoir (only for SNN mode)
    sim = None
    reservoir_state = None
    if args.mode == "snn":
        reservoir_state = config.init_reservoir()
        sim = PQN_RNN_onGPU.PQN_Reservoir_GPU(reservoir_state, cfg)

    for split, base_dir in base_dirs.items():
        for cat in categories:
            subdir = cat["dir"]
            gender = cat["gender"]
            digit = cat["digit"]
            
            # パスの取得
            if args.mode == "snn":
                search_path = os.path.join(base_dir, subdir, "*.npy")
            elif args.mode == "linear":
                search_path = os.path.join(base_dir, subdir, "*.npy")
            elif args.mode == "feature":
                # featureモードの場合、保存先ディレクトリから読み込む
                # 例: features_female_zero
                feat_subdir = f"features_{subdir.replace('coch_', '')}" 
                search_path = os.path.join(output_base_dir, split, feat_subdir, "*.npy")
            
            files = glob.glob(search_path)
            
            for path in tqdm(files, desc=f"Loading {split} {gender} {digit}"):
                # データのロードまたは計算
                if args.mode == "snn":
                    coch = np.load(path)
                    feat = PQN_RNN_onGPU.main(
                        input_data=coch,            #TODO check distribution of value
                        coch=True,
                        reservoir_state=reservoir_state,
                        return_feature=True,
                        is_debug_print=False,
                        record=False,
                        S_durt=cfg.INPUT_DT_COCH,        #TODO
                        cfg=cfg,
                        sim=sim
                    )
                else:
                    feat = np.load(path) # feature or linear
                
                # 特徴量の前処理 (時間積分)
                feat_flat = pad_and_integrate(feat)     #TODO  このままじゃPCAできない
                
                all_data.append({
                    "feat": feat_flat,
                    "gender": gender,
                    "digit": digit,
                    "split": split, # original split ('train' or 'test')
                    "path": path
                })
    
    return all_data

def prepare_task_data(all_data, task):
    """
    タスクに応じてトレーニングデータとテストデータを振り分け、ラベルを付与する。
    """
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_test_paths = []

    for item in all_data:
        feat = item["feat"]
        gender = item["gender"]
        digit = item["digit"]
        original_split = item["split"]
        
        is_train = False
        is_test = False
        label = -1

        # --- Task Logic ---
        if task == "digit":
            # 0 vs 1 (All genders)
            # Train/Test split follows the original dataset split
            label = digit
            if original_split == "train": is_train = True
            elif original_split == "test": is_test = True

        elif task == "gender":
            # Female(0) vs Male(1) (All digits)
            label = 0 if gender == "female" else 1
            if original_split == "train": is_train = True
            elif original_split == "test": is_test = True

        elif task == "m2f_digit":
            # Train: Male (Digit 0/1), Test: Female (Digit 0/1)
            # Use ALL male data for training, ALL female data for testing
            if gender == "male" and original_split == "train":
                is_train = True
                label = digit
            elif gender == "female" and original_split == "test":
                is_test = True
                label = digit

        elif task == "f2m_digit":
            # Train: Female, Test: Male
            if gender == "female" and original_split == "train":
                is_train = True
                label = digit
            elif gender == "male" and original_split == "test":
                is_test = True
                label = digit

        elif task == "z2o_gender":
            # Train: Digit 0 (Gender F/M), Test: Digit 1 (Gender F/M)
            # Label: Female(0), Male(1)
            if digit == 0 and original_split == "train":
                is_train = True
                label = 0 if gender == "female" else 1
            elif digit == 1 and original_split == "test":
                is_test = True
                label = 0 if gender == "female" else 1

        elif task == "o2z_gender":
            # Train: Digit 1, Test: Digit 0
            if digit == 1:
                is_train = True
                label = 0 if gender == "female" else 1
            elif digit == 0:
                is_test = True
                label = 0 if gender == "female" else 1

        # --- Append Data ---
        if is_train:
            X_train.append(feat)
            y_train.append(label)
        elif is_test:
            X_test.append(feat)
            y_test.append(label)
            X_test_paths.append(item["path"])

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test),
        X_test_paths
    )

# ================================
# 3. Training & Evaluation
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
# 4. Main Process
# ================================

def main():
    print(f"Mode: {args.mode}")
    print(f"Task: {args.task}")
    print(f"Reservoir Cells: {cfg.N}")

    # 1. Load Data
    print("\n--- Loading Data ---")
    all_data_list = load_all_data()
    
    # 2. Prepare Task Data
    print(f"\n--- Preparing Data for Task: {args.task} ---")
    X_train, y_train, X_test, y_test, X_test_paths = prepare_task_data(all_data_list, args.task)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    if len(X_train) == 0:
        print("Error: No training data available.")
        return

    # 3. Train Readout
    print("\n--- Training Readout ---")
    # Ridge regression with regularization
    W_out = train_readout(X_train, y_train, lambda_reg=1e-2)

    # 4. Evaluate
    acc_train = evaluate(W_out, X_train, y_train)
    acc_test = evaluate(W_out, X_test, y_test)

    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%")

    # 5. Confusion Matrix & Analysis
    class_names = get_class_names(args.task)
    num_classes = len(class_names)
    
    preds_test = predict(W_out, X_test)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    misclassified = []
    for i in range(len(y_test)):
        true_l = y_test[i]
        pred_l = preds_test[i]
        conf_matrix[true_l, pred_l] += 1
        if true_l != pred_l:
            misclassified.append((X_test_paths[i], true_l, pred_l))

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
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black", fontsize=14)
            
    plt.colorbar()
    plt.tight_layout()
    
    save_fig_dir = "audio_rc/figs/confusion_matrix"
    os.makedirs(save_fig_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"conf_{args.task}_{timestamp}.png"
    plt.savefig(os.path.join(save_fig_dir, filename))
    plt.close()
    print(f"Saved confusion matrix to {os.path.join(save_fig_dir, filename)}")

    # 6. Save Results
    result_entry = {
        "timestamp": timestamp,
        "mode": args.mode,
        "task": args.task,
        "seed": cfg.SEED,
        "cells": cfg.N,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "misclassified_count": len(misclassified)
    }
    
    results_dir = "audio_rc/result"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results_snn_readout.jsonl"), "a") as f:
        f.write(json.dumps(result_entry) + "\n")

    # Save Weights
    weight_dir = "audio_rc/reservoir_outputs"
    os.makedirs(weight_dir, exist_ok=True)
    np.save(os.path.join(weight_dir, f"W_out_{args.task}.npy"), W_out)
    print(f"Saved weights to W_out_{args.task}.npy")

    if misclassified:
        print(f"\n{len(misclassified)} Misclassified Samples (First 5):")
        for m in misclassified[:5]:
            print(f"  Path: {os.path.basename(m[0])}, True: {class_names[m[1]]}, Pred: {class_names[m[2]]}")

if __name__ == "__main__":
    main()