import numpy as np
import random
import os
import shutil
import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
import config
from tqdm import tqdm

cfg = config.Config
rng = np.random.RandomState(cfg.SEED)

# --- ディレクトリパス設定 ---
BASE_DIR = "RNN_analyze"
INPUT_DIR = os.path.join(BASE_DIR, "reservoir_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "reservoir_outputs")
RESULT_DIR = os.path.join(BASE_DIR, "result")

def make_data():
    print(f"Generating data with N={cfg.N}, Strength={cfg.INPUT_STRENGTH}...")
    
    phases = ["train", "test"]
    cat = "pulse"
    for phase in phases:
        target_dir = os.path.join(INPUT_DIR, phase, cat)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            print(f"ディレクトリ {target_dir} を削除しました。")

    # Configから計算
    dt = cfg.INPUT_DT
    N = cfg.N
    hoge = int(N//6)
    in_neurons = [np.arange(0, hoge), np.arange(hoge, 2*hoge), np.arange(2*hoge, 3*hoge)]  # 入力ニューロン群のインデックス
    
    stim_steps = int(cfg.DURATION_STIM / dt)
    entire_steps = int(cfg.DURATION_INTERVAL / dt)
    
    # フォルダ作成
    for phase in phases:
        os.makedirs(os.path.join(INPUT_DIR, phase, cat), exist_ok=True)

    # データ生成ループ (Train)
    for i in tqdm(range(cfg.N_TRAIN)):
        input_data = np.zeros((entire_steps, cfg.INPUT_NODES), dtype=float)
        target = rng.randint(0, 3)
        input_data[:stim_steps, in_neurons[target]] = cfg.INPUT_STRENGTH
        save_path = os.path.join(INPUT_DIR, "train", cat, f"pulse{i}.npy")
        np.save(save_path, input_data)

    # データ生成ループ (Test)
    for i in tqdm(range(cfg.N_TEST)):
        input_data = np.zeros((entire_steps, cfg.INPUT_NODES), dtype=float)
        target = rng.randint(0, 3)
        input_data[:stim_steps, in_neurons[target]] = cfg.INPUT_STRENGTH
        save_path = os.path.join(INPUT_DIR, "test", cat, f"pulse{i}.npy")
        np.save(save_path, input_data)

if __name__ == "__main__":
    # RNN_config.set_global_seed(cfg.SEED)
    make_data()
