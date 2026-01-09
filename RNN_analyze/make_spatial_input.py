import numpy as np
import random
import os
import shutil
import RNN_config

cfg = RNN_config.Config
rng = np.random.RandomState(cfg.SEED)

def make_data():
    print(f"Generating data with N={cfg.N}, Strength={cfg.INPUT_STRENGTH}...")
    
    target_dir = cfg.INPUT_DIR
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
    categories = ["top", "middle", "bottom"]
    phases = ["train", "test"]
    for phase in phases:
        for cat in categories:
            os.makedirs(os.path.join(cfg.INPUT_DIR, phase, cat), exist_ok=True)

    # データ生成ループ (Train)
    for i in range(cfg.N_TRAIN):
        input_data = np.zeros((entire_steps, int(N//2)), dtype=float)
        target = rng.randint(0, 3)
        
        # Configの強度を使用
        input_data[:stim_steps, in_neurons[target]] = cfg.INPUT_STRENGTH
        
        cat = categories[target]
        save_path = os.path.join(cfg.INPUT_DIR, "train", cat, f"{i}.npy")
        np.save(save_path, input_data)

    # データ生成ループ (Test)
    for i in range(cfg.N_TEST):
        input_data = np.zeros((entire_steps, int(N//2)), dtype=float)
        target = rng.randint(0, 3)
        input_data[:stim_steps, in_neurons[target]] = cfg.INPUT_STRENGTH
        
        cat = categories[target]
        save_path = os.path.join(cfg.INPUT_DIR, "test", cat, f"{i}.npy")
        np.save(save_path, input_data)

if __name__ == "__main__":
    # RNN_config.set_global_seed(cfg.SEED)
    make_data()
