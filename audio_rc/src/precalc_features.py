import numpy as np
import glob
import argparse
import os
import shutil
from tqdm import tqdm
import sys
from pathlib import Path

# プロジェクトルートへのパス設定
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
import config
import src.PQN_RNN_onGPU as PQN_RNN_onGPU

# Load Config
cfg = config.Config

# --- ディレクトリパス設定 ---
BASE_DIR = "audio_rc"
INPUT_DIR = os.path.join(BASE_DIR, "reservoir_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "reservoir_outputs")
RESULT_DIR = os.path.join(BASE_DIR, "result")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=cfg.SEED, help="random seed")
args = parser.parse_args()

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

def collect_file_metadata():
    """全入力ファイルのメタデータを収集"""
    base_dirs = {
        "train": os.path.join(INPUT_DIR, "train"),
        "test": os.path.join(INPUT_DIR, "test")
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
                    "split": split, # original split ('train' or 'test')
                    "subdir_name": subdir
                })
    return metadata_list

def select_required_files(metadata_list, rng):
    """必要な5クラスのみを定数個選出する"""
    candidates_train = []
    candidates_test = []
    
    # 1. 必要なクラスの定義 (条件フィルター)
    def is_required(item):
        g, d, r = item["gender"], item["digit"], item["reverse"]
        # Forward: Female/Male, 0/1 すべて必要
        if not r: return True
        # Reverse: Female 0 のみ必要 (reverse_f_z タスク用)
        if r and g == "female" and d == 0: return True
        return False

    # 2. フィルタリング
    for item in metadata_list:
        if is_required(item):
            if item["split"] == "train":
                candidates_train.append(item)
            elif item["split"] == "test":
                candidates_test.append(item)
    # rng.shuffle(candidates) # シードに基づいてシャッフル

    # 3. クラスごとの定数選出
    # train/test それぞれで定員を設ける
    n_train_target = getattr(cfg, "N_TRAIN_COCH", 0)
    n_test_target = getattr(cfg, "N_TEST_COCH", 0)
    
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

    # Configから回数を取得
    n_train_target = getattr(cfg, "N_TRAIN_COCH", 20)
    n_test_target = getattr(cfg, "N_TEST_COCH", 10)

    train_items = expand_representative_files(candidates_train, n_train_target)
    test_items = expand_representative_files(candidates_test, n_test_target)
            
    return train_items, test_items

def main():
    print(f"--- Pre-calculating Features (SEED: {args.seed}) ---")

    # 1. 保存先ディレクトリのクリーンアップ 
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
        print(f"deleted following directory: {RESULT_DIR} ( preparing for result )")
    os.makedirs(os.path.join(RESULT_DIR, "figs"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "data"), exist_ok=True)

    # 2. リザバーの初期化
    print("Initializing Reservoir...")
    reservoir_state = config.init_reservoir(args.seed)
    sim = PQN_RNN_onGPU.PQN_Reservoir_GPU(reservoir_state, cfg)

    # 3. ファイル選択
    rng = np.random.RandomState(args.seed)
    all_metadata = collect_file_metadata()
    train_items, test_items = select_required_files(all_metadata, rng)
    
    print(f"train files to process: {len(train_items)}")
    print(f"test files to process: {len(test_items)}")
    
    # 4. 処理実行 (SNNシミュレーション -> 保存)
    recorded_voice = set()
    target_items = train_items + test_items
    
    for item in tqdm(target_items, desc="Extracting Features"):
        input_path = item["input_path"]
        rep_idx = item["rep_idx"]
        subdir_name = item["subdir_name"]
        split = item["split"]
        # print(input_path, rep_idx, subdir_name, split)
        
        # クラス名生成 (デバッグ/録音用)
        class_name = f"{item['gender']}{item['digit']}"
        if item['reverse']: class_name = f"rev_{class_name}"

        # 保存パス決定
        save_path, save_dir = get_feature_save_path(input_path, split, subdir_name, rep_idx)
        os.makedirs(save_dir, exist_ok=True)

        # 録音設定 (各クラス最初の1つだけ記録する)
        current_record = None
        # 必要なら以下を有効化
        if class_name not in recorded_voice and split == "train":
            current_record = {
                "result_dir": RESULT_DIR,
                "filename": class_name,
            }
            # print("Recording class:", class_name)
            recorded_voice.add(class_name)

        # 時間設定
        dt = cfg.DT
        tmax = None
        n_steps = None
        if split == "train":
            if hasattr(cfg, "DURATION_INTERVAL_COCH"):
                tmax = cfg.DURATION_INTERVAL_COCH
                n_steps = int(tmax / dt) 
        else:
            if hasattr(cfg, "TEACHING_DURATION"):
                tmax = cfg.TEACHING_DURATION
                n_steps = int(tmax / dt)
        
        # データロード & SNN実行
        coch = np.load(input_path)
        
        # PQN_RNN_onGPU.main を呼び出して特徴量を取得
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
        
        # 保存
        np.save(save_path, feat)

    print("--- Pre-calculation Complete ---")

if __name__ == "__main__":
    main()