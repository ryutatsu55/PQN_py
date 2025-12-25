import subprocess
import os
import shutil
from datetime import datetime
import sys

# ここで実験パラメータを一時的に書き換えて実行することも可能ですが、
# 基本は RNN_config.py を編集してからこれを実行する運用を想定します。

def run_experiment(exp_name: str):
    # 1. 実験ID (タイムスタンプ) の生成
    datestamp = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%H%M%S")
    if exp_name == "":
        archive_dir = os.path.join("RNN_analyze", "archive", f"{datestamp}/{timestamp}")
    else:
        archive_dir = os.path.join("RNN_analyze", "archive", f"{datestamp}/{timestamp}_{exp_name}")
    os.makedirs(archive_dir, exist_ok=True)

    print(f"=== Starting Experiment: {timestamp}_{exp_name} ===")
    print(f"Results will be saved to: {archive_dir}")

    # 2. 現在の条件（RNN_config.py）を結果フォルダにコピー保存
    # これにより、後から「このグラフの時のパラメータは何だっけ？」を確認できます。
    config_src = os.path.join("RNN_analyze", "RNN_config.py") # パスは環境に合わせて調整
    shutil.copy(config_src, os.path.join(archive_dir, "RNN_config_snapshot.py"))

    try:
        # 3. データ生成 (make_spatial_input.py)
        print("\n--- Step 1: Generating Input Data ---")
        subprocess.run(["python", "make_spatial_input.py"], check=True)

        # 4. 空間認識タスク (spatial_recognition.py)
        # SNNモードで特徴抽出まで行う
        print("\n--- Step 2: Spatial Recognition (SNN Mode) ---")
        # seedはConfigのデフォルトを使うなら指定なしでも良いが、明示的に指定も可
        subprocess.run([
            "python", "RNN_analyze/recognition_test.py",
            "--mode", "snn",
            "--classifier", "both" 
        ], check=True)

        # # 5. 時空間認識タスク (spatiotemp_recognition.py)
        # # featureモードで、さっき作ったデータを使って解析
        # print("\n--- Step 3: Spatiotemporal Recognition (Feature Mode) ---")
        # subprocess.run([
        #     "python", "RNN_analyze/spatiotemp_recognition.py",
        #     "--mode", "feature",
        #     "--seed", "100"
        # ], check=True)

        # 6. 生成された図やデータを結果フォルダに集約
        # (各スクリプトが RNN_analyze/data や figs に保存しているものを移動またはコピー)
        
        # 例: resultフォルダの中身をコピー
        result_src = os.path.join("RNN_analyze", "result")
        if os.path.exists(result_src):
            shutil.copytree(result_src, os.path.join(archive_dir), dirs_exist_ok=True)

        print(f"\n=== Experiment Completed Successfully ===")
        print(f"All results and config saved in: {archive_dir}")

    except subprocess.CalledProcessError as e:
        print(f"\n!!! Experiment Failed at step: {e.cmd} !!!")
        print(e)

if __name__ == "__main__":
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "")