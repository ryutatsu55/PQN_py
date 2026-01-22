import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns

# 自作モジュールのインポート
import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
import config
import src.PQN_RNN_onGPU as PQN_RNN

# --- ディレクトリパス設定 ---
BASE_DIR = "RNN_analyze"
INPUT_DIR = os.path.join(BASE_DIR, "reservoir_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "reservoir_outputs")
RESULT_DIR = os.path.join(BASE_DIR, "result")

def analyze():
    # ==========================================
    # 1. 設定と初期化
    # ==========================================
    cfg = config.Config
    # シードを固定しないと毎回結果が変わります（必要に応じて固定）
    # RNN_config.set_global_seed(cfg.SEED) 
    
    print(">>> Initializing Reservoir and Simulator...")
    
    # ネットワーク構造の生成
    # (RNN_config.init_reservoir内で乱数が使われるので、構造もここで決まります)
    reservoir_state = config.init_reservoir()
    
    # GPUシミュレータのインスタンス化
    sim = PQN_RNN.PQN_Reservoir_GPU(reservoir_state, cfg)
    
    # ==========================================
    # 2. シミュレーション条件の設定
    # ==========================================
    dt = cfg.DT
    duration = 30.0       # 解析する秒数 (相関を見るには長いほうが安定します: 20~30秒推奨)
    warmup = 2.0          # 最初の過渡応答を捨てる秒数
    
    total_steps = int(duration / dt)
    warmup_steps = int(warmup / dt)
    N = cfg.N
    N_input = len(reservoir_state["input_indices"])
    
    # 自発活動 (Spontaneous Activity) のためのバックグラウンドノイズ
    # 全く入力がないと沈黙してしまう場合、わずかな確率でランダム発火させます
    spontaneous_freq = 0.5 # *10[Hz]( = 5Hz) (論文でも自発活動を記録しています)
    
    # 各ニューロンへの入力確率ベクトル
    prob_input = np.full((total_steps, N_input), spontaneous_freq * dt, dtype=np.float32)
    
    print(f">>> Starting Simulation for {duration}s (Warmup: {warmup}s)...")
    print(f"    Total steps: {total_steps}")

    # ==========================================
    # 3. ステップ実行 (Step-by-Step Simulation)
    # ==========================================
    
    # --- Phase 1: ウォームアップ (記録しない) ---
    print("    Warming up network state...")
    sim.run(num_steps=warmup_steps, record=False)
        
    # --- Phase 2: 本番計測 (手動でデータを取得) ---
    print("    Recording spontaneous activity...")
    result = sim.run(num_steps=total_steps, record=True)
    sim.plot_results(result, 0, cfg)
        
    # 3. リストに保存 (copyを忘れずに)
    activity_log = result["input"]

    
    # ==========================================
    # 4. 相関行列の計算
    # ==========================================
    print(">>> Calculating Correlation Matrix...")
    
    # np.corrcoef は (変数, 観測値) の形を期待するので転置します -> (Neuron, Time)
    # これで「ニューロンiとニューロンjの活動の類似度」が計算されます
    correlation_matrix = np.corrcoef(activity_log.T)
    # NaNが含まれる場合（全く活動しなかったニューロンなど）を0に置換
    correlation_matrix = np.nan_to_num(correlation_matrix)
    correlation_matrix = np.abs(correlation_matrix)
    
    # 平均相関係数（対角成分を除く）
    off_diag = correlation_matrix[~np.eye(N, dtype=bool)]
    mean_corr = np.mean(np.abs(off_diag))
    print(f"    Mean Absolute Correlation: {mean_corr:.4f}")

    # ==========================================
    # 5. 結果の可視化
    # ==========================================
    # プロット
    plt.figure(figsize=(12, 6))

    # (A) 重み行列 (構造)
    plt.subplot(1, 2, 1)
    W = reservoir_state["reservoir_weight"]
    max_w = np.max(np.abs(W))
    # 重みは疎行列なので、見やすくするために非ゼロ要素を強調あるいはバイナリで表示
    sns.heatmap(W, cmap="vlag", center=0, cbar=True, square=True)
    plt.title("Structural Connectivity (Weights)")
    plt.xlabel("Neuron From")
    plt.ylabel("Neuron To")

    # (B) 相関行列 (機能)
    plt.subplot(1, 2, 2)
    sns.heatmap(correlation_matrix, cmap="viridis", center=0.5, vmin=0, vmax=1, cbar=True, square=True)
    plt.title(f"Functional Connectivity (Correlation)\n{mean_corr:.4f}")
    plt.xlabel("Neuron ID")
    plt.ylabel("Neuron ID")

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/figs/correlation_matrix.png")
    plt.close()
    print("Plot saved to: correlation_matrix.png")

    
    np.save(f"{RESULT_DIR}/data/correlation_matrix.npy", correlation_matrix)
    print("Saved correlation_matrix.npy")
    
    
    
    
    # オマケ: 最初の数ニューロンの活動時系列を表示
    plt.figure(figsize=(12, 4))
    time_axis = np.arange(total_steps) * dt
    # 最初の5個のニューロンだけ表示
    for i in range(5):
        plt.plot(time_axis, activity_log[:, i], label=f"Neuron {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Synaptic Input Current (arb.)")
    plt.title("Sample Activity Traces")
    plt.legend(loc='upper right')
    plt.xlim(0, 1.0) # 最初の1秒だけ拡大
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/figs/activity_trace.png")

if __name__ == "__main__":
    analyze()