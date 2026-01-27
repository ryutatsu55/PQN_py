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

cfg = config.Config

# --- ディレクトリパス設定 ---
BASE_DIR = "RNN_analyze"
INPUT_DIR = os.path.join(BASE_DIR, "reservoir_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "reservoir_outputs")
RESULT_DIR = os.path.join(BASE_DIR, "result")

def analyze():
    print(">>> Initializing Reservoir and Simulator...")
    reservoir_state = config.init_reservoir()
    sim = PQN_RNN.PQN_Reservoir_GPU(reservoir_state, cfg)
    
    dt = cfg.DT
    duration = 30.0       # 解析する秒数 (相関を見るには長いほうが安定します: 20~30秒推奨)
    warmup = 2.0          # 最初の過渡応答を捨てる秒数
    
    total_steps = int(duration / dt)
    warmup_steps = int(warmup / dt)
    N = cfg.N
    
    print(f">>> Starting Simulation for {duration}s (Warmup: {warmup}s)...")
    print(f"    Total steps: {total_steps}")


    print("    Warming up network state...")
    sim.run(num_steps=warmup_steps, record=False)


    print("    Recording spontaneous activity...")
    record={"result_dir": RESULT_DIR, "filename": "spontaneous_activity"}
    result = sim.run(num_steps=total_steps, record=record)
    sim.plot_results(result, 0, record)
        
    raw_spikes = result["rasters"]

    smoothing_window_ms = 100.0  # 100 ms smoothing window
    target_fs = 20.0             # 20 Hz sampling rate (50 ms interval)
    
    window_steps = int((smoothing_window_ms / 1000.0) / dt) # 窓のステップ数
    downsample_steps = int((1.0 / target_fs) / dt)          # 間引きステップ数
    
    # --- Smoothing (移動平均) ---
    # 畳み込みを使って平滑化する (Firing Rateへの変換)
    window = np.ones(window_steps) / (window_steps * dt) # 単位を [Hz] にする場合
    # 相関を見るだけなら絶対値は問わないので単純平均でOK
    # window = np.ones(window_steps) / window_steps
    
    n_steps, n_neurons = raw_spikes.shape
    smoothed_activity = np.zeros((n_steps, n_neurons), dtype=np.float32)
    for i in tqdm(range(n_neurons), desc="Smoothing"):
        smoothed_activity[:, i] = np.convolve(raw_spikes[:, i], window, mode='same')

    # --- Downsampling ---
    activity_downsampled = smoothed_activity[::downsample_steps, :]
    print(f"    Processed Data Shape: {activity_downsampled.shape}")


    print(">>> Calculating Correlation Matrix...")

    correlation_matrix = np.corrcoef(activity_downsampled.T)
    correlation_matrix = np.nan_to_num(correlation_matrix)
    W_corr = np.abs(correlation_matrix)

    # 式[7]によるモジュール性Qの計算
    Q_val = calculate_weighted_modularity_formula7(W_corr, n_modules=4)
    # 平均相関係数（対角成分を除く）
    mean_corr = np.mean(W_corr[~np.eye(cfg.N, dtype=bool)])
    print(f"    Mean Absolute Correlation: {mean_corr:.4f}")
    print(f"    Weighted Modularity Q:     {Q_val:.4f}")

    # ==========================================
    # 5. 結果の可視化
    # ==========================================
    # プロット
    plt.figure(figsize=(8, 6))
    sns.heatmap(W_corr, cmap="viridis", center=0.5, vmin=0, vmax=1, cbar=True, square=True)
    plt.title(f"Functional Connectivity (Correlation)\n Q = {Q_val:.4f}")
    plt.xlabel("Neuron ID")
    plt.ylabel("Neuron ID")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/figs/correlation_matrix.png")
    plt.close()
    print("Plot saved to: correlation_matrix.png")
    np.save(f"{RESULT_DIR}/data/correlation_matrix.npy", W_corr)
    print("Saved correlation_matrix.npy")

    # 重み行列 (構造)
    plt.figure(figsize=(8, 6))
    W = reservoir_state["reservoir_weight"]
    max_w = np.max(np.abs(W))
    sns.heatmap(W, cmap="plasma", center=0, cbar=True, square=True)           # cmap="vlag"/"plasma"
    plt.title("Structural Connectivity (Weights)")
    plt.xlabel("Neuron From")
    plt.ylabel("Neuron To")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/figs/weight_matrix.png")
    plt.close()
    print("Plot saved to: weight_matrix.png")
    np.save(f"{RESULT_DIR}/data/weight_matrix.npy", W)
    print("Saved weight_matrix.npy")
    
    # filename = "activity_trace"
    # # オマケ: 最初の数ニューロンの活動時系列を表示
    # plt.figure(figsize=(12, 4))
    # time_axis = np.arange(total_steps) * dt
    # # 最初の5個のニューロンだけ表示
    # for i in range(5):
    #     plt.plot(time_axis, activity_log[:, i], label=f"Neuron {i}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Synaptic Input Current (arb.)")
    # plt.title("Sample Activity Traces")
    # plt.legend(loc='upper right')
    # plt.xlim(0, 30.0) # 最初の1秒だけ拡大
    # plt.tight_layout()
    # plt.savefig(f"{RESULT_DIR}/figs/{filename}.png")

    # # --- データの保存 ---
    # # 保存先のディレクトリを作成
    # save_data_dir = os.path.join(RESULT_DIR, "data")
    # os.makedirs(save_data_dir, exist_ok=True)
    # #   (TimeSteps, 6)  col 0: Time, col 1: Neuron 0, col 2: Neuron 1  ...
    # data_to_save = np.column_stack((time_axis, activity_log[:, :5]))
    # save_path = os.path.join(save_data_dir, f"{filename}.npy")
    # np.save(save_path, data_to_save)

def calculate_weighted_modularity_formula7(correlation_matrix, n_modules=4):
    """
    論文の式[7]（Newmanの重み付きモジュール性）に基づくQ値の計算
    
    Q = (1 / 2M) * sum_ij [ (r_ij - (k_i * k_j) / 2M) * delta(m_i, m_j) ]
    
    Args:
        correlation_matrix (np.ndarray): (N, N) 重み行列として扱う相関行列 r_ij (通常は絶対値 |r_ij| を使用)
        n_modules (int): モジュール数 (4)
        
    Returns:
        Q (float): モジュール性
    """
    N = cfg.N
    
    # 対角成分（自己相関）は通常モジュール性計算から除外（0にする）
    W = correlation_matrix.copy()
    np.fill_diagonal(W, 0)
    
    # 1. 各項の計算
    # k_i = sum_j W_ij (強度, Weighted Degree)
    k = np.sum(W, axis=1)
    
    # M = (1/2) * sum_ij W_ij (全重みの半分)
    M = np.sum(k) / 2.0
    
    if M == 0:
        return 0.0

    # 2. モジュール割り当て m_i (ID順に等分割)
    # [0,0,..,0, 1,1,..,1, 2,..,2, 3,..,3]
    module_size = N // n_modules
    communities = np.zeros(N, dtype=int)
    for i in range(n_modules):
        communities[i*module_size : (i+1)*module_size] = i
        
    # 3. デルタ関数行列 delta(m_i, m_j)
    # 同じモジュールなら1, 違うなら0
    delta = (communities[:, None] == communities[None, :]).astype(float)
    
    # 4. Qの計算 (行列表現で高速化)
    # P_ij = (k_i * k_j) / 2M
    P = np.outer(k, k) / (2 * M)
    
    # Q = (1/2M) * sum( (W - P) * delta )
    term = (W - P) * delta
    Q = np.sum(term) / (2 * M)
    
    return Q

if __name__ == "__main__":
    analyze()