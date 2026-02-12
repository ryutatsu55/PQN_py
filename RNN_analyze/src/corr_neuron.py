import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from tqdm import tqdm
import seaborn as sns
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, default=cfg.SEED,
    help="random seed"
)
args = parser.parse_args()
rng = np.random.RandomState(args.seed)

def main():
    print(">>> Initializing Reservoir and Simulator...")
    reservoir_state = config.init_reservoir(args.seed)
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
    with np.errstate(invalid='ignore'):
        correlation_matrix = np.corrcoef(activity_downsampled.T)
    correlation_matrix = np.nan_to_num(correlation_matrix)
    W_corr = np.abs(correlation_matrix)

    # 式[7]によるモジュール性Qの計算
    Q_val = calculate_weighted_modularity(W_corr, n_modules=4)
    # 平均相関係数（対角成分を除く）
    mean_corr = np.mean(W_corr[~np.eye(cfg.N, dtype=bool)])
    print(f"    Mean Absolute Correlation (Activity): {mean_corr:.4f}")
    print(f"    Weighted Modularity Q (Activity):     {Q_val:.4f}")
    
    weight_matrix = reservoir_state["reservoir_weight"]
    # 1. 重みの絶対値をとる (相関の強さに対応させるため)
    W_abs = np.abs(weight_matrix)
    # 2. 対称化する (相関行列は対称行列であるため、形式を合わせる)
    # 双方向の結合の平均強度をエッジの重みとする
    W_struct = (W_abs + W_abs.T) / 2.0
    # 3. 既存のQ計算ロジックを流用
    # (calculate_weighted_modularity内で対角成分除去や正規化は行われるためそのまま渡せます)
    Q_struct = calculate_weighted_modularity(W_struct, n_modules=4)
    mean_corr = np.mean(W_struct[~np.eye(cfg.N, dtype=bool)])
    print(f"    Mean Absolute Correlation (Structure): {mean_corr:.4f}")
    print(f"    Weighted Modularity Q (Structure):     {Q_struct:.4f}")
    

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
    W = reservoir_state["reservoir_weight"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(W, cmap="plasma", center=0, cbar=True, square=True)           # cmap="vlag"/"plasma"
    plt.title(f"Structural Connectivity (Weights)\n Q = {Q_struct: .4f}")
    plt.xlabel("Neuron From")
    plt.ylabel("Neuron To")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/figs/weight_matrix.png")
    plt.close()
    print("Plot saved to: weight_matrix.png")
    np.save(f"{RESULT_DIR}/data/weight_matrix.npy", W)
    print("Saved weight_matrix.npy")
    
    visualize_reservoir_structure(W, n_modules=4)
    

def calculate_weighted_modularity(correlation_matrix, n_modules=4):
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

def visualize_reservoir_structure(weight_matrix, n_modules=4):
    N = cfg.N
    module_size = N // n_modules
    G = nx.DiGraph()

    # ノードの追加
    for i in range(N):
        # モジュールID (0, 1, 2, 3)
        module_id = i // module_size
        
        # 抑制性ニューロンかどうか判定
        is_excited = np.any(weight_matrix[:, i] > 0)
        
        G.add_node(i, module=module_id, color='red' if is_excited else 'blue')

    # エッジの追加
    rows, cols = np.where(weight_matrix != 0)
    for r, c in zip(rows, cols):
        weight = weight_matrix[r, c] # c(From) -> r(To) への結合を表す
        
        # NetworkX は G.add_edge(u, v) で u -> v
        G.add_edge(c, r, weight=abs(weight), color='red' if weight > 0 else 'blue')

    # 3. レイアウト設定 (ここが重要: モジュールごとに固める)
    pos = {}
    
    # 4つのモジュールの中心座標 (Fig 4Aのように四角形に配置)
    module_centers = {
        0: np.array([-1, 1]),  # 左上
        1: np.array([-1, -1]),   # 左下
        2: np.array([1, -1]),  # 右下
        3: np.array([1, 1])  # 右上
    }
    
    for mod_id in range(n_modules):
        # このモジュールに所属するノードリスト
        nodelist = [n for n in range(N) if G.nodes[n]["module"] == mod_id]
        
        # NetworkXのcircular_layoutを使って円周座標を取得 (中心は0,0)
        # scaleは円の半径
        sub_pos = nx.circular_layout(nodelist, scale=0.4)
        
        # モジュールの中心座標へずらす
        center = module_centers[mod_id]
        for n in nodelist:
            pos[n] = sub_pos[n] + center

    # 4. 描画
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # ノード描画
    
    nodes = G.nodes(data=True)
    for mod_id in range(n_modules):
        nodelist = [n for n, d in nodes if d['module'] == mod_id]
        node_colors = [d['color'] for n, d in nodes if d['module'] == mod_id]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=nodelist, 
            node_color=node_colors,
            node_size=300, 
            edgecolors='black', # 枠線
            label=f"Module {mod_id}"
        )
        
    # ニューロン番号ラベル (必要なら)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    # エッジ描画
    # 興奮性 (Exc) -> 青, 抑制性 (Inh) -> 赤
    edges = G.edges(data=True)
    edgelist = [(u, v) for u, v, d in edges]
    edge_colors = [d['color'] for u, v, d in edges]
    edge_weights = [d['weight'] for u, v, d in edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=edgelist, 
        edge_color=edge_colors, 
        alpha=0.3, 
        width=edge_weights, 
        arrows=True, arrowsize=10, 
        connectionstyle="arc3,rad=0.1" # 少しカーブさせる
    )

    plt.title(f"mBNN Reservoir Structure (N={N}, 4 Modules)", fontsize=16)
    plt.axis('off') # 軸を消す
    
    # 凡例用のダミー
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Excitatory'),
        Line2D([0], [0], color='blue', lw=2, label='Inhibitory'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/figs/network.png", dpi=300)
    plt.close()
    print("Network graph saved to: network.png")

if __name__ == "__main__":
    os.makedirs(os.path.join(RESULT_DIR, "figs"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "data"), exist_ok=True)
    main()
