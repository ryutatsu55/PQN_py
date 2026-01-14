import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import RNN_config

def analyze_correlation():
    # 1. 設定の読み込みとリザバーの初期化
    print("Initializing reservoir based on RNN_config...")
    cfg = RNN_config.Config
    # シード固定（再現性のため）
    RNN_config.set_global_seed(cfg.SEED)
    
    # ネットワーク構築
    # init_reservoir() は辞書を返します: {"reservoir_weight": ..., "mask": ..., ...}
    reservoir_params = RNN_config.init_reservoir()
    W = reservoir_params["reservoir_weight"]
    N = cfg.N
    
    print(f"Network size: {N} neurons")
    print(f"Modularity structure should be visible in weights.")

    # 2. 自発活動のシミュレーション (Spontaneous Activity Simulation)
    # 構造的特徴（モジュール性）が機能的結合（相関）にどう反映されるかを確認するため、
    # 簡易的な非線形ダイナミクス(tanh)で自発活動を生成します。
    
    print("Simulating spontaneous activity...")
    steps = 3000
    transient = 500  # 初期過渡期（捨てる分）
    dt = 0.5         # 漏れ率相当
    noise_level = 0.1
    
    # 状態ベクトルの初期化
    x = np.random.randn(N)
    activity_log = []

    for t in range(steps):
        # x(t+1) = (1-dt)x(t) + dt * tanh( W*x(t) + noise )
        # 入力がない状態（自発活動）でのダイナミクス
        noise = np.random.randn(N) * noise_level
        recurrence = np.dot(W, x)
        x = (1 - dt) * x + dt * np.tanh(recurrence + noise)
        
        if t >= transient:
            activity_log.append(x.copy())

    # 形状: (Time, Neurons)
    activity_data = np.array(activity_log)

    # 3. 相関行列の計算 (Correlation Matrix)
    # 各ニューロン(列)間のピアソン相関係数を計算
    # rowvar=False にすると、列(ニューロン)を変数として計算します
    correlation_matrix = np.corrcoef(activity_data, rowvar=False)
    
    # NaN処理（活動がないニューロンがいる場合など）
    correlation_matrix = np.nan_to_num(correlation_matrix)

    # 4. 結果の表示と可視化
    print("\n--- Analysis Results ---")
    print(f"Correlation Matrix Shape: {correlation_matrix.shape}")
    
    # 全成分の平均相関係数（対角成分除く）
    mask_diag = ~np.eye(correlation_matrix.shape[0], dtype=bool)
    mean_corr = np.mean(np.abs(correlation_matrix[mask_diag]))
    print(f"Mean Absolute Correlation (off-diagonal): {mean_corr:.4f}")

    # プロット
    plt.figure(figsize=(12, 5))

    # (A) 重み行列 (構造)
    plt.subplot(1, 2, 1)
    # 重みは疎行列なので、見やすくするために非ゼロ要素を強調あるいはバイナリで表示
    sns.heatmap(W, cmap="vlag", center=0, cbar=True, square=True)
    plt.title("Structural Connectivity (Weights)")
    plt.xlabel("Neuron From")
    plt.ylabel("Neuron To")

    # (B) 相関行列 (機能)
    plt.subplot(1, 2, 2)
    sns.heatmap(correlation_matrix, cmap="RdBu_r", center=0, vmin=-1, vmax=1, cbar=True, square=True)
    plt.title("Functional Connectivity (Correlation)")
    plt.xlabel("Neuron ID")
    plt.ylabel("Neuron ID")

    plt.tight_layout()
    filename = "correlation_analysis.png"
    plt.savefig(f"{cfg.RESULT_DIR}/figs/{filename}")
    plt.close()
    print("Saved plot to 'correlation_analysis.png'")

    
    np.save(f"{cfg.RESULT_DIR}/data/correlation_matrix.npy", correlation_matrix)
    print("Saved correlation_matrix.npy")

    return correlation_matrix

if __name__ == "__main__":
    analyze_correlation()