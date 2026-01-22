import numpy as np
import random
import os

class Config:
    # --- 基本設定 ---
    SEED = 100           # ベースとなるシード値
    N = 240             # ニューロン総数
    DT = 0.0001
    INPUT_DT = 0.01           # タイムステップ [s] (シミュレーション用)
    SPATIO_TEMP_DT = 0.01    # タイムステップ [s] (時空間認識タスク用)
    # --- 入力データ生成設定(空間認識) ---
    DURATION_STIM = 0.1           # 刺激時間 [s]
    DURATION_INTERVAL = 1.5       # 1試行の長さ [s] (spatial task)
    INPUT_STRENGTH = 1.0          # 入力強度
    
    N_TRAIN = 20                 # 学習データ数
    N_TEST = 20                  # テストデータ数
    # 14 / 6 for spoken digit
    
    # --- リザバー結合パラメータ ---
    RESERVOIR_CONN = 0.06    # 結合強度係数 (元のコードの * 0.02)
    READOUT_NODES = 60            # 読み出し層のノード数

    SPONTANEOUS_FREQ = 0.1
    INPUT_FREQ = 10

def init_reservoir(seed=Config.SEED):
    N = Config.N
    seed = Config.SEED
    input_size = N // 2
    rng = np.random.RandomState(seed)
    resovoir_origin, mask, type = create_moduled_matrix(N, rng)
    resovoir_weight = np.copy(resovoir_origin) * Config.RESERVOIR_CONN
    N_S = np.count_nonzero(resovoir_weight)
    tau_rec_h, tau_inact_h, tau_faci_h, U1_h, U_h, mask_faci_h, tr, td = synapses_init(resovoir_weight, N, N_S)
    neuron_from_h, calc_matrix_h, neuron_to_h = calc_init(resovoir_weight, N, N_S)
    delayed_row_h = delay_init(resovoir_weight, N, N_S, mask, rng)

    input_indices = np.arange(input_size)
    # candidate_indices = np.arange(input_size, N)
    candidate_indices = np.arange(N)
    readout_num = Config.READOUT_NODES
    if len(candidate_indices) < readout_num:
        raise ValueError(f"num of neuron N={N} is too small")
    output_indices = rng.choice(candidate_indices, readout_num, replace=False)
    return {
        "N": N,
        "reservoir_weight": resovoir_origin,
        "mask": mask,
        "type": type,
        "N_S": N_S,
        "tr" : tr,
        "td" : td,
        "tau_rec_h": tau_rec_h,
        "tau_inact_h": tau_inact_h,
        "tau_faci_h": tau_faci_h,
        "U1_h": U1_h,
        "U_h": U_h,
        "mask_faci_h": mask_faci_h,
        "neuron_from_h": neuron_from_h,
        "calc_matrix_h": calc_matrix_h,
        "neuron_to_h": neuron_to_h,
        "delayed_row_h": delayed_row_h,
        "input_indices": input_indices,
        "output_indices": output_indices,
    }

def create_moduled_matrix(N, rng):
    resovoir_weight = np.zeros((N, N))
    block_size = N // 4
    crust_idx = 0
    G = 0.5
    p = 0.05
    offset = 1.0
    while crust_idx != 4:
        i1 = int(crust_idx * N / 4)
        i2 = int((crust_idx + 1) * N / 4)
        # resovoir_weight[i1:i2, i1:i2] = ((G * rng.randn(N // 4, N // 4)) + offset) * (
        #     rng.rand(N // 4, N // 4) < p
        # )
        # resovoir_weight[i1:i2, i1:i2] = (
        #     G*(rng.rand(N//4, N//4)-0.5) + offset
        #     ) * (rng.rand(N//4, N//4) < p)
        
        # 1. ニューロンごとに異なる「接続確率」を作成する
        # 対数正規分布を使って「ムラ」を作る (sigmaが大きいほどムラが激しくなる)
        # size=(1, block_size) にすることで「列（前ニューロン）」ごとに確率を変える
        variability = rng.lognormal(mean=0.0, sigma=2.0, size=(1, block_size))
        # 平均が元の p (0.08) になるように正規化
        variability = variability / np.mean(variability)
        p_vec = p * variability
        # 確率が 1.0 を超えないようにクリップ
        p_vec = np.clip(p_vec, 0.0, 1.0)
        # rng.rand(N, N) < (1, N) の比較により、ブロードキャスト
        mask = rng.rand(block_size, block_size) < p_vec
        resovoir_weight[i1:i2, i1:i2] = (
            G * (rng.rand(block_size, block_size) - 0.5) + offset
        ) * mask

        crust_idx += 1

    # クラスター間の接続
    M = 4
    G = 0.5
    p = 0.01
    offset = 1.0
    for hoge in range(M):
        i_range1 = int((hoge * N / 4) % N)
        i_range2 = int((hoge + 1) * N / 4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int(((hoge + 1) * N / 4) % N)
        j_range2 = int((hoge + 2) * N / 4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        # resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
        #     (G * rng.randn(N // 4, N // 4)) + offset
        # ) * (rng.rand(N // 4, N // 4) < p)
        # resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
        #     G*(rng.rand(N//4, N//4)-0.5) + offset
        #     ) * (rng.rand(N//4, N//4) < p)
        
        variability = rng.lognormal(mean=0.0, sigma=2.0, size=(1, block_size))
        variability = variability / np.mean(variability)
        p_vec = p * variability
        p_vec = np.clip(p_vec, 0.0, 1.0)
        mask = rng.rand(block_size, block_size) < p_vec
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            G * (rng.rand(block_size, block_size) - 0.5) + offset
        ) * mask

        i_range1 = int(((hoge + 1) * N / 4) % N)
        i_range2 = int((hoge + 2) * N / 4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int((hoge * N / 4) % N)
        j_range2 = int((hoge + 1) * N / 4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        # resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
        #     (G * rng.randn(N // 4, N // 4)) + offset
        # ) * (rng.rand(N // 4, N // 4) < p)
        # resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
        #     G*(rng.rand(N//4, N//4)-0.5) + offset
        #     ) * (rng.rand(N//4, N//4) < p)
        
        variability = rng.lognormal(mean=0.0, sigma=2.0, size=(1, block_size))
        variability = variability / np.mean(variability)
        p_vec = p * variability
        p_vec = np.clip(p_vec, 0.0, 1.0)
        mask = rng.rand(block_size, block_size) < p_vec
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            G * (rng.rand(block_size, block_size) - 0.5) + offset
        ) * mask

    # 抑制結合の設定
    mask = np.ones((N, N))
    inhi_idx = rng.choice(np.arange(N), int(N/4), replace=False)
    mask[:, inhi_idx] = -1
    resovoir_weight = resovoir_weight * mask
    # resovoir_weight = np.zeros((N, N))#test
    # resovoir_weight[0, 1] = 1       #test
    type = np.where(mask[0,:] == 1, 0, 1)
    mask = (resovoir_weight != 0) * mask

    return resovoir_weight, mask, type


def create_random_matrix(N, rng):
    resovoir_weight = np.zeros((N, N))
    G = 0.1
    p = 0.05
    resovoir_weight = ((G * rng.randn(N, N)) + 1) * (rng.rand(N, N) < p)

    # 抑制結合の設定
    mask = np.ones((N, N))
    inhi_idx = rng.choice(np.arange(N), int(N/4), replace=False)
    mask[:, inhi_idx] = -1
    resovoir_weight = resovoir_weight * mask
    # resovoir_weight = np.zeros((N, N))#test
    # resovoir_weight[0, 1] = 1       #test
    type = np.where(mask[0,:] == 1, 0, 1)
    mask = (resovoir_weight != 0) * mask

    return resovoir_weight, mask, type


def synapses_init(resovoir_weight, N, N_S):
    tr = 5e-3
    td = 1e-1
    tau_rec = np.full(N_S, 0.5, dtype=np.float32)
    tau_inact = np.full(N_S, 0.3, dtype=np.float32)
    tau_faci = np.full(N_S, 0.53, dtype=np.float32)
    U1 = np.full(N_S, 0.05, dtype=np.float32)
    U = np.full(N_S, 0.1, dtype=np.float32)
    mask_faci = np.zeros(N_S, dtype=np.uint8)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        if resovoir_weight[r, c] < 0:
            mask_faci[i] = 1
            U[i] = 0
            tau_rec[i] = 0.1
            tau_inact[i] = 0.0015
        # if r == c and c%(N//4) < N//5:
        #     U[i] = 0.1
        #     tau_rec[i] = 0.1
    return tau_rec, tau_inact, tau_faci, U1, U, mask_faci, tr, td


def calc_init(resovoir_weight, N, N_S):
    neuron_from = np.zeros(N_S, dtype=np.int32)
    resovoir_weight_calc = np.zeros(N_S, dtype=np.float32)
    # resovoir_weight_calc = np.zeros((N, N_S), dtype=np.float32)
    neuron_to = np.zeros(N_S, dtype=np.int32)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        neuron_from[i] = c
        resovoir_weight_calc[i] = resovoir_weight[r, c]
        # resovoir_weight_calc[r, i] = resovoir_weight[r, c]
        neuron_to[i] = r
    return neuron_from, resovoir_weight_calc, neuron_to


def delay_init(resovoir_weight, N, N_S, mask, rng):
    delays = rng.randint(100, 1000, size=(N,N))
    # delays = np.full((N, N), 1000, dtype=np.int32)
    # delays = (40 + 60 * rng.rand(N, N)).astype(np.int32)  # 平均4ms 標準偏差0.75ms
    delays = delays * (mask != 0)
    delay_row = np.zeros(N_S, dtype=np.int32)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        delay_row[i] = delays[r, c]
        if r % (N // 4) != c % (N // 4):
            delay_row[i] = 30
    return delay_row

def set_global_seed(seed: int):
    """すべての乱数生成器のシードを固定する"""
    random.seed(seed)
    np.random.seed(seed)
    # cp.random.seed(seed)
    Config.SEED = seed  # Configの値も更新
    print(f"[Config] Global seed set to {seed}")
