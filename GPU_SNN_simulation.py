import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import random
from line_profiler import LineProfiler
import networkx as nx
import time
import os
import pandas as pd
from src.PQN import PQNparam


SEED = int(random.random() * 1000)
# SEED = 678
random.seed(SEED)  # for reproducibility
np.random.seed(SEED)

# -------------------------------------------------------------
# 1. 外部の .cu ファイルを読み込んで文字列として取得
# -------------------------------------------------------------
with open("my_kernel.cu", "r", encoding="utf-8") as f:
    my_kernel_code = f.read()

# CuPyのRawKernelとしてカーネルをコンパイル
module = SourceModule(my_kernel_code)
update_neuron_state = module.get_function("update_neuron_state")
copy_arrival_spike = module.get_function("copy_arrival_spike")
propagate_spikes = module.get_function("propagate_spikes")
synapses_calc = module.get_function("synapses_calc")
mat_vec_mul = module.get_function("mat_vec_mul")


# -------------------------------------------------------------
# 2. メイン処理
# -------------------------------------------------------------
def main(
    input_data: np.ndarray | None = None,
    reservoir_state=None,
    label: str = "unknown",
    return_feature: bool = False,
    is_debug_print: bool = True,
    Nin: int = 100,
    density: float = 0.1,
    N: int = 500,
    record: bool = False,
    save_dir="graphs",
):
    """
    SNNシミュレーションのメイン関数
    Parameters:
        input_data: np.ndarray or None
            shape = (T, N_in)
            cochleagram or other time-series input.
    """
    # --- 初期設定 ---
    # Use a single PQN neuron model for all neurons (no RSexci/RSinhi branching)
    neuron_classes = ["RS_exci","RS_inhi","FS","EB","LTS","IB","PB","Class2"]
    neuron_mode = "RSexci"  # TODO: make this configurable (e.g., FS, LTS, ...)
    cell = PQNparam(mode=neuron_mode)
    cell_param_h = param_h_init(cell)
    tmax = 10  # [s]
    dt = cell.PARAM['dt']
    S_durt = 8e-3  # 8[ms]
    # --- 外部入力がある場合はシミュレーション長とtmaxを調整 ---
    if input_data is not None:
        tmax = (
            input_data.shape[0] * S_durt
        )  # assuming input_data was downsampled to 125Hz
        num_steps = int(tmax / dt)
    else:
        num_steps = int(tmax / dt)

    # --- layer definition: input, reservoir, output ---
    # --- layer definition replaced: now using reservoir_state indices ---
    if reservoir_state is None:
        raise ValueError("reservoir_state must be provided.")
    input_indices = reservoir_state["input_indices"]
    output_indices = reservoir_state["output_indices"]
    Nin_layer = len(input_indices)
    Nout_layer = len(output_indices)
    # if input_data is not None:
    #     num_steps = input_data.shape[0]
    #     tmax = num_steps * dt
    # else:
    #     num_steps = int(tmax / dt)

    # CPU-side linear projection of input_data (M -> Nin_layer)
    projected_input = None
    if input_data is not None:
        M = input_data.shape[1]
        # 入力チャンネル数分だけ入力層を用意し、そのニューロンにのみ外部入力を与える
        rng = np.random.default_rng(42)
        W_in = rng.normal(0, 1, size=(M, M)).astype(np.float32)
        # shape: (T, Nin_layer)  → 入力層ニューロンの「前処理された入力」
        projected_input = input_data @ W_in

    v = np.zeros((num_steps, N))
    rasters = np.zeros((num_steps, N), dtype=np.uint8)
    input = np.zeros((num_steps, N), dtype=np.float32)
    buffer_size = 1001
    plot_num = 0

    # PyCUDAでストリームとイベントを作成
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()
    stream3 = cuda.Stream()
    event_update_neuron = cuda.Event()
    event_update_input = cuda.Event()

    # 3. ホスト側(CPU)でデータを準備

    Vs_h = np.full(N, cell.state_variable_v, dtype=np.int64)  # membrane potential
    Ns_h = np.full(N, cell.state_variable_n, dtype=np.int64)
    Qs_h = np.full(N, cell.state_variable_q, dtype=np.int64)
    Us_h = np.full(N, cell.state_variable_u, dtype=np.int64)
    raster_h = np.full(N, 0, dtype=np.uint8)
    spike_in_h = np.full(N, 0, dtype=np.uint8)

    # ---- 重み行列の作成 ----
    if reservoir_state is None:
        raise ValueError("reservoir_state must be provided.")
    reservoir_weight = reservoir_state["reservoir_weight"]
    mask = reservoir_state["mask"]
    type = reservoir_state["type"]
    N_S = reservoir_state["N_S"]
    tau_rec_h = reservoir_state["tau_rec_h"]
    tau_inact_h = reservoir_state["tau_inact_h"]
    tau_faci_h = reservoir_state["tau_faci_h"]
    U1_h = reservoir_state["U1_h"]
    U_h = reservoir_state["U_h"]
    mask_faci_h = reservoir_state["mask_faci_h"]
    neuron_from_h = reservoir_state["neuron_from_h"]
    calc_matrix_h = reservoir_state["calc_matrix_h"]
    neuron_to_h = reservoir_state["neuron_to_h"]
    delayed_row_h = reservoir_state["delayed_row_h"]

    td_float32 = np.float32(1e-2)
    tr_float32 = np.float32(5e-3)

    # 4. デバイス側(GPU)にメモリを確保し、データを転送
    PQN_param_d, PQN_param_d_size = module.get_global("PQN_param")
    cuda.memcpy_htod(PQN_param_d, cell_param_h)
    # RSexci_param_d = gpuarray.to_gpu(RSexci_param_h)
    # RSinhi_param_d = gpuarray.to_gpu(RSinhi_param_h)

    dt_float32 = np.float32(dt)
    dt_d, dt_d_size = module.get_global("dt")
    cuda.memcpy_htod(dt_d, dt_float32)

    buffer_size_int32 = np.int32(buffer_size)
    buffer_size_d, buffer_size_d_size = module.get_global("buffer_size")
    cuda.memcpy_htod(buffer_size_d, buffer_size_int32)

    n_int32 = np.int32(N)
    n_d, n_d_size = module.get_global("num_neurons")
    cuda.memcpy_htod(n_d, n_int32)

    n_s_int32 = np.int32(N_S)
    ns_d, ns_d_size = module.get_global("num_synapses")
    cuda.memcpy_htod(ns_d, n_s_int32)

    Vs_d = gpuarray.to_gpu(Vs_h)
    Ns_d = gpuarray.to_gpu(Ns_h)
    Qs_d = gpuarray.to_gpu(Qs_h)
    Us_d = gpuarray.to_gpu(Us_h)

    x_h = np.full(N_S, 1.0, dtype=np.float32)
    z_h = np.full(N_S, 0.0, dtype=np.float32)
    x_d = gpuarray.to_gpu(x_h)
    y_d = gpuarray.zeros(N_S, dtype=np.float32)
    z_d = gpuarray.to_gpu(z_h)
    r_d = gpuarray.zeros(N_S, dtype=np.float32)
    hr_d = gpuarray.zeros(N_S, dtype=np.float32)
    tau_rec_d = gpuarray.to_gpu(tau_rec_h)
    tau_inact_d = gpuarray.to_gpu(tau_inact_h)
    tau_faci_d = gpuarray.to_gpu(tau_faci_h)
    U1_d = gpuarray.to_gpu(U1_h)
    U_d = gpuarray.to_gpu(U_h)
    mask_faci_d = gpuarray.to_gpu(mask_faci_h)
    neuron_from_d = gpuarray.to_gpu(neuron_from_h)
    calc_matrix_d = gpuarray.to_gpu(calc_matrix_h)
    neuron_to_d = gpuarray.to_gpu(neuron_to_h)
    delayed_row_d = gpuarray.to_gpu(delayed_row_h)
    last_spike_d = gpuarray.zeros(N, dtype=np.uint8)
    raster_d = gpuarray.to_gpu(raster_h)
    delayed_spikes_d = gpuarray.zeros((buffer_size, N_S), dtype=np.uint8)
    arrival_spike_d = gpuarray.zeros(N_S, dtype=np.uint8)
    spike_in_d = gpuarray.to_gpu(spike_in_h)
    synapses_out_d = gpuarray.zeros(N, dtype=np.float32)
    I_input_d = gpuarray.zeros(N, dtype=np.float32)

    # 5. カーネルの実行設定
    neuron_threads_per_block = 256
    neuron_blocks_per_grid = (
        N + neuron_threads_per_block - 1
    ) // neuron_threads_per_block
    synapse_threads_per_block = 256
    synapse_blocks_per_grid = int(
        (N_S + synapse_threads_per_block - 1) // synapse_threads_per_block
    )

    # 6. シミュレーションループ
    start = time.perf_counter()
    loop_iter = tqdm(range(num_steps)) if is_debug_print else range(num_steps)

    # 各ニューロンに対するスパイク生成確率（入力層のみ非ゼロ）
    prob_all = np.zeros(N, dtype=np.float32)

    for i in loop_iter:
        read_idx = np.int32(i % buffer_size)
        update_neuron_state(  # stream1
            Vs_d.gpudata,
            Ns_d.gpudata,
            Qs_d.gpudata,
            Us_d.gpudata,
            I_input_d.gpudata,
            synapses_out_d.gpudata,
            last_spike_d.gpudata,
            raster_d.gpudata,
            np.int32(i),
            block=(neuron_threads_per_block, 1, 1),
            grid=(neuron_blocks_per_grid, 1),
            stream=stream1,
        )
        event_update_neuron.record(stream1)
        propagate_spikes(  # stream1
            delayed_spikes_d.gpudata,
            raster_d.gpudata,
            spike_in_d.gpudata,
            neuron_from_d.gpudata,
            delayed_row_d.gpudata,
            read_idx,
            block=(synapse_threads_per_block, 1, 1),
            grid=(synapse_blocks_per_grid, 1),
            stream=stream1,
        )

        synapses_calc(  # stream2
            x_d.gpudata,
            y_d.gpudata,
            z_d.gpudata,
            r_d.gpudata,
            hr_d.gpudata,
            delayed_spikes_d.gpudata,
            mask_faci_d.gpudata,
            tau_rec_d.gpudata,
            tau_inact_d.gpudata,
            tau_faci_d.gpudata,
            U1_d.gpudata,
            U_d.gpudata,
            td_float32,
            tr_float32,
            read_idx,
            block=(synapse_threads_per_block, 1, 1),
            grid=(synapse_blocks_per_grid, 1),
            stream=stream2,
        )
        stream2.wait_for_event(event_update_neuron)
        mat_vec_mul(  # stream2
            synapses_out_d.gpudata,
            neuron_to_d.gpudata,
            calc_matrix_d.gpudata,
            r_d.gpudata,
            block=(synapse_threads_per_block, 1, 1),
            grid=(synapse_blocks_per_grid, 1),
            stream=stream2,
        )
        event_update_input.record(stream2)

        stream3.wait_for_event(event_update_neuron)
        if record:
            cuda.memcpy_dtoh_async(Vs_h, Vs_d.gpudata, stream=stream3)
        cuda.memcpy_dtoh_async(rasters[i], raster_d.gpudata, stream=stream3)

        steps_per_frame = int(round(S_durt / dt))
        if projected_input is not None:
            # S_durt ごとに入力フレームを1ステップ進める
            if i % steps_per_frame == 0:
                idx = i // steps_per_frame
                if idx >= projected_input.shape[0]:
                    idx = projected_input.shape[0] - 1
                # 入力層ニューロン用の確率のみ更新
                prob_input = 1 / (1 + np.exp(-projected_input[idx]))
                prob_all[:] = 0.0
                prob_all[input_indices] = prob_input

        # 入力層ニューロンのみ確率的スパイクを発火させる
        spike_in_h = (np.random.rand(N) < prob_all).astype(np.uint8)
        cuda.memcpy_htod_async(spike_in_d.gpudata, spike_in_h, stream=stream3)

        # if i%(S_durt/dt) == 0:
        #     idx =  int(i/(S_durt/dt)-1)
        #     I_input_h = projected_input[idx].astype(np.float32)
        #     cuda.memcpy_htod_async(I_input_d.gpudata, I_input_h, stream=stream3)

        # if projected_input is not None and i < num_steps:
        #     prob = 1 / (1 + np.exp(-projected_input[i]))  # sigmoid on projected input
        #     spike_in_h = (np.random.rand(N) < prob).astype(np.uint8)
        # else:
        #     spike_in_h = np.zeros(N, dtype=np.uint8)
        # cuda.memcpy_htod_async(spike_in_d.gpudata, spike_in_h, stream=stream3)

        stream3.synchronize()
        if record:
            v[i] = Vs_h
            # input[i] = I_input_h
        # rasters[i] = rasters[i] | spike_in_h
        # stream2.synchronize()
        # stream1.synchronize()
        # input[i] = x_d.get()
        # v[i] = y_d.get()

        stream1.wait_for_event(event_update_input)

    end = time.perf_counter()

    if is_debug_print:
        print(
            f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}"
        )
        print(f"SEED value was {SEED}")
    v = v / 2**cell.BIT_WIDTH_FRACTIONAL
    # ---- plot simulation result ----
    if record:
        os.makedirs(save_dir, exist_ok=True)

        all_indices = set(range(N))
        input_set = set(input_indices)
        output_set = set(output_indices)
        hidden_indices = list(all_indices - input_set - output_set)
        hidden_indices.sort()
        sorted_indices = list(input_indices) + hidden_indices + list(output_indices)
        v_arranged = v[:, sorted_indices]
        plot_single_neuron(tmax, v_arranged, save_dir, plot_num)
        plot_num += 1

        rasters_arranged = rasters[:, sorted_indices]
        plot_raster(dt, tmax, rasters_arranged, save_dir, plot_num)
        plot_num += 1

    # plt.show()

    if return_feature:
        plt.close("all")
        read_indices = reservoir_state["output_indices"]
        x_t = rasters[:, read_indices].astype(np.float32)  # shape = (T, M)
        return x_t


def param_h_init(PQN):
    """Build a fixed-size (34) parameter vector for the CUDA kernel.

    Layout:
      0-26 : existing RS/FS/EB/LTS/IB/PB/Class2 parameters (as before)
      27   : u_v
      28   : u_u
      29   : u_c
      30   : ru
      31   : n_uS (eta0)
      32   : n_uL (eta1)
      33   : v_u (PB only)

    For modes that do not use these terms, the entries remain 0 so the CUDA
    kernel behaves exactly like the old implementation.
    """
    param = np.zeros(34, dtype=np.int32)

    # --- Common 0-26 mapping (works for RS/FS/EB/LTS/IB/PB) ---
    if PQN.mode in ["RSexci", "RSinhi", "FS", "EB", "LTS", "IB", "PB"]:
        param[0] = PQN.BIT_Y_SHIFT
        param[1] = PQN.BIT_WIDTH_FRACTIONAL
        param[2] = PQN.Y.get("v_vv_S", 0)
        param[3] = PQN.Y.get("v_v_S", 0)
        param[4] = PQN.Y.get("v_c_S", 0)
        param[5] = PQN.Y.get("v_n", 0)
        param[6] = PQN.Y.get("v_q", 0)
        param[7] = PQN.Y.get("v_I", 0)
        param[8] = PQN.Y.get("v_vv_L", 0)
        param[9] = PQN.Y.get("v_v_L", 0)
        param[10] = PQN.Y.get("v_c_L", 0)
        param[11] = PQN.Y.get("rg", 0)
        param[12] = PQN.Y.get("n_vv_S", 0)
        param[13] = PQN.Y.get("n_v_S", 0)
        param[14] = PQN.Y.get("n_c_S", 0)
        param[15] = PQN.Y.get("n_n", 0)
        param[16] = PQN.Y.get("n_vv_L", 0)
        param[17] = PQN.Y.get("n_v_L", 0)
        param[18] = PQN.Y.get("n_c_L", 0)
        param[19] = PQN.Y.get("rh", 0)
        param[20] = PQN.Y.get("q_vv_S", 0)
        param[21] = PQN.Y.get("q_v_S", 0)
        param[22] = PQN.Y.get("q_c_S", 0)
        param[23] = PQN.Y.get("q_q", 0)
        param[24] = PQN.Y.get("q_vv_L", 0)
        param[25] = PQN.Y.get("q_v_L", 0)
        param[26] = PQN.Y.get("q_c_L", 0)

        # --- Extended u-related params (LTS/IB/PB) ---
        # u dynamics (LTS/IB/PB)
        param[27] = PQN.Y.get("u_v", 0)
        param[28] = PQN.Y.get("u_u", 0)
        param[29] = PQN.Y.get("u_c", 0)

        # IB/LTS: n scaling by u threshold
        param[30] = PQN.Y.get("ru", 0)
        param[31] = PQN.Y.get("n_uS", 0)
        param[32] = PQN.Y.get("n_uL", 0)

        # PB: v-u coupling (already includes sign in PQN.Y['v_u'])
        param[33] = PQN.Y.get("v_u", 0)

        return param

    elif PQN.mode == "Class2":
        # Class2 doesn't have q or u terms; keep missing entries as 0
        param[0] = PQN.BIT_Y_SHIFT
        param[1] = PQN.BIT_WIDTH_FRACTIONAL
        param[2] = PQN.Y.get("v_vv_S", 0)
        param[3] = PQN.Y.get("v_v_S", 0)
        param[4] = PQN.Y.get("v_c_S", 0)
        param[5] = PQN.Y.get("v_n", 0)
        param[7] = PQN.Y.get("v_I", 0)
        param[8] = PQN.Y.get("v_vv_L", 0)
        param[9] = PQN.Y.get("v_v_L", 0)
        param[10] = PQN.Y.get("v_c_L", 0)
        param[11] = PQN.Y.get("rg", 0)
        param[12] = PQN.Y.get("n_vv_S", 0)
        param[13] = PQN.Y.get("n_v_S", 0)
        param[14] = PQN.Y.get("n_c_S", 0)
        param[15] = PQN.Y.get("n_n", 0)
        param[16] = PQN.Y.get("n_vv_L", 0)
        param[17] = PQN.Y.get("n_v_L", 0)
        param[18] = PQN.Y.get("n_c_L", 0)
        return param

    else:
        raise ValueError("Invalid PQN mode")


def create_moduled_matrix(N):
    reservoir_weight = np.zeros((N, N))
    crust_idx = 0
    G = 0.1
    p = 0.05
    while crust_idx != 4:
        i1 = int(crust_idx * N / 4)
        i2 = int((crust_idx + 1) * N / 4)
        reservoir_weight[i1:i2, i1:i2] = ((G * np.random.randn(N // 4, N // 4)) + 1) * (
            np.random.rand(N // 4, N // 4) < p
        )
        # print(reservoir_weight[i1:i2, i1:i2])
        crust_idx += 1

    # クラスター間の接続
    M = 4
    G = 0.1
    p = 0.01
    for hoge in range(M):
        i_range1 = int((hoge * N / 4) % N)
        i_range2 = int((hoge + 1) * N / 4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int(((hoge + 1) * N / 4) % N)
        j_range2 = int((hoge + 2) * N / 4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            (G * np.random.randn(N // 4, N // 4)) + 1
        ) * (np.random.rand(N // 4, N // 4) < p)

        i_range1 = int(((hoge + 1) * N / 4) % N)
        i_range2 = int((hoge + 2) * N / 4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int((hoge * N / 4) % N)
        j_range2 = int((hoge + 1) * N / 4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            (G * np.random.randn(N // 4, N // 4)) + 1
        ) * (np.random.rand(N // 4, N // 4) < p)

    # 抑制結合の設定
    base_mask = np.ones((N, int(N / 4)))
    base_mask[:, N // 5 :] = -1
    mask = np.hstack([base_mask for _ in range(4)])
    reservoir_weight = reservoir_weight * mask
    # reservoir_weight = np.zeros((N, N))#test
    # reservoir_weight[0, 1] = 1       #test
    type = mask
    mask = (reservoir_weight != 0) * mask

    return reservoir_weight, mask, type


def create_random_matrix(N):
    reservoir_weight = np.zeros((N, N))
    G = 0.1
    p = 0.05
    reservoir_weight = ((G * np.random.randn(N, N)) + 1) * (np.random.rand(N, N) < p)

    # 抑制結合の設定
    mask = np.ones((N, N))
    mask[:, int(4 * N / 5) :] = -1
    reservoir_weight = reservoir_weight * mask
    # reservoir_weight = np.zeros((N, N))#test
    # reservoir_weight[0, 1] = 1       #test
    mask = (reservoir_weight != 0) * mask

    return reservoir_weight, mask


def synapses_init(reservoir_weight, N, N_S):
    tau_rec = np.full(N_S, 0.2, dtype=np.float32)
    tau_inact = np.full(N_S, 0.003, dtype=np.float32)
    tau_faci = np.full(N_S, 0.53, dtype=np.float32)
    U1 = np.full(N_S, 0.3, dtype=np.float32)
    U = np.full(N_S, 0.5, dtype=np.float32)
    mask_faci = np.zeros(N_S, dtype=np.uint8)
    col_indices, row_indices = np.where(reservoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        if r % (N // 4) > N // 5:
            mask_faci[i] = 1
            U[i] = 0
            tau_rec[i] = 0.1
            tau_inact[i] = 0.0015
        # if r == c and c%(N//4) < N//5:
        #     U[i] = 0.1
        #     tau_rec[i] = 0.1
    return tau_rec, tau_inact, tau_faci, U1, U, mask_faci


def calc_init(reservoir_weight, N, N_S):
    neuron_from = np.zeros(N_S, dtype=np.int32)
    reservoir_weight_calc = np.zeros(N_S, dtype=np.float32)
    # reservoir_weight_calc = np.zeros((N, N_S), dtype=np.float32)
    neuron_to = np.zeros(N_S, dtype=np.int32)
    col_indices, row_indices = np.where(reservoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        neuron_from[i] = c
        reservoir_weight_calc[i] = reservoir_weight[r, c]
        # reservoir_weight_calc[r, i] = reservoir_weight[r, c]
        neuron_to[i] = r
    return neuron_from, reservoir_weight_calc, neuron_to


def delay_init(reservoir_weight, N, N_S, mask):
    # delays = np.random.randint(100, 700, size=(N,N))
    # delays = np.full((N, N), 1000, dtype=np.int32)
    delays = (40 + 7.5 * np.random.randn(N, N)).astype(
        np.int32
    )  # 平均4ms 標準偏差0.75ms
    delays = delays * (mask != 0)
    delay_row = np.zeros(N_S, dtype=np.int32)
    col_indices, row_indices = np.where(reservoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        delay_row[i] = delays[r, c]
        if r % (N // 4) != c % (N // 4):
            delay_row[i] = 30
    return delay_row


def init_reservoir(N, seed, input_size):
    random.seed(seed)
    np.random.seed(seed)
    reservoir_origin, mask, type = create_moduled_matrix(N)
    reservoir_weight = np.copy(reservoir_origin) * 0.03

    N_S = np.count_nonzero(reservoir_weight)
    tau_rec_h, tau_inact_h, tau_faci_h, U1_h, U_h, mask_faci_h = synapses_init(
        reservoir_weight, N, N_S
    )
    neuron_from_h, calc_matrix_h, neuron_to_h = calc_init(reservoir_weight, N, N_S)
    delayed_row_h = delay_init(reservoir_weight, N, N_S, mask)

    rng = np.random.default_rng(seed)

    input_indices = rng.choice(N, size=input_size, replace=False)
    remaining = np.setdiff1d(np.arange(N), input_indices)
    output_indices = rng.choice(remaining, size=input_size, replace=False)
    visualize_matrix(reservoir_origin, 10)
    visualize_network(reservoir_origin, input_indices, output_indices, 11)
    return {
        "reservoir_weight": reservoir_weight,
        "mask": mask,
        "type": type,
        "N_S": N_S,
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


def visualize_matrix(matrix, num):
    plt.figure(num=num, figsize=(8, 6))
    max_abs = np.max(np.abs(matrix))
    im = plt.imshow(matrix, aspect="auto", cmap="plasma", vmin=-max_abs, vmax=max_abs)
    plt.gca().invert_yaxis()
    plt.colorbar(im, label="Weight Value")
    plt.title("Reservoir Weight Matrix")
    plt.xlabel("Pre Neuron")
    plt.ylabel("Post Neuron")
    plt.tight_layout()
    save_path = os.path.join("graphs", "reservoir_weight_matrix.png")
    plt.savefig(save_path)
    save_path = os.path.join("graphs", "reservoir_weight.npy")
    np.save(save_path, matrix)

def visualize_network(reservoir_weight, input_indices, output_indices, num):
    N = reservoir_weight.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            w = reservoir_weight[i][j]
            if w != 0:
                G.add_edge(j, i, weight=w)  # j→i（pre→post）

    for i in range(N):
        if i in input_indices:
            G.nodes[i]['subset'] = 0
        elif i in output_indices:
            G.nodes[i]['subset'] = 2
        else:
            G.nodes[i]['subset'] = 1
    
    straight_edges = []
    straight_widths = []
    straight_sizes = []
    curved_edges = []
    curved_widths = []
    curved_sizes = []
    for u, v, d in G.edges(data=True):
        edge_widths = max(0.1, 5 * (abs(d['weight']) - 0.7))
        arrow_size = 20*abs(d["weight"])
        
        if G.nodes[u]['subset'] == G.nodes[v]['subset']:
            curved_edges.append((u, v))
            curved_widths.append(edge_widths)
            curved_sizes.append(arrow_size)
        else:
            straight_edges.append((u, v))
            straight_widths.append(edge_widths)
            straight_sizes.append(arrow_size)

    #   ノード配置（円形 or 自動レイアウト） 
    # pos = nx.spring_layout(G, seed=42)  # spring_layout / circular_layout / kamada_kawai_layout など
    pos = nx.multipartite_layout(G, subset_key='subset')

    input_set = set(input_indices)
    output_set = set(output_indices)
    nodes_list = list(G.nodes())
    node_colors = []
    for node in nodes_list:
        if node in input_set:
            node_colors.append('red')
        elif node in output_set:
            node_colors.append('green')
        else:
            node_colors.append('blue')

    #   可視化 
    plt.figure(num=num, figsize=(8, 11))
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_colors)
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=straight_edges,
        width=straight_widths, 
        edge_color="blue", 
        arrows=True, 
        arrowstyle='-|>',
        arrowsize = straight_sizes
    )
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=curved_edges,
        width=curved_widths, 
        edge_color="blue", 
        arrows=True, 
        arrowstyle='-|>',
        arrowsize = curved_sizes,
        connectionstyle='arc3, rad=0.4'
    )
    plt.title("Reservoir Network Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graphs/network.png")
    # plt.show(block=False)

def plot_single_neuron(tmax, v0, save_dir, num):
    matrix_csv_path = os.path.join(save_dir, "membrane_potential.csv")
    np.savetxt(matrix_csv_path, v0, delimiter=",")
    # print(f"Matrix CSV Saved: {matrix_csv_path}")

    num_neurons = v0.shape[1]

    fig = plt.figure(num=num, figsize=(10, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        v0.T,
        aspect="auto",
        origin="lower",
        extent=[0, tmax, 0, num_neurons],
        cmap="viridis",
        vmin=-5,
        vmax=5,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Membrane Potential (mV)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Membrane Potential of All Neurons")
    save_path = os.path.join(save_dir, "membrane_potential.png")
    plt.savefig(save_path)
    plt.close()


def plot_raster(dt, tmax, rasters, save_dir, num):
    N = rasters.shape[1]
    times, neuron_ids = np.nonzero(rasters)
    times = times * dt

    df = pd.DataFrame({
        'time': times,
        'neuron_id': neuron_ids
    })
    csv_path = os.path.join(save_dir, "raster_data.csv")
    df.to_csv(csv_path, index=False)
    # print(f"CSV Saved: {csv_path}")

    cluster_colors = ["red", "blue", "green"]
    cluster_id = (neuron_ids) // 16  # 0,1,2 のクラスタID
    colors = [cluster_colors[c % 3] for c in cluster_id]
    plt.figure(num=num, figsize=(9, 5))
    plt.scatter(times, neuron_ids, s=2.0, color=colors)
    plt.xlabel("time")
    plt.xlim(0, tmax)
    plt.ylabel("neuron ID")
    plt.ylim(0, N)
    plt.title("Raster Plot")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "raster.png")
    plt.savefig(save_path)


if __name__ == "__main__":
    # profiler = LineProfiler()
    # profiler.add_function(main)
    # profiler.runcall(main)
    # profiler.print_stats()

    # coch = np.load("coch_zero.npy")
    reservoir = init_reservoir(N=48, seed=123, input_size=16)
    # main(
    #     N=48,
    #     input_data=coch,
    #     reservoir_state=reservoir,
    #     label="cochleagram",
    #     record=True,
    # )
