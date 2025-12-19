import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from line_profiler import LineProfiler
import time
import os
import sys
from pathlib import Path

# sys.path.append(
#     str(Path(__file__).resolve().parents[1])
# )  # HACK: 親ディレクトリをパスに追加
from src.PQN import PQNparam


# SEED = int(random.random() * 1000)
# # SEED = 678
# random.seed(SEED)  # for reproducibility
# np.random.seed(SEED)

SEED = 0
# REC = True
REC = False
if REC:
    DATE = time.strftime("%Y%m%d")
    TIMESTAMP = time.strftime("%H%M")
    OUTDIR = f"sim_results/{DATE}/{TIMESTAMP}_SEED{SEED}"
    os.makedirs(OUTDIR, exist_ok=True)

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


# メイン処理
def main(
    input_data: np.ndarray | None = None,
    reservoir_state=None,
    return_feature: bool = False,
    is_debug_print: bool = True,
    density: float = 0.1,
    N: int = 500,
    record: bool = False,
    S_durt = 1e-2,  # 10[ms]
    cfg=None,
):
    """
    SNNシミュレーションのメイン関数
    Parameters:
        input_data: np.ndarray or None
            shape = (T, N_in)
            cochleagram or other time-series input.

    reservoir state : typeとneuron_type_hが直接対応するように変更 (クロキ)
    type = {-1, 1} -> {1, 0}
    """
    # --- 初期設定 ---
    tmax = 10  # [s]
    dt = 1e-4
    # --- 外部入力がある場合はシミュレーション長とtmaxを調整 ---
    if input_data is not None:
        tmax = (
            input_data.shape[0] * S_durt
        )
        num_steps = int(tmax / dt)
    else:
        num_steps = int(tmax / dt)

    # --- layer definition: input, reservoir, output ---
    # --- layer definition replaced: now using reservoir_state indices ---
    if reservoir_state is None:
        raise ValueError("reservoir_state must be provided.")
    input_indices = reservoir_state["input_indices"]
    output_indices = reservoir_state["output_indices"]
    # Nin = len(input_indices)
    # Nout = len(output_indices)

    # CPU-side linear projection of input_data (M -> Nin_layer)
    projected_input = None
    if input_data is not None:
        M = input_data.shape[1]
        # 入力チャンネル数分だけ入力層を用意し、そのニューロンにのみ外部入力を与える
        rng = np.random.default_rng(42)
        W_in = rng.normal(0, 1, size=(M, M)).astype(np.float32)
        # shape: (T, Nin_layer)  → 入力層ニューロンの「前処理された入力」
        projected_input = input_data @ W_in

    v = np.zeros((num_steps, N), dtype=np.int64)
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
    neuron_type_h = np.zeros(N, dtype=np.uint8)  # 各ニューロンのタイプ
    Vs_h = np.full(N, -4906, dtype=np.int64)  # 各ニューロンの膜電位などの状態変数
    Ns_h = np.full(N, 27584, dtype=np.int64)
    Qs_h = np.full(N, -3692, dtype=np.int64)
    raster_h = np.full(N, 0, dtype=np.uint8)
    spike_in_h = np.full(N, 0, dtype=np.uint8)

    RSexci = PQNparam(mode="RSexci")
    RSexci_param_h = param_h_init(RSexci)
    RSinhi = PQNparam(mode="RSinhi")
    RSinhi_param_h = param_h_init(RSinhi)

    # ---- 重み行列の作成 ----
    if reservoir_state is None:
        raise ValueError("reservoir_state must be provided.")
    neuron_type_h = reservoir_state["type"]      # exciRS / inhiRS / ...
    # synapse
    N_S = reservoir_state["N_S"]

    for i in range(N):
        if neuron_type_h[i] == 0:
            Vs_h[i] = RSexci.state_variable_v
            Ns_h[i] = RSexci.state_variable_n
            Qs_h[i] = RSexci.state_variable_q
        elif neuron_type_h[i] == 1:
            Vs_h[i] = RSinhi.state_variable_v
            Ns_h[i] = RSinhi.state_variable_n
            Qs_h[i] = RSinhi.state_variable_q

    # synaptic filter time constants
    td_float32 = np.float32(1e-2)
    tr_float32 = np.float32(5e-3)

    # 4. デバイス側(GPU)にメモリを確保し、データを転送
    RSexci_param_d, RSexci_param_d_size = module.get_global("RSexci_param")
    cuda.memcpy_htod(RSexci_param_d, RSexci_param_h)
    RSinhi_param_d, RSinhi_param_d_size = module.get_global("RSinhi_param")
    cuda.memcpy_htod(RSinhi_param_d, RSinhi_param_h)
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

    neuron_type_d = gpuarray.to_gpu(neuron_type_h)
    Vs_d = gpuarray.to_gpu(Vs_h)
    Ns_d = gpuarray.to_gpu(Ns_h)
    Qs_d = gpuarray.to_gpu(Qs_h)

    x_h = np.full(N_S, 1.0, dtype=np.float32)
    x_d = gpuarray.to_gpu(x_h)
    y_d = gpuarray.zeros(N_S, dtype=np.float32)
    z_h = np.full(N_S, 0.0, dtype=np.float32)
    z_d = gpuarray.to_gpu(z_h)
    r_d = gpuarray.zeros(N_S, dtype=np.float32)
    hr_d = gpuarray.zeros(N_S, dtype=np.float32)
    tau_rec_d = gpuarray.to_gpu(reservoir_state["tau_rec_h"])
    tau_inact_d = gpuarray.to_gpu(reservoir_state["tau_inact_h"])
    tau_faci_d = gpuarray.to_gpu(reservoir_state["tau_faci_h"])
    U1_d = gpuarray.to_gpu(reservoir_state["U1_h"])
    U_d = gpuarray.to_gpu(reservoir_state["U_h"])
    mask_faci_d = gpuarray.to_gpu(reservoir_state["mask_faci_h"])
    neuron_from_d = gpuarray.to_gpu(reservoir_state["neuron_from_h"])
    calc_matrix_d = gpuarray.to_gpu(reservoir_state["calc_matrix_h"])
    neuron_to_d = gpuarray.to_gpu(reservoir_state["neuron_to_h"])
    delayed_row_d = gpuarray.to_gpu(reservoir_state["delayed_row_h"])
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
            neuron_type_d.gpudata,
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
            cuda.memcpy_dtoh_async(v[i], Vs_d.gpudata, stream=stream3)
            # cuda.memcpy_dtoh_async(input[i], synapses_out_d.gpudata, stream=stream3)
        cuda.memcpy_dtoh_async(rasters[i], raster_d.gpudata, stream=stream3)
        cuda.memcpy_dtoh_async(input[i], synapses_out_d.gpudata, stream=stream3)
        # rasters[i] = spike_in_h

        steps_per_frame = int(round(S_durt / dt))
        if input_data is not None:
            # S_durt ごとに入力フレームを1ステップ進める
            if i % steps_per_frame == 0:
                idx = i // steps_per_frame
                if idx >= input_data.shape[0]:
                    idx = input_data.shape[0] - 1
                # 入力層ニューロン用の確率のみ更新
                # prob_input = 1 / (1 + np.exp(-input_data[idx]))
                prob_input = 10*dt*input_data[idx]
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
        rasters[i] = rasters[i] | spike_in_h
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
        # print(f"SEED value was {SEED}")
    v = v / 2**RSexci.BIT_WIDTH_FRACTIONAL
    # ---- plot simulation result ----
    if record:
        visualize_matrix(reservoir_state["reservoir_weight"], plot_num, cfg)
        plot_num += 1

        plot_single_neuron(0, dt, tmax, num_steps, input, v, plot_num, cfg)
        plot_num += 1

        plot_raster(dt, tmax, rasters, N, plot_num, cfg)
        plot_num += 1

    # plt.show()

    if return_feature:
        plt.close("all")
        read_indices = reservoir_state["output_indices"]
        x_t = input[:, read_indices].astype(np.float32)  # shape = (T, Nout)
        return x_t


def param_h_init(PQN):
    if PQN.mode in ["RSexci", "RSinhi", "FS", "EB"]:
        param = np.zeros(27, dtype=np.int32)
        param[0] = PQN.BIT_Y_SHIFT
        param[1] = PQN.BIT_WIDTH_FRACTIONAL
        param[2] = PQN.Y["v_vv_S"]
        param[3] = PQN.Y["v_v_S"]
        param[4] = PQN.Y["v_c_S"]
        param[5] = PQN.Y["v_n"]
        param[6] = PQN.Y["v_q"]
        param[7] = PQN.Y["v_I"]
        param[8] = PQN.Y["v_vv_L"]
        param[9] = PQN.Y["v_v_L"]
        param[10] = PQN.Y["v_c_L"]
        param[11] = PQN.Y["rg"]
        param[12] = PQN.Y["n_vv_S"]
        param[13] = PQN.Y["n_v_S"]
        param[14] = PQN.Y["n_c_S"]
        param[15] = PQN.Y["n_n"]
        param[16] = PQN.Y["n_vv_L"]
        param[17] = PQN.Y["n_v_L"]
        param[18] = PQN.Y["n_c_L"]
        param[19] = PQN.Y["rh"]
        param[20] = PQN.Y["q_vv_S"]
        param[21] = PQN.Y["q_v_S"]
        param[22] = PQN.Y["q_c_S"]
        param[23] = PQN.Y["q_q"]
        param[24] = PQN.Y["q_vv_L"]
        param[25] = PQN.Y["q_v_L"]
        param[26] = PQN.Y["q_c_L"]
        return param
    elif PQN.mode in ["LTS", "IB"]:
        param = np.zeros(27, dtype=np.int32)

        return param
    elif PQN.mode == "PB":
        param = np.zeros(27, dtype=np.int32)

        return param
    elif PQN.mode == "Class2":
        param = np.zeros(27, dtype=np.int32)

        return param
    else:
        raise ValueError("Invalid PQN mode")


def visualize_matrix(matrix, num, cfg):
    plt.figure(num=num, figsize=(8, 6))
    max_abs = np.max(np.abs(matrix))
    im = plt.imshow(matrix, aspect="auto", cmap="plasma", vmin=-max_abs, vmax=max_abs)
    plt.gca().invert_yaxis()
    plt.colorbar(im, label="Weight Value")
    plt.title("Reservoir Weight Matrix")
    plt.xlabel("Pre Neuron")
    plt.ylabel("Post Neuron")
    plt.tight_layout()
    plt.savefig(f"{cfg.RESULT_DIR}/figs/reservoir_weight_matrix.png")
    np.save(f"{cfg.RESULT_DIR}/data/reservoir.npy", matrix)
    if REC:
        save_path = os.path.join(OUTDIR, "resovoir_weight_matrix.png")
        plt.savefig(save_path)


def plot_single_neuron(id, dt, tmax, number_of_iterations, I, v0, num, cfg):
    fig = plt.figure(num=num, figsize=(10, 4))
    spec = gridspec.GridSpec(
        ncols=1, nrows=2, figure=fig, hspace=0.1, height_ratios=[1, 4]
    )
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax0.set_xticks([])
    ax0.plot([i * dt for i in range(0, number_of_iterations)], I[:, id], color="black")
    ax0.set_xlim(0, tmax)
    ax1.plot([i * dt for i in range(0, number_of_iterations)], v0[:, id])
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("v")
    ax0.set_ylabel("I")
    ax1.set_xlabel("[s]")
    plt.savefig(f"{cfg.RESULT_DIR}/figs/single_neuron.png")
    if REC:
        save_path = os.path.join(OUTDIR, f"single_neuron.png")
        plt.savefig(save_path)


def plot_raster(dt, tmax, rasters, N, num, cfg):
    times, neuron_ids = np.nonzero(rasters)
    times = times * dt
    neuron_ids = neuron_ids  # Adjust neuron IDs to start from 1
    cluster_colors = ["red", "blue", "green", "orange"]
    cluster_id = (neuron_ids) // (N // 4)  # 0,1,2,3 のクラスタID
    colors = [cluster_colors[c % 4] for c in cluster_id]
    plt.figure(num=num, figsize=(9, 5))
    plt.scatter(times, neuron_ids, s=1.1, color=colors)
    plt.xlabel("time")
    plt.xlim(0, tmax)
    plt.ylabel("neuron ID")
    plt.ylim(0, N)
    plt.title("Raster Plot")
    plt.tight_layout()
    plt.savefig(f"{cfg.RESULT_DIR}/figs/raster.png")
    np.save(f"{cfg.RESULT_DIR}/data/raster.npy", rasters)
    if REC:
        save_path = os.path.join(OUTDIR, "raster.png")
        plt.savefig(save_path)

