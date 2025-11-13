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
import time
import os
from src.PQN import PQNparam


# SEED = int(random.random() * 1000)
SEED = 678
random.seed(SEED)  # for reproducibility
np.random.seed(SEED)

record = True
record = False
if record:
    DATE = time.strftime("%Y%m%d")
    TIMESTAMP = time.strftime("%H%M")
    OUTDIR = f"sim_results/{DATE}/{TIMESTAMP}_SEED{SEED}"
    os.makedirs(OUTDIR, exist_ok=True)

# -------------------------------------------------------------
# 1. 外部の .cu ファイルを読み込んで文字列として取得
# -------------------------------------------------------------
with open("my_kernel.cu", "r", encoding="utf-8") as f:
    my_kernel_code = f.read()

# CuPyのRawKernelとしてカーネルをコンパイル（この行は変更なし）
module = SourceModule(my_kernel_code)
update_neuron_state = module.get_function("update_neuron_state")
copy_arrival_spike = module.get_function("copy_arrival_spike")
propagate_spikes = module.get_function("propagate_spikes")
synapses_calc = module.get_function("synapses_calc")
mat_vec_mul = module.get_function("mat_vec_mul")


# -------------------------------------------------------------
# 2. メイン処理
# -------------------------------------------------------------
def main(input_data: np.ndarray | None = None, label: str = "unknown"):
    """
    SNNシミュレーションのメイン関数
    Parameters:
        input_data: np.ndarray or None
            shape = (T, N_in)
            cochleagram or other time-series input.
    """
    # --- 初期設定 ---
    N = 100
    tmax = 10  # [s]
    dt = 1e-4
    # --- 外部入力がある場合はシミュレーション長とtmaxを調整 ---
    if input_data is not None:
        num_steps = input_data.shape[0]
        tmax = num_steps * dt
    else:
        num_steps = int(tmax / dt)
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
    k = 0.03
    resovoir_origin, mask, type = create_moduled_matrix(N)
    # resovoir_origin, mask = create_random_matrix(N)
    resovoir_weight = np.copy(resovoir_origin)
    resovoir_weight = resovoir_weight * k
    visualize_matrix(resovoir_origin, plot_num)
    plot_num += 1

    for i in range(N):
        if type[0, i] == 1:
            Vs_h[i] = RSexci.state_variable_v
            Ns_h[i] = RSexci.state_variable_n
            Qs_h[i] = RSexci.state_variable_q
            neuron_type_h[i] = 0
        else:
            Vs_h[i] = RSinhi.state_variable_v
            Ns_h[i] = RSinhi.state_variable_n
            Qs_h[i] = RSinhi.state_variable_q
            neuron_type_h[i] = 1

    N_S = np.count_nonzero(resovoir_weight)

    td_float32 = np.float32(1e-2)
    tr_float32 = np.float32(5e-3)
    tau_rec_h, tau_inact_h, tau_faci_h, U1_h, U_h, mask_faci_h = synapses_init(
        resovoir_weight, N, N_S
    )
    neuron_from_h, calc_matrix_h, neuron_to_h = calc_init(resovoir_weight, N, N_S)
    delayed_row_h = delay_init(resovoir_weight, N, N_S, mask)

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

    # 5. カーネルの実行設定
    neuron_threads_per_block = 256
    neuron_blocks_per_grid = (
        N + neuron_threads_per_block - 1
    ) // neuron_threads_per_block
    synapse_threads_per_block = 256
    synapse_blocks_per_grid = int(
        (N_S + synapse_threads_per_block - 1) // synapse_threads_per_block
    )

    # 入力次元チェック
    if input_data is not None:
        N_input = input_data.shape[1]
        if N_input != N:
            raise ValueError(f"Input dimension mismatch: expected N={N}, got {N_input}")

    # 6. シミュレーションループ
    start = time.perf_counter()
    for i in tqdm(range(num_steps)):
        read_idx = np.int32(i % buffer_size)
        update_neuron_state(  # stream1
            Vs_d.gpudata,
            Ns_d.gpudata,
            Qs_d.gpudata,
            neuron_type_d.gpudata,
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
        cuda.memcpy_dtoh_async(Vs_h, Vs_d.gpudata, stream=stream3)
        cuda.memcpy_dtoh_async(rasters[i], raster_d.gpudata, stream=stream3)

        if input_data is not None and i < num_steps:
            # cochleagram → スパイク確率へ変換
            prob = 1 / (1 + np.exp(-input_data[i]))  # sigmoid
            spike_in_h = (np.random.rand(N) < prob).astype(np.uint8)
        else:
            spike_in_h = np.zeros(N, dtype=np.uint8)
        cuda.memcpy_htod_async(spike_in_d.gpudata, spike_in_h, stream=stream3)

        stream3.synchronize()
        v[i] = Vs_h
        rasters[i] = rasters[i] | spike_in_h
        # stream2.synchronize()
        # stream1.synchronize()
        # input[i] = x_d.get()
        # v[i] = y_d.get()

        stream1.wait_for_event(event_update_input)

    end = time.perf_counter()

    print(
        f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}"
    )
    print(f"SEED value was {SEED}")

    v = v / 2**RSexci.BIT_WIDTH_FRACTIONAL
    # ---- plot simulation result ----
    plot_single_neuron(0, dt, tmax, num_steps, input, v, plot_num, label)
    plot_num += 1

    plot_raster(dt, tmax, rasters, N, plot_num)
    plot_num += 1

    # plt.show()


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


def create_moduled_matrix(N):
    resovoir_weight = np.zeros((N, N))
    crust_idx = 0
    G = 0.1
    p = 0.05
    while crust_idx != 4:
        i1 = int(crust_idx * N / 4)
        i2 = int((crust_idx + 1) * N / 4)
        resovoir_weight[i1:i2, i1:i2] = ((G * np.random.randn(N // 4, N // 4)) + 1) * (
            np.random.rand(N // 4, N // 4) < p
        )
        # print(resovoir_weight[i1:i2, i1:i2])
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
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
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
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            (G * np.random.randn(N // 4, N // 4)) + 1
        ) * (np.random.rand(N // 4, N // 4) < p)

    # 抑制結合の設定
    base_mask = np.ones((N, int(N / 4)))
    base_mask[:, N // 5 :] = -1
    mask = np.hstack([base_mask for _ in range(4)])
    resovoir_weight = resovoir_weight * mask
    # resovoir_weight = np.zeros((N, N))#test
    # resovoir_weight[0, 1] = 1       #test
    type = mask
    mask = (resovoir_weight != 0) * mask

    return resovoir_weight, mask, type


def create_random_matrix(N):
    resovoir_weight = np.zeros((N, N))
    G = 0.1
    p = 0.05
    resovoir_weight = ((G * np.random.randn(N, N)) + 1) * (np.random.rand(N, N) < p)

    # 抑制結合の設定
    mask = np.ones((N, N))
    mask[:, int(4 * N / 5) :] = -1
    resovoir_weight = resovoir_weight * mask
    # resovoir_weight = np.zeros((N, N))#test
    # resovoir_weight[0, 1] = 1       #test
    mask = (resovoir_weight != 0) * mask

    return resovoir_weight, mask


def synapses_init(resovoir_weight, N, N_S):
    tau_rec = np.full(N_S, 0.2, dtype=np.float32)
    tau_inact = np.full(N_S, 0.003, dtype=np.float32)
    tau_faci = np.full(N_S, 0.53, dtype=np.float32)
    U1 = np.full(N_S, 0.3, dtype=np.float32)
    U = np.full(N_S, 0.5, dtype=np.float32)
    mask_faci = np.zeros(N_S, dtype=np.uint8)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
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


def delay_init(resovoir_weight, N, N_S, mask):
    # delays = np.random.randint(100, 700, size=(N,N))
    # delays = np.full((N, N), 1000, dtype=np.int32)
    delays = (40 + 7.5 * np.random.randn(N, N)).astype(
        np.int32
    )  # 平均4ms 標準偏差0.75ms
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
    save_path = os.path.join("graphs", "resovoir_weight_matrix.png")
    plt.savefig(save_path)
    if record:
        save_path = os.path.join(OUTDIR, "resovoir_weight_matrix.png")
        plt.savefig(save_path)


def plot_single_neuron(id, dt, tmax, number_of_iterations, I, v0, num, label):
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
    save_path = os.path.join("graphs", "single_neuron.png")
    plt.savefig(save_path)
    if record:
        save_path = os.path.join(OUTDIR, f"single_neuron_{label}.png")
        plt.savefig(save_path)


def plot_raster(dt, tmax, rasters, N, num):
    times, neuron_ids = np.nonzero(rasters)
    times = times * dt
    neuron_ids = neuron_ids  # Adjust neuron IDs to start from 1
    cluster_colors = ["red", "blue", "green", "orange"]
    cluster_id = (neuron_ids) // (N // 4)  # 0,1,2,3 のクラスタID
    colors = [cluster_colors[c % 4] for c in cluster_id]
    plt.figure(num=num, figsize=(9, 5))
    plt.scatter(times, neuron_ids, s=0.1, color=colors)
    plt.xlabel("time")
    plt.xlim(0, tmax)
    plt.ylabel("neuron ID")
    plt.ylim(0, N)
    plt.title("Raster Plot")
    plt.tight_layout()
    save_path = os.path.join("graphs", "raster.png")
    plt.savefig(save_path)
    if record:
        save_path = os.path.join(OUTDIR, "raster.png")
        plt.savefig(save_path)


if __name__ == "__main__":
    # profiler = LineProfiler()
    # profiler.add_function(main)
    # profiler.runcall(main)
    # profiler.print_stats()

    coch = np.load("coch_zero.npy")

    main(input_data=coch, label="cochleagram")
