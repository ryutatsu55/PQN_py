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


SEED = int(random.random() * 1000)
# SEED = 678
random.seed(SEED) # for reproducibility
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
with open('my_kernel.cu', 'r', encoding='utf-8') as f:
    my_kernel_code = f.read()

# CuPyのRawKernelとしてカーネルをコンパイル（この行は変更なし）
module = SourceModule(my_kernel_code)
update_neuron_state = module.get_function('update_neuron_state')
copy_arrival_spike = module.get_function('copy_arrival_spike')
propagate_spikes = module.get_function('propagate_spikes')
synapses_calc = module.get_function('synapses_calc')
mat_vec_mul = module.get_function('mat_vec_mul')

# -------------------------------------------------------------
# 2. メイン処理
# -------------------------------------------------------------
def main():
    # --- 初期設定 ---
    N = 2
    tmax = 2    #[s]
    dt = 1e-4
    num_steps = int(tmax/dt)
    v = np.zeros((num_steps,N))
    input_h = np.zeros((num_steps,N), dtype=np.float32)
    buffer_size = 1001
    plot_num = 0

    v = np.zeros((num_steps, N))
    buffer_size = 1001
    plot_num = 0

    # PyCUDAでストリームとイベントを作成
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()
    stream3 = cuda.Stream()
    event_update_neuron = cuda.Event()
    event_update_input = cuda.Event()

    # 3. ホスト側(CPU)でデータを準備
    # Use a single PQN neuron model for all neurons (no RSexci/RSinhi branching)
    neuron_classes = ["RSexci","RSinhi","FS","LTS","IB","EB","PB","Class2"]
    neuron_mode = "FS"  # TODO: make this configurable (e.g., FS, LTS, ...)
    cell = PQNparam(mode=neuron_mode)
    cell_param_h = param_h_init(cell)
    dt = cell.PARAM['dt']

    Vs_h = np.full(N, cell.state_variable_v, dtype=np.int64)  # membrane potential
    Ns_h = np.full(N, cell.state_variable_n, dtype=np.int64)
    Qs_h = np.full(N, cell.state_variable_q, dtype=np.int64)
    Us_h = np.full(N, cell.state_variable_u, dtype=np.int64)
    raster_h = np.full(N, 0, dtype=np.uint8)
    spike_in_h = np.full(N, 0, dtype=np.uint8)


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

    Vs_d = gpuarray.to_gpu(Vs_h)
    Ns_d = gpuarray.to_gpu(Ns_h)
    Qs_d = gpuarray.to_gpu(Qs_h)
    Us_d = gpuarray.to_gpu(Us_h)

    synapses_out_d = gpuarray.zeros(N, dtype=np.float32)
    last_spike_d = gpuarray.zeros(N, dtype=np.uint8)
    raster_d = gpuarray.to_gpu(raster_h)

    # PyCUDAでストリームとイベントを作成
    stream1 = cuda.Stream()
    stream3 = cuda.Stream()
    event_update_neuron = cuda.Event()

    # 3. ホスト側(CPU)でデータを準備

    input_d = gpuarray.zeros(N, dtype=np.float32)

    # 5. カーネルの実行設定
    neuron_threads_per_block = 256
    neuron_blocks_per_grid = (N + neuron_threads_per_block - 1) // neuron_threads_per_block

    # 6. シミュレーションループ
    start = time.perf_counter()
    for i in tqdm(range(num_steps)):

        if ( 5000 < i & i < 15000 ):
            input_h[i] = np.float32(0.1)
        else:
            input_h[i] = np.float32(0.0)
        cuda.memcpy_htod(input_d.gpudata, input_h[i])

        update_neuron_state(            #stream1
            Vs_d.gpudata,
            Ns_d.gpudata,
            Qs_d.gpudata,
            Us_d.gpudata,
            input_d.gpudata,
            synapses_out_d.gpudata,
            last_spike_d.gpudata,
            raster_d.gpudata,
            np.int32(i),
            block=(neuron_threads_per_block, 1, 1),
            grid=(neuron_blocks_per_grid, 1),
            stream=stream1
        )
        event_update_neuron.record(stream1)
        
        stream3.wait_for_event(event_update_neuron)        
        cuda.memcpy_dtoh_async(Vs_h, Vs_d.gpudata, stream = stream3)
        
        stream3.synchronize()
        v[i] = Vs_h

    end = time.perf_counter()

    print(f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}")
    print(f"SEED value was {SEED}")

    v = v/2**cell.BIT_WIDTH_FRACTIONAL
    # ---- plot simulation result ----
    plot_single_neuron(0, dt, tmax, num_steps, input_h, v, plot_num)
    plot_num += 1

    # plt.show()


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

def plot_single_neuron(id, dt, tmax, number_of_iterations, I, v0, num):
    fig = plt.figure(num=num, figsize=(10,4))
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.1, height_ratios=[1, 4])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax0.set_xticks([])
    ax0.plot([i*dt for i in range(0, number_of_iterations)], I[:,id], color="black")
    ax1.plot([i*dt for i in range(0, number_of_iterations)], v0[:,id])    
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("v")
    ax0.set_ylabel("I")
    ax1.set_xlabel("[s]")
    save_path = os.path.join("graphs", "single_neuron.png")
    plt.savefig(save_path)
    if record:
        save_path = os.path.join(OUTDIR, "single_neuron.png")
        plt.savefig(save_path)

if __name__ == "__main__":
    # profiler = LineProfiler()
    # profiler.add_function(main)
    # profiler.runcall(main)
    # profiler.print_stats()
    main()
