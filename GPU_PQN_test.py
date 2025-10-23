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

    # PyCUDAでストリームとイベントを作成
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()
    stream3 = cuda.Stream()
    event_update_neuron = cuda.Event()
    event_update_input = cuda.Event()

    # 3. ホスト側(CPU)でデータを準備    
    neuron_type_h = np.zeros(N, dtype=np.uint8)  # 各ニューロンのタイプ
    Vs_h=np.zeros(N, dtype=np.int64)  # 各ニューロンの膜電位などの状態変数
    Ns_h=np.zeros(N, dtype=np.int64)
    Qs_h=np.zeros(N, dtype=np.int64)
    raster_h=np.zeros(N, dtype=np.uint8)
    spike_in_h = np.zeros(N, dtype=np.uint8)

    RSexci = PQNparam(mode='RSexci')
    RSexci_param_h = param_h_init(RSexci)
    Vs_h[0]=RSexci.state_variable_v
    Ns_h[0]=RSexci.state_variable_n
    Qs_h[0]=RSexci.state_variable_q
    neuron_type_h[0] = 0
    RSinhi = PQNparam(mode='RSinhi')
    RSinhi_param_h = param_h_init(RSinhi)
    Vs_h[1]=RSinhi.state_variable_v
    Ns_h[1]=RSinhi.state_variable_n
    Qs_h[1]=RSinhi.state_variable_q
    neuron_type_h[1] = 1

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

    neuron_type_d = gpuarray.to_gpu(neuron_type_h)
    Vs_d = gpuarray.to_gpu(Vs_h)
    Ns_d = gpuarray.to_gpu(Ns_h)
    Qs_d = gpuarray.to_gpu(Qs_h)

    last_spike_d = gpuarray.zeros(N, dtype=np.uint8)
    raster_d = gpuarray.to_gpu(raster_h)
    spike_in_d = gpuarray.to_gpu(spike_in_h)
    input_d = gpuarray.zeros(N, dtype=np.float32)

    # 5. カーネルの実行設定
    neuron_threads_per_block = 256
    neuron_blocks_per_grid = (N + neuron_threads_per_block - 1) // neuron_threads_per_block

    # 6. シミュレーションループ
    start = time.perf_counter()
    for i in tqdm(range(num_steps)):

        if ( 5000 < i & i < 15000 ):
            input_h[i] = np.float32(0.09)
        else:
            input_h[i] = np.float32(0.0)
        cuda.memcpy_htod(input_d.gpudata, input_h[i])

        update_neuron_state(            #stream1
            Vs_d.gpudata,
            Ns_d.gpudata,
            Qs_d.gpudata,
            neuron_type_d.gpudata,
            RSexci_param_d.gpudata,
            RSinhi_param_d.gpudata,
            input_d.gpudata,
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

    v = v/2**RSexci.BIT_WIDTH_FRACTIONAL
    # ---- plot simulation result ----
    plot_single_neuron(0, dt, tmax, num_steps, input_h, v, plot_num)
    plot_num += 1

    # plt.show()


def param_h_init(PQN):
    if (PQN.mode in ['RSexci', 'RSinhi', 'FS', 'EB']):
        param = np.zeros(27, dtype=np.int32)
        param[0]=PQN.BIT_Y_SHIFT
        param[1]=PQN.BIT_WIDTH_FRACTIONAL
        param[2]=PQN.Y['v_vv_S']
        param[3]=PQN.Y['v_v_S']
        param[4]=PQN.Y['v_c_S']
        param[5]=PQN.Y['v_n']
        param[6]=PQN.Y['v_q']
        param[7]=PQN.Y['v_I']
        param[8]=PQN.Y['v_vv_L']
        param[9]=PQN.Y['v_v_L']
        param[10]=PQN.Y['v_c_L']
        param[11]=PQN.Y['rg']
        param[12]=PQN.Y['n_vv_S']
        param[13]=PQN.Y['n_v_S']
        param[14]=PQN.Y['n_c_S']
        param[15]=PQN.Y['n_n']
        param[16]=PQN.Y['n_vv_L']
        param[17]=PQN.Y['n_v_L']
        param[18]=PQN.Y['n_c_L']
        param[19]=PQN.Y['rh']
        param[20]=PQN.Y['q_vv_S']
        param[21]=PQN.Y['q_v_S']
        param[22]=PQN.Y['q_c_S']
        param[23]=PQN.Y['q_q']
        param[24]=PQN.Y['q_vv_L']
        param[25]=PQN.Y['q_v_L']
        param[26]=PQN.Y['q_c_L']
        return param
    elif (PQN.mode in ['LTS', 'IB']):
        param = np.zeros(27, dtype=np.int32)

        return param
    elif PQN.mode == 'PB':
        param = np.zeros(27, dtype=np.int32)

        return param
    elif PQN.mode == 'Class2':
        param = np.zeros(27, dtype=np.int32)

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
    ax0.set_xlim(0, tmax)
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
