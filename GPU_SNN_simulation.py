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
    # SEED = int(random.random() * 1000)
    SEED = 678
    random.seed(SEED) # for reproducibility
    np.random.seed(SEED)
    N = 1000
    tmax = 3    #[s]
    dt = 1e-4
    num_steps = int(tmax/dt)
    v = np.zeros((num_steps,N))
    rasters = np.zeros((num_steps,N), dtype=np.uint8)
    output = np.zeros((num_steps,N), dtype=np.float32)
    buffer_size = 13
    plot_num = 0

    # PyCUDAでストリームとイベントを作成
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()
    stream3 = cuda.Stream()
    event = cuda.Event()

    # 3. ホスト側(CPU)でデータを準備    
    Vs_h=np.full(N, -4906, dtype=np.int32)  # 各ニューロンの膜電位などの状態変数
    Ns_h=np.full(N, 27584, dtype=np.int32)
    Qs_h=np.full(N, -3692, dtype=np.int32)
    I_float_h=np.full((num_steps,N), 0, dtype=np.float32)
    for i in range(100):
        temp = 0.9*random.random()/dt
        I_float_h[int(temp):int(0.1/dt+temp),random.randint(0,N-1)] = 0.12
    # background noise
    for i in range(N):
        if i%(N//4) < N//5:
            I_float_h[:,i] += 1.5e-1*np.random.rand(num_steps)
        else:
            I_float_h[:,i] += 1.5e-1*np.random.rand(num_steps)
    I_float_h[int(num_steps/4):int(num_steps/4*3),:] = 0.09
    raster_h=np.full(N, 0, dtype=np.uint8)
    PARAM = PARAM_init()
    BIT_WIDTH=18#21
    BIT_WIDTH_FRACTIONAL=10#22
    BIT_Y_SHIFT=20#23
    Y=Y_init(PARAM,BIT_WIDTH_FRACTIONAL,BIT_Y_SHIFT)
    param_h = param_h_init(PARAM,Y,BIT_Y_SHIFT,BIT_WIDTH_FRACTIONAL)

    # ---- 重み行列の作成 ----
    resovoir_origin, mask = create_reservoir_matrix(N)
    resovoir_weight = np.copy(resovoir_origin)
    resovoir_weight = resovoir_weight*500/N
    visualize_matrix(resovoir_origin, plot_num)
    plot_num += 1

    N_S = np.int32(np.count_nonzero(resovoir_weight))

    tau_rec_h, tau_inact_h, tau_faci_h, U1_h, U_h, mask_faci_h = synapses_init(resovoir_weight, N, N_S)
    which_neuron_h, calc_matrix_h = calc_init(resovoir_weight, N, N_S)
    delayed_row_h = delay_init(resovoir_weight, N, N_S, mask)

    # 4. デバイス側(GPU)にメモリを確保し、データを転送
    Vs_d = gpuarray.to_gpu(Vs_h)
    Ns_d = gpuarray.to_gpu(Ns_h)
    Qs_d = gpuarray.to_gpu(Qs_h)
    I_float_d = gpuarray.to_gpu(I_float_h)
    param_d, param_d_size = module.get_global("const_param")
    cuda.memcpy_htod(param_d, param_h)
    x_d = gpuarray.ones(N_S, dtype=np.float32)
    y_d = gpuarray.zeros(N_S, dtype=np.float32)
    z_d = gpuarray.zeros(N_S, dtype=np.float32)
    tau_rec_d = gpuarray.to_gpu(tau_rec_h)
    tau_inact_d = gpuarray.to_gpu(tau_inact_h)
    tau_faci_d = gpuarray.to_gpu(tau_faci_h)
    U1_d = gpuarray.to_gpu(U1_h)
    U_d = gpuarray.to_gpu(U_h)
    mask_faci_d = gpuarray.to_gpu(mask_faci_h)
    which_neuron_d = gpuarray.to_gpu(which_neuron_h)
    calc_matrix_d = gpuarray.to_gpu(calc_matrix_h)
    delayed_row_d = gpuarray.to_gpu(delayed_row_h)
    last_spike_d = gpuarray.zeros(N, dtype=np.uint8)
    raster_d = gpuarray.to_gpu(raster_h)
    delayed_spikes_d = gpuarray.zeros((buffer_size, N_S), dtype=np.uint8)
    arrival_spike_d = gpuarray.zeros(N_S, dtype=np.uint8)
    synapses_out_d = gpuarray.zeros(N, dtype=np.float32)

    # 5. カーネルの実行設定
    neuron_threads_per_block = 256
    neuron_blocks_per_grid = (N + neuron_threads_per_block - 1) // neuron_threads_per_block
    synapse_threads_per_block =256
    synapse_blocks_per_grid = int((N_S + synapse_threads_per_block - 1) // synapse_threads_per_block)

    # 6. シミュレーションループ
    start = time.perf_counter()
    for i in tqdm(range(num_steps)):
    # for i in range(1000):
        read_idx = int(i%buffer_size)
        update_neuron_state(            #stream1
            Vs_d.gpudata,
            Ns_d.gpudata,
            Qs_d.gpudata,
            I_float_d.gpudata,
            synapses_out_d.gpudata,
            last_spike_d.gpudata,
            raster_d.gpudata,
            np.int32(N),
            np.int32(i),
            block=(neuron_threads_per_block, 1, 1),
            grid=(neuron_blocks_per_grid, 1),
            stream=stream1
        )
        event.record(stream1)
        propagate_spikes(               #stream1
            delayed_spikes_d.gpudata,
            raster_d.gpudata,
            which_neuron_d.gpudata,
            delayed_row_d.gpudata,
            np.uint32(N_S),
            np.uint32(read_idx),
            np.uint32(buffer_size),
            block=(synapse_threads_per_block, 1, 1), 
            grid=(synapse_blocks_per_grid, 1),
            stream=stream1
        )

        copy_arrival_spike(             #stream2
            arrival_spike_d.gpudata, 
            delayed_spikes_d.gpudata, 
            np.int32(read_idx),
            np.int32(N_S),
            block=(synapse_threads_per_block, 1, 1), 
            grid=(synapse_blocks_per_grid, 1),
            stream=stream2
        )
        synapses_calc(                   #stream2
            x_d.gpudata,
            y_d.gpudata,
            z_d.gpudata,
            arrival_spike_d.gpudata,
            mask_faci_d.gpudata,
            tau_rec_d.gpudata,
            tau_inact_d.gpudata,
            tau_faci_d.gpudata,
            U1_d.gpudata,
            U_d.gpudata,
            np.int32(N_S),
            np.float32(dt),
            block=(synapse_threads_per_block, 1, 1), 
            grid=(synapse_blocks_per_grid, 1),
            stream=stream2
        )
        # mat_vec_mul(                     #stream3
        #     synapses_out_d.gpudata,
        #     calc_matrix_d.gpudata,
        #     y_d.gpudata,
        #     np.int32(N),         # 行数
        #     np.int32(N_S),       # 列数
        #     block=(neuron_threads_per_block, 1, 1),
        #     grid=(neuron_blocks_per_grid, 1),
        #     stream=stream2
        # )
        
        stream3.wait_for_event(event)
        cuda.memcpy_dtoh_async(Vs_h, Vs_d.gpudata, stream = stream3)
        cuda.memcpy_dtoh_async(raster_h, raster_d.gpudata, stream = stream3)
        
        stream3.synchronize()
        v[i] = Vs_h
        rasters[i] = raster_h
        # if np.count_nonzero(delayed_spikes_d.get()) != 0:
        #     print(f"step {i}")
        #     print(f"read_idx: {read_idx}")
        #     print(raster_h)
        #     print(delayed_spikes_d.get())
        #     print(arrival_spike_d.get())
        #     print(y_d.get())
        #     print(synapses_out_d.get())
        #     print(np.dot(calc_matrix_h, y_d.get()))
        
        stream1.synchronize()
        stream2.synchronize()
        arrival_spike = arrival_spike_d.get()
        hoge = y_d.get()
        synapses_out_h = np.dot(calc_matrix_h, hoge)
        output[i] = synapses_out_h + I_float_h[i]
        synapses_out_d = gpuarray.to_gpu(synapses_out_h)
        # if arrival_spike[0] == 1:
        # # if raster_h[0] == 1:
        # # if synapses_out_h[0] != 0:
        #     print(hoge[0])
        #     print(synapses_out_h[0])
        #     print(i)

    end = time.perf_counter()

    # print(np.where(rasters[20000]==1))
    # print(delayed_row_h)
    # print(calc_matrix_h[0])
    print(f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}")
    print(f"SEED value was {SEED}")

    v = v/2**BIT_WIDTH_FRACTIONAL
    # ---- plot simulation result ----
    plot_single_neuron(0, dt, tmax, num_steps, output, v, plot_num)
    plot_num += 1

    plot_raster(dt, tmax, rasters, N, plot_num)
    plot_num += 1

    # plt.show()



def PARAM_init():
    PARAM={}
    PARAM['dt']=0.0001 # time step [s]  #1
    PARAM['afn']=1.5625#2
    PARAM['afp']=-0.5625#3
    PARAM['bfn']=-1.125#4
    PARAM['cfn']=0#5
    PARAM['agn']=1#6
    PARAM['agp']=10.28125#7
    PARAM['bgn']=0.40625#8
    PARAM['cgn']=0#9
    PARAM['ahn']=0.28125#10
    PARAM['ahp']=9.125#11
    PARAM['bhn']=-7.18753125#12
    PARAM['chn']=-2.8125#13
    PARAM['tau']=0.0064#14
    PARAM['I0']=2.375#15
    PARAM['k']=36.4375#16
    PARAM['phi']=4.75#17
    PARAM['epsq']=0.0693359375#18
    PARAM['rg']=0.0625#19
    PARAM['rh']=15.71875#20
    PARAM['bfp']=PARAM['afn']*PARAM['bfn']/PARAM['afp']#24
    PARAM['cfp']=PARAM['afn']*PARAM['bfn']**2+PARAM['cfn']-PARAM['afp']*PARAM['bfp']**2#25
    PARAM['bgp']=PARAM['rg']-PARAM['agn']*(PARAM['rg']-PARAM['bgn'])/PARAM['agp']#26
    PARAM['cgp']=PARAM['agn']*(PARAM['rg']-PARAM['bgn'])**2+PARAM['cgn']-PARAM['agp']*(PARAM['rg']-PARAM['bgp'])**2#27
    PARAM['bhp']=PARAM['rh']-PARAM['ahn']*(PARAM['rh']-PARAM['bhn'])/PARAM['ahp']#28
    PARAM['chp']=PARAM['ahn']*(PARAM['rh']-PARAM['bhn'])**2+PARAM['chn']-PARAM['ahp']*(PARAM['rh']-PARAM['bhp'])**2#29
    return PARAM

def Y_init(PARAM,BIT_WIDTH_FRACTIONAL,BIT_Y_SHIFT):
    f0=PARAM['dt']/PARAM['tau']*PARAM['phi']
    g0=PARAM['dt']/PARAM['tau']
    Y={}
    Y['v_vv_S']=int(f0*PARAM['afn']*2**BIT_Y_SHIFT)#30
    Y['v_vv_L']=int(f0*PARAM['afp']*2**BIT_Y_SHIFT)#31
    Y['v_v_S']=int(f0*(-2)*PARAM['afn']*PARAM['bfn']*2**BIT_Y_SHIFT)#32
    Y['v_v_L']=int(f0*(-2)*PARAM['afp']*PARAM['bfp']*2**BIT_Y_SHIFT)#33
    Y['v_c_S']=int((f0*(PARAM['afn']*PARAM['bfn']*PARAM['bfn']+PARAM['cfn']+PARAM['I0'])*2**BIT_WIDTH_FRACTIONAL))#34
    Y['v_c_L']=int((f0*(PARAM['afp']*PARAM['bfp']*PARAM['bfp']+PARAM['cfp']+PARAM['I0'])*2**BIT_WIDTH_FRACTIONAL))#35
    Y['v_n']=int(-f0*(2**BIT_Y_SHIFT))#36
    Y['v_q']=int(-f0*(2**BIT_Y_SHIFT))#37
    Y['v_I']=int(f0*PARAM['k']*2**BIT_Y_SHIFT)#38
    Y['n_vv_S']=int(g0*PARAM['agn']*2**BIT_Y_SHIFT)#39
    Y['n_vv_L']=int(g0*PARAM['agp']*2**BIT_Y_SHIFT)#40
    Y['n_v_S']=int(g0*(-2)*PARAM['agn']*PARAM['bgn']*2**BIT_Y_SHIFT)#41
    Y['n_v_L']=int(g0*(-2)*PARAM['agp']*PARAM['bgp']*2**BIT_Y_SHIFT)#42
    Y['n_n']=int(-g0*(2**BIT_Y_SHIFT))#43
    Y['n_c_S']=int((g0*(PARAM['agn']*PARAM['bgn']*PARAM['bgn']+PARAM['cgn'])*2**BIT_WIDTH_FRACTIONAL))#44
    Y['n_c_L']=int((g0*(PARAM['agp']*PARAM['bgp']*PARAM['bgp']+PARAM['cgp'])*2**BIT_WIDTH_FRACTIONAL))#45
    Y['rg']=int(PARAM['rg']*(2**BIT_WIDTH_FRACTIONAL))#46
    h0=PARAM['dt']/PARAM['tau']*PARAM['epsq']
    Y['q_vv_S']=int(h0*PARAM['ahn']*2**BIT_Y_SHIFT)#47
    Y['q_vv_L']=int(h0*PARAM['ahp']*2**BIT_Y_SHIFT)#48
    Y['q_v_S']=int(h0*(-2)*PARAM['ahn']*PARAM['bhn']*2**BIT_Y_SHIFT)#49
    Y['q_v_L']=int(h0*(-2)*PARAM['ahp']*PARAM['bhp']*2**BIT_Y_SHIFT)#50
    Y['q_q']=int(-h0*(2**BIT_Y_SHIFT))#51
    Y['q_c_S']=int((h0*(PARAM['ahn']*PARAM['bhn']*PARAM['bhn']+PARAM['chn'])*2**BIT_WIDTH_FRACTIONAL))#52
    Y['q_c_L']=int((h0*(PARAM['ahp']*PARAM['bhp']*PARAM['bhp']+PARAM['chp'])*2**BIT_WIDTH_FRACTIONAL))#53
    Y['rh']=int(PARAM['rh']*(2**BIT_WIDTH_FRACTIONAL))#54
    return Y

def param_h_init(PARAM,Y,BID_Y_SHIFT,BID_WIDTH_FRACTIONAL):
    param = np.zeros(27, dtype=np.int32)
    param[0]=BID_Y_SHIFT
    param[1]=BID_WIDTH_FRACTIONAL
    param[2]=Y['v_vv_S']
    param[3]=Y['v_vv_L']
    param[4]=Y['v_v_S']
    param[5]=Y['v_v_L']
    param[6]=Y['v_c_S']
    param[7]=Y['v_c_L']
    param[8]=Y['v_n']
    param[9]=Y['v_q']
    param[10]=Y['v_I']
    param[11]=Y['n_vv_S']
    param[12]=Y['n_vv_L']
    param[13]=Y['n_v_S']
    param[14]=Y['n_v_L']
    param[15]=Y['n_n']
    param[16]=Y['n_c_S']
    param[17]=Y['n_c_L']
    param[18]=Y['rg']
    param[19]=Y['q_vv_S']
    param[20]=Y['q_vv_L']
    param[21]=Y['q_v_S']
    param[22]=Y['q_v_L']
    param[23]=Y['q_q']
    param[24]=Y['q_c_S']
    param[25]=Y['q_c_L']
    param[26]=Y['rh']
    return param

def create_reservoir_matrix(N):
    resovoir_weight = np.zeros((N, N))
    crust_idx = 0
    G = 0.05
    p = 0.05
    while crust_idx != 4:
        i1 = int(crust_idx * N / 4)
        i2 = int((crust_idx + 1) * N / 4)
        resovoir_weight[i1:i2, i1:i2] = ((G * np.random.randn(N//4, N//4)) / (np.sqrt(N) * p) + 1) * (np.random.rand(N//4, N//4) < p)
        # print(resovoir_weight[i1:i2, i1:i2])
        crust_idx += 1

    #クラスター間の接続
    M = 4
    G = 0.001
    p = 0.001
    for hoge in range(M):
        i_range1 = int((hoge*N/4)%N)
        i_range2 = int((hoge+1)*N/4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int(((hoge+1)*N/4)%N)
        j_range2 = int((hoge+2)*N/4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = ((G * np.random.randn(N//4, N//4)) / (np.sqrt(N) * p) + 1) * (np.random.rand(N//4, N//4) < p)

        i_range1 = int(((hoge+1)*N/4)%N)
        i_range2 = int((hoge+2)*N/4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int((hoge*N/4)%N)
        j_range2 = int((hoge+1)*N/4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = ((G * np.random.randn(N//4, N//4)) / (np.sqrt(N) * p) + 1) * (np.random.rand(N//4, N//4) < p)

    # 抑制結合の設定
    base_mask = np.ones((N, int(N/4)))
    base_mask[:, N//5:] = -1
    mask = np.hstack([base_mask for _ in range(4)])
    resovoir_weight = resovoir_weight * mask
    # resovoir_weight = np.zeros((N, N))#test
    # resovoir_weight[0, 1] = 1       #test
    mask = (resovoir_weight != 0) * mask

    return resovoir_weight, mask

def synapses_init(resovoir_weight, N, N_S):
    tau_rec = np.full(N_S, 1.0, dtype=np.float32)
    tau_inact = np.full(N_S, 0.003, dtype=np.float32)
    tau_faci = np.full(N_S, 0.53, dtype=np.float32)
    U1 = np.full(N_S, 0.03, dtype=np.float32)
    U = np.full(N_S, 0.5, dtype=np.float32)
    mask_faci = np.zeros(N_S, dtype=np.uint8)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        if resovoir_weight[r, c] < 0:
            mask_faci[i] = 1
            U[i] = 0
        # if r == c and c%(N//4) < N//5:
        #     U[i] = 0.1
        #     tau_rec[i] = 0.1
    return tau_rec, tau_inact, tau_faci, U1, U, mask_faci

def calc_init(resovoir_weight, N, N_S):
    which_neuron = np.zeros(N_S, dtype=np.int32)
    resovoir_weight_calc = np.zeros((N, N_S), dtype=np.float32)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        which_neuron[i] = c
        resovoir_weight_calc[r, i] = resovoir_weight[r, c]
    return which_neuron, resovoir_weight_calc

def delay_init(resovoir_weight, N, N_S, mask):
    delays = np.random.randint(8, 13, size=(N,N))
    delays = delays * (mask != 0)
    delay_row = np.zeros(N_S, dtype=np.int32)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        delay_row[i] = delays[r, c]
    return delay_row

def visualize_matrix(matrix, num):
    plt.figure(num=num, figsize=(8, 6))
    max_abs = np.max(np.abs(matrix))
    im = plt.imshow(matrix, aspect='auto', cmap='plasma', vmin=-max_abs, vmax=max_abs)
    plt.gca().invert_yaxis()
    plt.colorbar(im, label='Weight Value')
    plt.title('Reservoir Weight Matrix')
    plt.xlabel('Pre Neuron')
    plt.ylabel('Post Neuron')
    plt.tight_layout()
    plt.savefig("resovoir_weight_matrix.png")

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
    fig.savefig("single_neuron.png")

def plot_raster(dt, tmax, rasters, N, num):
    times, neuron_ids = np.nonzero(rasters)
    times = times * dt
    neuron_ids = neuron_ids  # Adjust neuron IDs to start from 1
    cluster_colors = ['red', 'blue', 'green', 'orange']
    cluster_id = ((neuron_ids) // (N // 4))  # 0,1,2,3 のクラスタID
    colors = [cluster_colors[c % 4] for c in cluster_id]
    plt.figure(num=num, figsize=(9, 5))
    plt.scatter(times, neuron_ids, s=1, color=colors)
    plt.xlabel("time")
    plt.xlim(0, tmax)
    plt.ylabel("neuron ID")
    plt.ylim(0, N)
    plt.title("Raster Plot")
    plt.tight_layout()
    plt.savefig("raster.png")

if __name__ == "__main__":
    # profiler = LineProfiler()
    # profiler.add_function(main)
    # profiler.runcall(main)
    # profiler.print_stats()
    main()
