import torch
import torch.nn as nn
import torch.optim as optim
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random

# all_rasters_h = np.zeros(self.N, dtype=np.uint8)
# all_rasters_d = gpuarray.to_gpu(all_rasters_h)

class SNNReservoirSimulator:
    def __init__(self, N=1000, seed=int(random.random() * 1000)):
        with open('../my_kernel.cu', 'r', encoding='utf-8') as f:
            my_kernel_code = f.read()
        self.module = SourceModule(my_kernel_code)
        self.update_neuron_state = self.module.get_function('update_neuron_state')
        self.copy_arrival_spike = self.module.get_function('copy_arrival_spike')
        self.propagate_spikes = self.module.get_function('propagate_spikes')
        self.synapses_calc = self.module.get_function('synapses_calc')
        self.mat_vec_mul = self.module.get_function('mat_vec_mul')
        
        self.N = N
        self.SEED = seed
        # self.tmax = 10    #[s]
        self.dt = 1e-4
        self.read_idx = 0
        # self.num_steps = int(self.tmax/self.dt)
        # self.rasters = np.zeros((self.num_steps,N), dtype=np.uint8)
        self.buffer_size = 1001
        self.window_size = 1000
        self.plot_num = 0

        # PyCUDAでストリームとイベントを作成
        self.stream1 = cuda.Stream()
        self.stream2 = cuda.Stream()
        self.stream3 = cuda.Stream()
        self.event_update_neuron = cuda.Event()
        self.event_update_input = cuda.Event()
        self.event_update_recurrent = cuda.Event()
        self.event_update_buffer = cuda.Event()

        # ホスト側(CPU)でデータを準備    
        self.Vs_h=np.full(N, -4906, dtype=np.int64)  # 各ニューロンの膜電位などの状態変数
        self.Ns_h=np.full(N, 27584, dtype=np.int64)
        self.Qs_h=np.full(N, -3692, dtype=np.int64)
        
        self.raster_h=np.full(N, 0, dtype=np.uint8)
        self.spike_in_h = np.full(N, 0, dtype=np.uint8)
        self.PARAM = self.PARAM_init()
        self.BIT_WIDTH=18#21
        self.BIT_WIDTH_FRACTIONAL=10#22
        self.BIT_Y_SHIFT=20#23
        self.Y = self.Y_init()
        self.PQN_param_h = self.param_h_init()

        
        # ---- 重み行列の作成 ----
        self.resovoir_origin, self.mask = self.create_moduled_matrix()
        # self.resovoir_origin, self.mask = self.create_random_matrix(self.N)
        self.resovoir_weight = np.copy(self.resovoir_origin)
        self.resovoir_weight = self.resovoir_weight*0.003
        # self.visualize_matrix(self.resovoir_origin, self.plot_num)
        self.plot_num += 1

        self.N_S = np.count_nonzero(self.resovoir_weight)

        self.tau_rec_h = np.full(self.N_S, 0.2, dtype=np.float32)
        self.tau_inact_h = np.full(self.N_S, 0.003, dtype=np.float32)
        self.tau_faci_h = np.full(self.N_S, 0.53, dtype=np.float32)
        self.U1_h = np.full(self.N_S, 0.3, dtype=np.float32)
        self.U_h = np.full(self.N_S, 0.5, dtype=np.float32)
        self.mask_faci_h = np.zeros(self.N_S, dtype=np.uint8)
        self.td_float32 = np.float32(1e-2)
        self.tr_float32 = np.float32(5e-3)
        self.synapses_init()
        self.neuron_from_h = np.zeros(self.N_S, dtype=np.int32)
        self.calc_matrix_h = np.zeros(self.N_S, dtype=np.float32)
        self.neuron_to_h = np.zeros(self.N_S, dtype=np.int32)
        self.calc_init()
        self.delayed_row_h = self.delay_init()

        # 4. デバイス側(GPU)にメモリを確保し、データを転送
        self.PQN_param_d, self.PQN_param_d_size = self.module.get_global("PQN_param")
        cuda.memcpy_htod(self.PQN_param_d, self.PQN_param_h)

        self.dt_float32 = np.float32(self.dt)
        self.dt_d, self.dt_d_size = self.module.get_global("dt")
        cuda.memcpy_htod(self.dt_d, self.dt_float32)

        self.buffer_size_int32 = np.int32(self.buffer_size)
        self.buffer_size_d, self.buffer_size_d_size = self.module.get_global("buffer_size")
        cuda.memcpy_htod(self.buffer_size_d, self.buffer_size_int32)

        self.n_int32 = np.int32(self.N)
        self.n_d, self.n_d_size = self.module.get_global("num_neurons")
        cuda.memcpy_htod(self.n_d, self.n_int32)

        self.n_s_int32 = np.int32(self.N_S)
        self.ns_d, self.ns_d_size = self.module.get_global("num_synapses")
        cuda.memcpy_htod(self.ns_d, self.n_s_int32)

        self.Vs_d = gpuarray.to_gpu(self.Vs_h)
        self.Ns_d = gpuarray.to_gpu(self.Ns_h)
        self.Qs_d = gpuarray.to_gpu(self.Qs_h)

        self.x_h=np.full(self.N_S, 0.9, dtype=np.float32)
        self.z_h=np.full(self.N_S, 0.1, dtype=np.float32)
        self.x_d = gpuarray.to_gpu(self.x_h)
        self.y_d = gpuarray.zeros(self.N_S, dtype=np.float32)
        self.z_d = gpuarray.to_gpu(self.z_h)
        self.r_d = gpuarray.zeros(self.N_S, dtype=np.float32)
        self.hr_d = gpuarray.zeros(self.N_S, dtype=np.float32)
        self.tau_rec_d = gpuarray.to_gpu(self.tau_rec_h)
        self.tau_inact_d = gpuarray.to_gpu(self.tau_inact_h)
        self.tau_faci_d = gpuarray.to_gpu(self.tau_faci_h)
        self.U1_d = gpuarray.to_gpu(self.U1_h)
        self.U_d = gpuarray.to_gpu(self.U_h)
        self.mask_faci_d = gpuarray.to_gpu(self.mask_faci_h)
        self.neuron_from_d = gpuarray.to_gpu(self.neuron_from_h)
        self.calc_matrix_d = gpuarray.to_gpu(self.calc_matrix_h)
        self.neuron_to_d = gpuarray.to_gpu(self.neuron_to_h)
        self.delayed_row_d = gpuarray.to_gpu(self.delayed_row_h)
        self.last_spike_d = gpuarray.zeros(N, dtype=np.uint8)
        self.raster_d = gpuarray.to_gpu(self.raster_h)
        self.delayed_spikes_d = gpuarray.zeros((self.buffer_size, self.N_S), dtype=np.uint8)
        self.arrival_spike_d = gpuarray.zeros(self.N_S, dtype=np.uint8)
        self.spike_in_d = gpuarray.to_gpu(self.spike_in_h)
        self.synapses_out_d = gpuarray.zeros(self.N, dtype=np.float32)

        # 5. カーネルの実行設定
        self.neuron_threads_per_block = 256
        self.neuron_blocks_per_grid = (self.N + self.neuron_threads_per_block - 1) // self.neuron_threads_per_block
        self.synapse_threads_per_block =256
        self.synapse_blocks_per_grid = int((self.N_S + self.synapse_threads_per_block - 1) // self.synapse_threads_per_block)

    def forward(self, num_sim_steps=100, p=0.1):
        spike_in_h = (np.random.rand(self.N) < p).astype(np.uint8)
        cuda.memcpy_htod_async(self.spike_in_d.gpudata, spike_in_h, stream = self.stream3)
        self.event_update_input.record(self.stream3)
        
        for i in range(num_sim_steps):
            self.read_idx = np.int32(self.read_idx%self.buffer_size)
            self.update_neuron_state(            #stream1
                self.Vs_d.gpudata,
                self.Ns_d.gpudata,
                self.Qs_d.gpudata,
                self.synapses_out_d.gpudata,
                self.last_spike_d.gpudata,
                self.raster_d.gpudata,
                np.int32(i),
                block=(self.neuron_threads_per_block, 1, 1),
                grid=(self.neuron_blocks_per_grid, 1),
                stream=self.stream1
            )
            self.event_update_neuron.record(self.stream1)
            self.stream1.wait_for_event(self.event_update_input)
            self.propagate_spikes(               #stream1
                self.delayed_spikes_d.gpudata,
                self.raster_d.gpudata,
                self.spike_in_d.gpudata,
                self.neuron_from_d.gpudata,
                self.delayed_row_d.gpudata,
                self.read_idx,
                block=(self.synapse_threads_per_block, 1, 1), 
                grid=(self.synapse_blocks_per_grid, 1),
                stream=self.stream1
            )
            self.event_update_buffer.record(self.stream1)
            
            self.synapses_calc(                   #stream2
                self.x_d.gpudata,
                self.y_d.gpudata,
                self.z_d.gpudata,
                self.r_d.gpudata,
                self.hr_d.gpudata,
                self.delayed_spikes_d.gpudata,
                self.mask_faci_d.gpudata,
                self.tau_rec_d.gpudata,
                self.tau_inact_d.gpudata,
                self.tau_faci_d.gpudata,
                self.U1_d.gpudata,
                self.U_d.gpudata,
                self.td_float32,
                self.tr_float32,
                self.read_idx,
                block=(self.synapse_threads_per_block, 1, 1), 
                grid=(self.synapse_blocks_per_grid, 1),
                stream=self.stream2
            )
            self.stream2.wait_for_event(self.event_update_neuron)
            self.mat_vec_mul(                     #stream2
                self.synapses_out_d.gpudata,
                self.neuron_to_d.gpudata,
                self.calc_matrix_d.gpudata,
                self.r_d.gpudata,
                block=(self.synapse_threads_per_block, 1, 1),
                grid=(self.synapse_blocks_per_grid, 1),
                stream = self.stream2
            )
            self.event_update_recurrent.record(self.stream2)
            
            self.stream3.wait_for_event(self.event_update_buffer)  
            if ( (i+1) % 10000 == 0 ):
                spike_in_h = (np.random.rand(self.N) < p).astype(np.uint8)
                # spike_in_h = np.ones(N, dtype=np.uint8)
            else:
                spike_in_h = np.zeros(self.N, dtype=np.uint8)
            cuda.memcpy_htod_async(self.spike_in_d.gpudata, spike_in_h, stream = self.stream3)
            self.event_update_input.record(self.stream3)

            # 各ステップのラスターを記録
            # all_rasters_d = all_rasters_d + self.raster_d
            
            self.stream1.wait_for_event(self.event_update_recurrent)

            self.read_idx += 1

  
        # all_rasters_h = all_rasters_d.get()
        # reservoir_state = np.float32(all_rasters_h/num_sim_steps)
        # return reservoir_state
    
    def get_readout_state(self):
        """
        現在のリザバーの状態を読み出す。
        例: 直近 `window_size` ステップの平均発火率を計算する
        """
        # rastersのリングバッファなどから直近の活動履歴を取得し、
        # 平均発火率を計算して返す。
        # (この部分は少し工夫が必要)
        # 最も簡単な実装は、現在のラスター(発火の有無)をそのまま返すこと。
        current_raster_h = self.raster_d.get()
        return current_raster_h.astype(np.float32)
    

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

    def Y_init(self):
        f0=self.PARAM['dt']/self.PARAM['tau']*self.PARAM['phi']
        g0=self.PARAM['dt']/self.PARAM['tau']
        Y={}
        Y['v_vv_S']=int(f0*self.PARAM['afn']*2**self.BIT_Y_SHIFT)#30
        Y['v_vv_L']=int(f0*self.PARAM['afp']*2**self.BIT_Y_SHIFT)#31
        Y['v_v_S']=int(f0*(-2)*self.PARAM['afn']*self.PARAM['bfn']*2**self.BIT_Y_SHIFT)#32
        Y['v_v_L']=int(f0*(-2)*self.PARAM['afp']*self.PARAM['bfp']*2**self.BIT_Y_SHIFT)#33
        Y['v_c_S']=int((f0*(self.PARAM['afn']*self.PARAM['bfn']*self.PARAM['bfn']+self.PARAM['cfn']+self.PARAM['I0'])*2**self.BIT_WIDTH_FRACTIONAL))#34
        Y['v_c_L']=int((f0*(self.PARAM['afp']*self.PARAM['bfp']*self.PARAM['bfp']+self.PARAM['cfp']+self.PARAM['I0'])*2**self.BIT_WIDTH_FRACTIONAL))#35
        Y['v_n']=int(-f0*(2**self.BIT_Y_SHIFT))#36
        Y['v_q']=int(-f0*(2**self.BIT_Y_SHIFT))#37
        Y['v_I']=int(f0*self.PARAM['k']*2**self.BIT_Y_SHIFT)#38
        Y['n_vv_S']=int(g0*self.PARAM['agn']*2**self.BIT_Y_SHIFT)#39
        Y['n_vv_L']=int(g0*self.PARAM['agp']*2**self.BIT_Y_SHIFT)#40
        Y['n_v_S']=int(g0*(-2)*self.PARAM['agn']*self.PARAM['bgn']*2**self.BIT_Y_SHIFT)#41
        Y['n_v_L']=int(g0*(-2)*self.PARAM['agp']*self.PARAM['bgp']*2**self.BIT_Y_SHIFT)#42
        Y['n_n']=int(-g0*(2**self.BIT_Y_SHIFT))#43
        Y['n_c_S']=int((g0*(self.PARAM['agn']*self.PARAM['bgn']*self.PARAM['bgn']+self.PARAM['cgn'])*2**self.BIT_WIDTH_FRACTIONAL))#44
        Y['n_c_L']=int((g0*(self.PARAM['agp']*self.PARAM['bgp']*self.PARAM['bgp']+self.PARAM['cgp'])*2**self.BIT_WIDTH_FRACTIONAL))#45
        Y['rg']=int(self.PARAM['rg']*(2**self.BIT_WIDTH_FRACTIONAL))#46
        h0=self.PARAM['dt']/self.PARAM['tau']*self.PARAM['epsq']
        Y['q_vv_S']=int(h0*self.PARAM['ahn']*2**self.BIT_Y_SHIFT)#47
        Y['q_vv_L']=int(h0*self.PARAM['ahp']*2**self.BIT_Y_SHIFT)#48
        Y['q_v_S']=int(h0*(-2)*self.PARAM['ahn']*self.PARAM['bhn']*2**self.BIT_Y_SHIFT)#49
        Y['q_v_L']=int(h0*(-2)*self.PARAM['ahp']*self.PARAM['bhp']*2**self.BIT_Y_SHIFT)#50
        Y['q_q']=int(-h0*(2**self.BIT_Y_SHIFT))#51
        Y['q_c_S']=int((h0*(self.PARAM['ahn']*self.PARAM['bhn']*self.PARAM['bhn']+self.PARAM['chn'])*2**self.BIT_WIDTH_FRACTIONAL))#52
        Y['q_c_L']=int((h0*(self.PARAM['ahp']*self.PARAM['bhp']*self.PARAM['bhp']+self.PARAM['chp'])*2**self.BIT_WIDTH_FRACTIONAL))#53
        Y['rh']=int(self.PARAM['rh']*(2**self.BIT_WIDTH_FRACTIONAL))#54
        return Y

    def param_h_init(self):
        param = np.zeros(27, dtype=np.int32)
        param[0]=self.BIT_Y_SHIFT
        param[1]=self.BIT_WIDTH_FRACTIONAL
        param[2]=self.Y['v_vv_S']
        param[3]=self.Y['v_vv_L']
        param[4]=self.Y['v_v_S']
        param[5]=self.Y['v_v_L']
        param[6]=self.Y['v_c_S']
        param[7]=self.Y['v_c_L']
        param[8]=self.Y['v_n']
        param[9]=self.Y['v_q']
        param[10]=self.Y['v_I']
        param[11]=self.Y['n_vv_S']
        param[12]=self.Y['n_vv_L']
        param[13]=self.Y['n_v_S']
        param[14]=self.Y['n_v_L']
        param[15]=self.Y['n_n']
        param[16]=self.Y['n_c_S']
        param[17]=self.Y['n_c_L']
        param[18]=self.Y['rg']
        param[19]=self.Y['q_vv_S']
        param[20]=self.Y['q_vv_L']
        param[21]=self.Y['q_v_S']
        param[22]=self.Y['q_v_L']
        param[23]=self.Y['q_q']
        param[24]=self.Y['q_c_S']
        param[25]=self.Y['q_c_L']
        param[26]=self.Y['rh']
        return param

    def create_moduled_matrix(self):
        resovoir_weight = np.zeros((self.N, self.N))
        crust_idx = 0
        G = 0.1
        p = 0.05
        while crust_idx != 4:
            i1 = int(crust_idx * self.N / 4)
            i2 = int((crust_idx + 1) * self.N / 4)
            resovoir_weight[i1:i2, i1:i2] = ((G * np.random.randn(self.N//4, self.N//4)) + 1) * (np.random.rand(self.N//4, self.N//4) < p)
            # print(resovoir_weight[i1:i2, i1:i2])
            crust_idx += 1

        #クラスター間の接続
        M = 4
        G = 0.1
        p = 0.01
        for hoge in range(M):
            i_range1 = int((hoge*self.N/4)%self.N)
            i_range2 = int((hoge+1)*self.N/4)
            if i_range2 > self.N:
                i_range2 = i_range2 % self.N
            j_range1 = int(((hoge+1)*self.N/4)%self.N)
            j_range2 = int((hoge+2)*self.N/4)
            if j_range2 > self.N:
                j_range2 = j_range2 % self.N
            resovoir_weight[i_range1:i_range2, j_range1:j_range2] = ((G * np.random.randn(self.N//4, self.N//4)) + 1) * (np.random.rand(self.N//4, self.N//4) < p)

            i_range1 = int(((hoge+1)*self.N/4)%self.N)
            i_range2 = int((hoge+2)*self.N/4)
            if i_range2 > self.N:
                i_range2 = i_range2 % self.N
            j_range1 = int((hoge*self.N/4)%self.N)
            j_range2 = int((hoge+1)*self.N/4)
            if j_range2 > self.N:
                j_range2 = j_range2 % self.N
            resovoir_weight[i_range1:i_range2, j_range1:j_range2] = ((G * np.random.randn(self.N//4, self.N//4)) + 1) * (np.random.rand(self.N//4, self.N//4) < p)

        # 抑制結合の設定
        base_mask = np.ones((self.N, int(self.N/4)))
        base_mask[:, self.N//5:] = -1
        mask = np.hstack([base_mask for _ in range(4)])
        resovoir_weight = resovoir_weight * mask
        # resovoir_weight = np.zeros((N, N))#test
        # resovoir_weight[0, 1] = 1       #test
        mask = (resovoir_weight != 0) * mask

        return resovoir_weight, mask

    def create_random_matrix(self):
        resovoir_weight = np.zeros((self.N, self.N))
        G = 0.1
        p = 0.05
        resovoir_weight = ((G * np.random.randn(self.N, self.N)) + 1) * (np.random.rand(self.N, self.N) < p)

        # 抑制結合の設定
        mask = np.ones((self.N, self.N))
        mask[:, int(4*self.N/5):] = -1
        resovoir_weight = resovoir_weight * mask
        # resovoir_weight = np.zeros((N, N))#test
        # resovoir_weight[0, 1] = 1       #test
        mask = (resovoir_weight != 0) * mask

        return resovoir_weight, mask

    def synapses_init(self):
        col_indices, row_indices = np.where(self.resovoir_weight.T != 0)
        for i in range(self.N_S):
            r = row_indices[i]
            c = col_indices[i]
            if r%(self.N//4) > self.N//5:
                self.mask_faci[i] = 1
                self.U[i] = 0
                self.tau_rec[i] = 0.1
                self.tau_inact[i] = 0.0015
            # if r == c and c%(self.N//4) < self.N//5:
            #     U[i] = 0.1
            #     tau_rec[i] = 0.1

    def calc_init(self):
        col_indices, row_indices = np.where(self.resovoir_weight.T != 0)
        for i in range(self.N_S):
            r = row_indices[i]
            c = col_indices[i]
            self.neuron_from[i] = c
            self.resovoir_weight_calc[i] = self.resovoir_weight[r, c]
            self.neuron_to[i] = r

    def delay_init(self):
        delays = np.random.randint(100, 700, size=(self.N,self.N))
        # delays = np.full((self.N, self.N), 1000, dtype=np.int32)
        # delays = (40 + 7.5*np.random.randn(self.N, self.N)).astype(np.int32)
        delays = delays * (self.mask != 0)
        delay_row = np.zeros(self.N_S, dtype=np.int32)
        col_indices, row_indices = np.where(self.resovoir_weight.T != 0)
        for i in range(self.N_S):
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
        plt.scatter(times, neuron_ids, s=0.1, color=colors)
        plt.xlabel("time")
        plt.xlim(0, tmax)
        plt.ylabel("neuron ID")
        plt.ylim(0, N)
        plt.title("Raster Plot")
        plt.tight_layout()
        plt.savefig("raster.png")
