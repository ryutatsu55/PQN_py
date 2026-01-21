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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.PQN import PQNparam

# -------------------------------------------------------------
# 1. 外部の .cu ファイルを読み込んで文字列として取得
# -------------------------------------------------------------
with open("src/my_kernel.cu", "r", encoding="utf-8") as f:
    my_kernel_code = f.read()

# CuPyのRawKernelとしてカーネルをコンパイル
module = SourceModule(my_kernel_code)
update_neuron_state = module.get_function("update_neuron_state")
copy_arrival_spike = module.get_function("copy_arrival_spike")
propagate_spikes = module.get_function("propagate_spikes")
synapses_calc = module.get_function("synapses_calc")
mat_vec_mul = module.get_function("mat_vec_mul")

# -------------------------------------------------------------
# 2. Reservoir クラス定義
# -------------------------------------------------------------
class PQN_Reservoir_GPU:
    def __init__(self, buffer_size=10001, seed=None):
        """
        GPUメモリの確保と静的パラメータの初期化を行う
        """
        self.buffer_size = buffer_size
        self.N = 1
        self.N_S = 1

        # --- ホスト側(CPU) パラメータ準備 ---
        self.neuron_type_h = [0]
        self.dt = 1e-3 if self.neuron_type_h[0] == 6 else 1e-4

        # PQNパラメータ初期化 (Excitatory / Inhibitory)
        self.RSexci = PQNparam(mode="RSexci")
        self.RSinhi = PQNparam(mode="RSinhi")
        self.FS = PQNparam(mode="FS")
        self.LTS = PQNparam(mode="LTS")
        self.IB = PQNparam(mode="IB")
        self.EB = PQNparam(mode="EB")
        self.PB = PQNparam(mode="PB")
        self.RSexci_param_h = self._param_h_init(self.RSexci)
        self.RSinhi_param_h = self._param_h_init(self.RSinhi)
        self.FS_param_h = self._param_h_init(self.FS)
        self.LTS_param_h = self._param_h_init(self.LTS)
        self.IB_param_h = self._param_h_init(self.IB)
        self.EB_param_h = self._param_h_init(self.EB)
        self.PB_param_h = self._param_h_init(self.PB)

        # 定数パラメータのGPU転送 (Globalメモリ)
        self._upload_constants()

        self.neuron_type_d = gpuarray.to_gpu(np.array(self.neuron_type_h, dtype=np.uint8))
        self.tau_rec_d = gpuarray.to_gpu(np.full(self.N, 0.5, dtype=np.float32))
        self.tau_inact_d = gpuarray.to_gpu(np.full(self.N, 0.3, dtype=np.float32))
        self.tau_faci_d = gpuarray.to_gpu(np.full(self.N, 0.53, dtype=np.float32))
        self.U1_d = gpuarray.to_gpu(np.full(self.N, 0.05, dtype=np.float32))
        self.U_d = gpuarray.to_gpu(np.full(self.N, 0.1, dtype=np.float32))
        self.mask_faci_d = gpuarray.to_gpu(np.zeros(self.N, dtype=np.uint8))
        self.neuron_from_d = gpuarray.to_gpu(np.zeros(self.N_S, dtype=np.int32))
        self.neuron_to_d = gpuarray.to_gpu(np.zeros(self.N_S, dtype=np.int32))
        self.calc_matrix_d = gpuarray.to_gpu(np.full(self.N_S, 0.0, dtype=np.float32))
        self.delayed_row_d = gpuarray.to_gpu(np.full(self.N_S, 1000, dtype=np.int32))

        # カーネル実行設定
        self.neuron_threads = 256
        self.neuron_blocks = int((self.N + 255) // 256)
        self.synapse_threads = 256
        self.synapse_blocks = int((self.N_S + 255) // 256)

        # ストリーム作成
        self.stream1 = cuda.Stream()
        self.stream2 = cuda.Stream()
        self.stream3 = cuda.Stream()
        self.evt_update_neuron = cuda.Event()
        self.evt_spike_written = cuda.Event()
        self.evt_update_input = cuda.Event()

        # 動的変数の初期化
        self.reset_state()

        # ログ用変数のプレースホルダー
        self.v_int = None
        self.raster_log = None
        self.I_input_log = None

    def _param_h_init(self, PQN):
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

    def _upload_constants(self):
        # Global変数への転送
        RSexci_param_d, _ = module.get_global("RSexci_param")
        cuda.memcpy_htod(RSexci_param_d, self.RSexci_param_h)
        RSinhi_param_d, _ = module.get_global("RSinhi_param")
        cuda.memcpy_htod(RSinhi_param_d, self.RSinhi_param_h)
        FS_param_d, _ = module.get_global("FS_param")
        cuda.memcpy_htod(FS_param_d, self.FS_param_h)
        LTS_param_d, _ = module.get_global("LTS_param")
        cuda.memcpy_htod(LTS_param_d, self.LTS_param_h)
        IB_param_d, _ = module.get_global("IB_param")
        cuda.memcpy_htod(IB_param_d, self.IB_param_h)
        EB_param_d, _ = module.get_global("EB_param")
        cuda.memcpy_htod(EB_param_d, self.EB_param_h)
        PB_param_d, _ = module.get_global("PB_param")
        cuda.memcpy_htod(PB_param_d, self.PB_param_h)

        dt_float32 = np.float32(self.dt)
        dt_d, _ = module.get_global("dt")
        cuda.memcpy_htod(dt_d, dt_float32)

        buffer_size_int32 = np.int32(self.buffer_size)
        buffer_size_d, _ = module.get_global("buffer_size")
        cuda.memcpy_htod(buffer_size_d, buffer_size_int32)

        n_int32 = np.int32(self.N)
        n_d, _ = module.get_global("num_neurons")
        cuda.memcpy_htod(n_d, n_int32)

        ns_int32 = np.int32(self.N_S)
        ns_d, _ = module.get_global("num_synapses")
        cuda.memcpy_htod(ns_d, ns_int32)

    def reset_state(self):
        """
        動的な状態変数（膜電位、スパイク履歴など）を初期化する。
        重みなどの静的なものは維持する。
        """
        self.step_count = 0

        neuron_models = [
            self.RSexci, # 0
            self.RSinhi, # 1
            self.FS,     # 2
            self.LTS,    # 3
            self.IB,     # 4
            self.EB,     # 5
            self.PB      # 6
        ]
        init_vs = np.array([m.state_variable_v for m in neuron_models], dtype=np.int64)
        init_ns = np.array([m.state_variable_n for m in neuron_models], dtype=np.int64)
        init_qs = np.array([m.state_variable_q for m in neuron_models], dtype=np.int64)
        init_us = np.array([m.state_variable_u for m in neuron_models], dtype=np.int64)
        Vs_h = np.zeros(self.N, dtype=np.int64)
        Ns_h = np.zeros(self.N, dtype=np.int64)
        Qs_h = np.zeros(self.N, dtype=np.int64)
        Us_h = np.zeros(self.N, dtype=np.int64)
        Vs_h = init_vs[self.neuron_type_h]
        Ns_h = init_ns[self.neuron_type_h]
        Qs_h = init_qs[self.neuron_type_h]
        Us_h = init_us[self.neuron_type_h]
        # GPU転送 (Dynamic Variables)
        self.Vs_d = gpuarray.to_gpu(Vs_h)
        self.Ns_d = gpuarray.to_gpu(Ns_h)
        self.Qs_d = gpuarray.to_gpu(Qs_h)
        self.Us_d = gpuarray.to_gpu(Us_h)
        
        # シナプス状態
        self.x_d = gpuarray.to_gpu(np.full(self.N_S, 1.0, dtype=np.float32))
        self.y_d = gpuarray.zeros(self.N_S, dtype=np.float32)
        self.z_d = gpuarray.zeros(self.N_S, dtype=np.float32)
        self.r_d = gpuarray.zeros(self.N_S, dtype=np.float32)
        self.hr_d = gpuarray.zeros(self.N_S, dtype=np.float32)

        # スパイク関連
        self.last_spike_d = gpuarray.zeros(self.N, dtype=np.uint8)
        self.raster_d = gpuarray.zeros(self.N, dtype=np.uint8)
        self.delayed_spikes_d = gpuarray.zeros((self.buffer_size, self.N_S), dtype=np.uint8)
        self.spike_in_d = gpuarray.zeros(self.N, dtype=np.uint8)
        self.synapses_out_d = gpuarray.zeros(self.N, dtype=np.float32)
        self.I_input_d = gpuarray.zeros(self.N, dtype=np.float32)

    def run(self, num_steps, input_data):
        """
        Parameters:
            input_data: np.ndarray or None
                shape = (T, N_in)
                cochleagram or other time-series input.
        """
        """
        指定ステップ数（または入力データの長さ）だけシミュレーションを実行する。
        （従来の main() 関数相当のロジック）
        """
        # 結果格納用配列
        if (self.v_int is not None) and (self.v_int.shape == (num_steps, self.N)):
            self.v_int.fill(0)  # メモリ再利用
            self.v_log.fill(0.0)
            self.raster_log.fill(0)
            self.I_input_log.fill(0.0)
        else:
            # サイズが違う、または初回の場合だけ新規作成
            self.v_int = np.zeros((num_steps, self.N), dtype=np.int64)
            self.v_log = np.zeros((num_steps, self.N), dtype=np.float32)
            self.raster_log = np.zeros((num_steps, self.N), dtype=np.uint8)
            self.I_input_log = np.zeros((num_steps, self.N), dtype=np.float32)
        
        
        # ループ実行
        iter_range = tqdm(range(num_steps))
        self.iter_count = 0

        for t in iter_range:
            self.step(input_data=input_data)
            self.iter_count += 1


        self.v_log = (self.v_int - self.v_int[0,:]) / (2**self.RSexci.BIT_WIDTH_FRACTIONAL)


        # 結果を辞書などで返す
        results = {}
        results["rasters"] = self.raster_log
        results["input"] = self.I_input_log
        results["v"] = self.v_log
        
        return results

    def step(self, input_data):
        """
        1ステップ分の計算を実行する
        
        Parameters:
            prob_input_vector: np.ndarray (N,) or None
                各ニューロンへの入力スパイク発生確率。
                指定がなければ前回の値を維持、あるいは0とする。
        
        Returns:
            v_cpu: np.ndarray (N,)  現在の膜電位 (CPU)
            raster_cpu: np.ndarray (N,) 現在のスパイク (CPU)
        """
        i = self.step_count
        read_idx = np.int32(i % self.buffer_size)

        j = self.iter_count
        cuda.memcpy_htod(self.I_input_d.gpudata, input_data[j])

        # 1. ニューロン状態更新カーネル
        update_neuron_state(
            self.Vs_d.gpudata,
            self.Ns_d.gpudata,
            self.Qs_d.gpudata,
            self.Us_d.gpudata,
            self.neuron_type_d.gpudata,
            self.I_input_d.gpudata,
            self.synapses_out_d.gpudata,
            self.last_spike_d.gpudata,
            self.raster_d.gpudata,
            np.int32(i),
            block=(self.neuron_threads, 1, 1),
            grid=(self.neuron_blocks, 1),
            stream=self.stream1,
        )
        self.evt_update_neuron.record(self.stream1)

        # 3. スパイク伝播 (Delay Lineへの書き込み)
        propagate_spikes(
            self.delayed_spikes_d.gpudata,
            self.raster_d.gpudata,
            self.spike_in_d.gpudata, # 外部入力スパイク
            self.neuron_from_d.gpudata,
            self.delayed_row_d.gpudata,
            read_idx,
            block=(self.synapse_threads, 1, 1),
            grid=(self.synapse_blocks, 1),
            stream=self.stream1,
        )
        self.evt_spike_written.record(self.stream2)

        # 4. シナプス計算 (Delay Lineからの読み出しと電流計算)
        synapses_calc(
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
            np.float32(1e-2), # td
            np.float32(5e-3), # tr
            read_idx,
            block=(self.synapse_threads, 1, 1),
            grid=(self.synapse_blocks, 1),
            stream=self.stream2,
        )
        
        self.stream2.wait_for_event(self.evt_update_neuron)
        
        # 5. 行列ベクトル積 (シナプス電流の総和)
        mat_vec_mul(
            self.synapses_out_d.gpudata,
            self.neuron_to_d.gpudata,
            self.calc_matrix_d.gpudata,
            self.r_d.gpudata,
            block=(self.synapse_threads, 1, 1),
            grid=(self.synapse_blocks, 1),
            stream=self.stream2,
        )
        self.evt_update_input.record(self.stream2)


        # 6. 結果の取得 (Async copy)
        # 次のステップのために待機が必要な箇所を同期
        self.stream3.wait_for_event(self.evt_update_neuron)
        cuda.memcpy_dtoh_async(self.raster_log[j], self.raster_d.gpudata, stream=self.stream3)
        cuda.memcpy_dtoh_async(self.v_int[j], self.Vs_d.gpudata, stream=self.stream3)
        cuda.memcpy_dtoh_async(self.I_input_log[j], self.synapses_out_d.gpudata, stream=self.stream3)

        self.stream3.wait_for_event(self.evt_spike_written)
        spike_in_h = np.zeros(self.N, dtype=np.uint8)
        cuda.memcpy_htod_async(self.spike_in_d.gpudata, spike_in_h, stream=self.stream3)

        self.step_count += 1
        self.stream1.wait_for_event(self.evt_update_input)
        
        
        # ここではシンプルに stream3 で同期後に取得
        self.stream3.synchronize()
        # self.stream2.synchronize()
        # self.stream1.synchronize()
        self.raster_log[j] = self.raster_log[j] | spike_in_h
        self.I_input_log[j] += input_data[j]

        return None

    def plot_results(self, results, plot_num):

        v = results["v"]
        rasters = results["rasters"]
        inputs = results["input"]
        
        tmax = v.shape[0] * self.dt
        
        # 1. Single Neuron Plot
        plot_single_neuron(0, self.dt, tmax, v.shape[0], inputs, v, plot_num)
        
        # 2. Raster Plot
        plot_raster(self.dt, tmax, rasters, self.N, plot_num+1)


# -------------------------------------------------------------
# 3. 互換性維持のための main 関数
# -------------------------------------------------------------
def main():

    sim = PQN_Reservoir_GPU()

    # ステップ数の計算
    tmax = 2
    num_steps = int(tmax / sim.dt)

    # 実行
    start = time.perf_counter()
    
    input_data = np.zeros((num_steps,1), dtype=np.float32)
    input_data[:] = np.float32(0.0)
    input_data[5000:15000] = np.float32(0.0)

    # runメソッドで一括実行
    results = sim.run(
        num_steps, 
        input_data=input_data
    )
    
    end = time.perf_counter()
    print(f"processing time for {tmax:.2f}s simulation was {(end - start):.2f} s")


    # 記録モードならプロット
    plot_num = 0
    sim.plot_results(results, plot_num)


def plot_single_neuron(id, dt, tmax, number_of_iterations, I, v0, num):
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
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/single_neuron.png")
    plt.close()

def plot_raster(dt, tmax, rasters, N, num):
    times, neuron_ids = np.nonzero(rasters)
    times = times * dt
    plt.figure(num=num, figsize=(9, 5))
    plt.scatter(times, neuron_ids, s=1.1)
    plt.xlabel("time")
    plt.xlim(0, tmax)
    plt.ylabel("neuron ID")
    plt.ylim(0, N)
    plt.title("Raster Plot")
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/raster.png")
    plt.close()
    
if __name__ == "__main__":
    main()
