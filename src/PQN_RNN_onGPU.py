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
    def __init__(self, reservoir_state, cfg, buffer_size=10001, seed=None):
        effective_seed = seed if seed is not None else cfg.SEED
        self.rng = np.random.RandomState(effective_seed)
        """
        GPUメモリの確保と静的パラメータの初期化を行う
        """
        self.cfg = cfg
        self.buffer_size = buffer_size
        self.N = reservoir_state["N"]
        self.N_S = reservoir_state["N_S"]
        self.reservoir_state = reservoir_state

        # --- ホスト側(CPU) パラメータ準備 ---
        # TODO PBだけdtが異なるが、現在未対応
        self.neuron_type_h = reservoir_state["type"]
        self.input_indices = reservoir_state["input_indices"]
        self.output_indices = reservoir_state["output_indices"]

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

        # --- GPUメモリの確保 (Static: 重みや遅延など変化しないもの) ---
        self.neuron_type_d = gpuarray.to_gpu(np.uint8(self.neuron_type_h))
        self.tau_rec_d = gpuarray.to_gpu(reservoir_state["tau_rec_h"])
        self.tau_inact_d = gpuarray.to_gpu(reservoir_state["tau_inact_h"])
        self.tau_faci_d = gpuarray.to_gpu(reservoir_state["tau_faci_h"])
        self.U1_d = gpuarray.to_gpu(reservoir_state["U1_h"])
        self.U_d = gpuarray.to_gpu(reservoir_state["U_h"])
        self.mask_faci_d = gpuarray.to_gpu(reservoir_state["mask_faci_h"])
        self.neuron_from_d = gpuarray.to_gpu(reservoir_state["neuron_from_h"])
        self.calc_matrix_d = gpuarray.to_gpu(reservoir_state["calc_matrix_h"])
        self.neuron_to_d = gpuarray.to_gpu(reservoir_state["neuron_to_h"])
        self.delayed_row_d = gpuarray.to_gpu(reservoir_state["delayed_row_h"])

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
        all_params = np.zeros((7, 34), dtype=np.int32)
        
        params = [
            self.RSexci_param_h, 
            self.RSinhi_param_h, 
            self.FS_param_h, 
            self.LTS_param_h, 
            self.IB_param_h, 
            self.EB_param_h, 
            self.PB_param_h
            ]
        
        for i, param in enumerate(params):
            # --- 重要: Branchless化のためのパッチ ---
            # LTS/IB以外の場合、param[31], param[32] (eta) が0
            # 計算式 dn = (dn * eta) >> shift を成立させるため、
            # eta に 1.0 (つまり 1 << shift) を代入しておく。
            if param[31] == 0 and param[32] == 0:
                one_scaled = 1 << param[0] # param[0] is BIT_Y_SHIFT
                param[31] = one_scaled
                param[32] = one_scaled
            
            all_params[i] = param

        all_params_d, _ = module.get_global("AllParams")
        cuda.memcpy_htod(all_params_d, all_params)

        dt_float32 = np.float32(self.cfg.DT)
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

        tr_float32 = np.float32(self.reservoir_state["tr"])
        tr_d, _ = module.get_global("tr")
        cuda.memcpy_htod(tr_d, tr_float32)

        td_dloat32 = np.float32(self.reservoir_state["td"])
        td_d, _ = module.get_global("td")
        cuda.memcpy_htod(td_d, td_dloat32)

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

        # 入力スパイク生成用の確率ベクトル
        self.prob_all = np.zeros(self.N, dtype=np.float32)

    # 入力層の重み行列(今は使っていない)
    def set_input_projection(self):
        """入力がある場合の射影行列を初期化（ランダム）"""
        # 同じシードで同じ射影になるようにここで固定しても良いが、
        # 呼び出し元で制御することを想定
        input_dim = len(self.reservoir_state["input_indices"])
        self.W_in = self.rng.normal(0, 1, size=(input_dim, input_dim)).astype(np.float32)

    def step(self, prob_input_vector=None, record=None):
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
        if record is not None:                                #recordの記述について追記必要(修正予定)
            cuda.memcpy_dtoh_async(self.raster_log[j], self.raster_d.gpudata, stream=self.stream3)
            cuda.memcpy_dtoh_async(self.v_int[j], self.Vs_d.gpudata, stream=self.stream3)
        cuda.memcpy_dtoh_async(self.I_input_log[j], self.synapses_out_d.gpudata, stream=self.stream3)

        self.stream3.wait_for_event(self.evt_spike_written)
        # 入力スパイクの生成と転送 (CPU -> GPU)
        if prob_input_vector is not None:
            # 入力確率に基づいてスパイク生成
            spike_in_h = (self.rng.rand(self.N) < prob_input_vector).astype(np.uint8)
            cuda.memcpy_htod_async(self.spike_in_d.gpudata, spike_in_h, stream=self.stream3)
        else:
            # 入力なし（すべて0）
            # もし前回の値を保持したくない場合はここでゼロクリア
            spike_in_h = np.zeros(self.N, dtype=np.uint8)
            cuda.memcpy_htod_async(self.spike_in_d.gpudata, spike_in_h, stream=self.stream3)
            # pass

        self.step_count += 1
        self.stream1.wait_for_event(self.evt_update_input)
        
        
        # ここではシンプルに stream3 で同期後に取得
        self.stream3.synchronize()
        # self.stream2.synchronize()
        # self.stream1.synchronize()
        if record is not None:
            self.raster_log[j] = self.raster_log[j] | spike_in_h

        return None

    def run(self, num_steps, input_data=None, record=None):
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
        S_durt = self.cfg.INPUT_DT
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
        
        # 入力データがある場合のフレームステップ数
        steps_per_frame = int(round(S_durt / self.cfg.DT))
        
        # ループ実行
        iter_range = range(num_steps)
        # iter_range = tqdm(range(num_steps), desc="Simulating")

        # 内部確率ベクトルの初期化
        prob_spontaneous = self.cfg.SPONTANEOUS_FREQ * self.cfg.DT
        self.prob_all[:] = prob_spontaneous
        self.iter_count = 0

        for t in iter_range:
            # --- 入力データの更新処理 ---
            if input_data is not None:
                if t % steps_per_frame == 0:
                    self.prob_all[:] = prob_spontaneous
                    idx = t // steps_per_frame
                    if idx < input_data.shape[0]:
                        # 入力強度を入力確率に変換
                        prob_input = self.cfg.INPUT_FREQ * self.cfg.DT * input_data[idx]
                        self.prob_all[self.input_indices] += prob_input

            
            # --- 1ステップ実行 ---
            # input_dataがある場合は更新された prob_all を使用
            # input_dataがない場合(None)は prob_all (ゼロ) を使用
            self.step(prob_input_vector=self.prob_all, record=record)
            self.iter_count += 1

        # --- データの記録 (必要に応じて) ---
        # if record:
        #     # GPUから取得 (非同期コピーキューを入れる)
        #     # 注: 高速化のためには、Page-locked memory等を使うのがベターだが、
        #     # 互換性維持のため簡易実装にする
            
        #     # step() 内で同期しているので、ここでは直接get()できる
        #     # (step()の同期を外した場合はここで同期が必要)
            
        #     # 膜電位 (固定小数点 -> float)
        #     self.v_log = self.v_int / (2**self.RSexci.BIT_WIDTH_FRACTIONAL)
        #     self.raster_log = self.raster_d.get() | self.spike_in_d.get() # 内部発火 + 外部入力
        #     self.I_input_log = self.synapses_out_d.get()

        self.v_log = (self.v_int - self.v_int[0,:]) / (2**self.RSexci.BIT_WIDTH_FRACTIONAL)


        # 結果を辞書などで返す
        results = {}
        # if record:
        results["rasters"] = self.raster_log
        results["input"] = np.abs(self.I_input_log)
        results["v"] = self.v_log
        
        return results

    def plot_results(self, results, plot_num, record):

        v = results["v"]
        rasters = results["rasters"]
        inputs = results["input"]
        
        tmax = v.shape[0] * self.cfg.DT
        
        # 1. Single Neuron Plot
        plot_single_neuron(0, self.cfg.DT, tmax, v.shape[0], inputs, v, plot_num, record)
        
        # 2. Raster Plot
        plot_raster(self.cfg.DT, tmax, rasters, self.N, plot_num+1, record)


# -------------------------------------------------------------
# 3. 互換性維持のための main 関数
# -------------------------------------------------------------
def main(
    input_data: np.ndarray | None = None,
    coch: bool = False,
    reservoir_state=None,
    return_feature: bool = True,
    is_debug_print: bool = False,
    record=None,
    tmax=None,
    S_durt = 1e-2,
    cfg=None,
    sim=None,
):
    """
    従来の関数インターフェースを維持したラッパー
    """
    if cfg is None:
        raise ValueError("Config (cfg) must be provided.")

    # 【修正】simが渡されていなければ作り、渡されていればリセットして使う
    if sim is None:
        if is_debug_print: print("Initializing new reservoir instance...")
        sim = PQN_Reservoir_GPU(reservoir_state, cfg)
    else:
        # 重み行列などはそのままに、膜電位などを初期化
        sim.reset_state()

    if coch:
        sim.input_indices = reservoir_state["input_indices_coch"]
        sim.output_indices = reservoir_state["output_indices_coch"]

    # ステップ数の計算
    dt = cfg.DT
    if tmax is None:
        tmax = input_data.shape[0] * S_durt
        num_steps = int(tmax / dt)
    else:
        num_steps = int(tmax / dt)

    # 実行
    start = time.perf_counter()
    
    # runメソッドで一括実行
    results = sim.run(
        num_steps, 
        input_data=input_data, 
        record=record
    )
    
    end = time.perf_counter()
    if is_debug_print:
        print(f"processing time for {tmax:.2f}s simulation was {(end - start):.2f} s")


    # 記録モードならプロット
    plot_num = 0
    if record is not None:        
        # 可視化関数の呼び出し
        # 可視化関数はクラス外に定義されているものを再利用
        # 重み行列
        # visualize_matrix(reservoir_state["reservoir_weight"], plot_num, cfg)
        # plot_num += 1

        # ニューロン応答とラスター
        sim.plot_results(results, plot_num, record)

    # return None
    if return_feature:
        plt.close("all")
        read_indices = reservoir_state["output_indices"]
        x_t = results["input"][:, read_indices].astype(np.float32)  # shape = (T, Nout)
        return x_t


# -------------------------------------------------------------
# 4. 可視化・ヘルパー関数 (従来通り)
# -------------------------------------------------------------
# def visualize_matrix(matrix, num, cfg):
#     plt.figure(num=num, figsize=(8, 6))
#     max_abs = np.max(np.abs(matrix))
#     im = plt.imshow(matrix, aspect="auto", cmap="plasma", vmin=-max_abs, vmax=max_abs)
#     plt.gca().invert_yaxis()
#     plt.colorbar(im, label="Weight Value")
#     plt.title("Reservoir Weight Matrix")
#     plt.xlabel("Pre Neuron")
#     plt.ylabel("Post Neuron")
#     plt.tight_layout()
#     os.makedirs(f"{RESULT_DIR}/figs", exist_ok=True)
#     os.makedirs(f"{RESULT_DIR}/data", exist_ok=True)
#     plt.savefig(f"{RESULT_DIR}/figs/reservoir_weight_matrix.png")
#     np.save(f"{RESULT_DIR}/data/reservoir.npy", matrix)
#     plt.close()

def plot_single_neuron(id, dt, tmax, number_of_iterations, I, v0, num, record):
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
    os.makedirs(f"{record["result_dir"]}/figs", exist_ok=True)
    plt.savefig(f"{record["result_dir"]}/figs/{record["filename"]}_MembranePotential.png")
    plt.close()
    time_axis = np.arange(number_of_iterations) * dt
    data_to_save = np.column_stack((time_axis, I[:, id], v0[:, id]))
    np.save(f"{record["result_dir"]}/data/{record["filename"]}_MembranePotential.npy", data_to_save)

def plot_raster(dt, tmax, rasters, N, num, record):
    times, neuron_ids = np.nonzero(rasters)
    times = times * dt
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
    os.makedirs(f"{record["result_dir"]}/figs", exist_ok=True)
    os.makedirs(f"{record["result_dir"]}/data", exist_ok=True)
    plt.savefig(f"{record["result_dir"]}/figs/{record["filename"]}_raster.png")
    np.save(f"{record["result_dir"]}/data/{record["filename"]}_raster.npy", rasters)
    plt.close()
