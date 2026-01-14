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
with open("RNN_analyze/src/my_kernel.cu", "r", encoding="utf-8") as f:
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
    def __init__(self, reservoir_state, cfg, buffer_size=1001, seed=None):
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
        self.neuron_type_h = reservoir_state["type"]
        self.input_indices = reservoir_state["input_indices"]
        self.output_indices = reservoir_state["output_indices"]

        # PQNパラメータ初期化 (Excitatory / Inhibitory)
        self.RSexci = PQNparam(mode="RSexci")
        self.RSinhi = PQNparam(mode="RSinhi")
        self.RSexci_param_h = self._param_h_init(self.RSexci)
        self.RSinhi_param_h = self._param_h_init(self.RSinhi)

        # 定数パラメータのGPU転送 (Globalメモリ)
        self._upload_constants()

        # --- GPUメモリの確保 (Static: 重みや遅延など変化しないもの) ---
        self.neuron_type_d = gpuarray.to_gpu(self.neuron_type_h)
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
        # (元の param_h_init 関数の中身を移植)
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

    def _upload_constants(self):
        # Global変数への転送
        RSexci_param_d, _ = module.get_global("RSexci_param")
        cuda.memcpy_htod(RSexci_param_d, self.RSexci_param_h)
        RSinhi_param_d, _ = module.get_global("RSinhi_param")
        cuda.memcpy_htod(RSinhi_param_d, self.RSinhi_param_h)

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

    def reset_state(self):
        """
        動的な状態変数（膜電位、スパイク履歴など）を初期化する。
        重みなどの静的なものは維持する。
        """
        self.step_count = 0

        # CPU側で初期値作成
        Vs_h = np.full(self.N, -4906, dtype=np.int64)
        Ns_h = np.full(self.N, 27584, dtype=np.int64)
        Qs_h = np.full(self.N, -3692, dtype=np.int64)
        
        # ニューロンタイプに応じた初期値設定
        for i in range(self.N):
            if self.neuron_type_h[i] == 0: # Exci
                Vs_h[i] = self.RSexci.state_variable_v
                Ns_h[i] = self.RSexci.state_variable_n
                Qs_h[i] = self.RSexci.state_variable_q
            elif self.neuron_type_h[i] == 1: # Inhi
                Vs_h[i] = self.RSinhi.state_variable_v
                Ns_h[i] = self.RSinhi.state_variable_n
                Qs_h[i] = self.RSinhi.state_variable_q

        # GPU転送 (Dynamic Variables)
        self.Vs_d = gpuarray.to_gpu(Vs_h)
        self.Ns_d = gpuarray.to_gpu(Ns_h)
        self.Qs_d = gpuarray.to_gpu(Qs_h)
        
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

    def step(self, prob_input_vector=None, record=False):
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
        if record:                                    #recordの記述について追記必要(修正予定)
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
        if record:
            self.raster_log[j] = self.raster_log[j] | spike_in_h

        return None

    def run(self, num_steps, input_data=None, record=False):
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
                    idx = t // steps_per_frame
                    if idx < input_data.shape[0]:
                        # 入力強度を入力確率に変換
                        self.prob_all[:] = prob_spontaneous
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

    def plot_results(self, results, plot_num, cfg):

        v = results["v"]
        rasters = results["rasters"]
        inputs = results["input"]
        
        tmax = v.shape[0] * self.cfg.DT
        
        # 1. Single Neuron Plot
        plot_single_neuron(0, self.cfg.DT, tmax, v.shape[0], inputs, v, plot_num, cfg)
        
        # 2. Raster Plot
        plot_raster(self.cfg.DT, tmax, rasters, self.N, plot_num+1, cfg)


# -------------------------------------------------------------
# 3. 互換性維持のための main 関数
# -------------------------------------------------------------
def main(
    input_data: np.ndarray | None = None,
    reservoir_state=None,
    return_feature: bool = True,
    is_debug_print: bool = False,
    density: float = 0.1,
    N: int = 500,
    record: bool = False,
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

    # ステップ数の計算
    tmax = 10
    dt = cfg.DT
    if input_data is not None:
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
    if record:        
        # 可視化関数の呼び出し
        # 可視化関数はクラス外に定義されているものを再利用
        # 重み行列
        visualize_matrix(reservoir_state["reservoir_weight"], plot_num, cfg)
        plot_num += 1

        # ニューロン応答とラスター
        sim.plot_results(results, plot_num, cfg)

    # return None
    if return_feature:
        plt.close("all")
        read_indices = reservoir_state["output_indices"]
        x_t = results["input"][:, read_indices].astype(np.float32)  # shape = (T, Nout)
        return x_t


# -------------------------------------------------------------
# 4. 可視化・ヘルパー関数 (従来通り)
# -------------------------------------------------------------
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
    os.makedirs(f"{cfg.RESULT_DIR}/figs", exist_ok=True)
    os.makedirs(f"{cfg.RESULT_DIR}/data", exist_ok=True)
    plt.savefig(f"{cfg.RESULT_DIR}/figs/reservoir_weight_matrix.png")
    np.save(f"{cfg.RESULT_DIR}/data/reservoir.npy", matrix)
    plt.close()

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
    os.makedirs(f"{cfg.RESULT_DIR}/figs", exist_ok=True)
    plt.savefig(f"{cfg.RESULT_DIR}/figs/single_neuron.png")
    plt.close()

def plot_raster(dt, tmax, rasters, N, num, cfg):
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
    os.makedirs(f"{cfg.RESULT_DIR}/figs", exist_ok=True)
    os.makedirs(f"{cfg.RESULT_DIR}/data", exist_ok=True)
    plt.savefig(f"{cfg.RESULT_DIR}/figs/raster.png")
    np.save(f"{cfg.RESULT_DIR}/data/raster.npy", rasters)
    plt.close()
