import os
from src import PQNModel
from src import Izhikevich
from src import LIF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import time
import networkx as nx
from src.Synapses import DoubleExponentialSynapse
from src.Synapses import tsodyks_markram
import random
from line_profiler import LineProfiler


def main():

    N = int(input("number of neurons: "))
    
    # ---- initialization ----
    SEED = int(random.random() * 1000)
    # SEED = 678
    random.seed(SEED) # for reproducibility
    np.random.seed(SEED)
    dt = 1e-4  # [s]
    tmax=3    # length of simulation [s]
    # cell0=LIF(N = N)
    # cell0=Izhikevich(N = N)
    cell0=PQNModel(N = N)
    number_of_iterations=int(tmax/dt)
    I=np.zeros((number_of_iterations, N), dtype=np.float32)  # input [nA]
    for i in range(100):
        temp = 0.9*random.random()/dt
        I[int(temp):int(0.1/dt+temp),random.randint(0,N-1)] = 0.13
    # background noise
    # for i in range(N):
    #     if i%(N//4) < N//5:
    #         I[:,i] += 2.5e-2*np.random.rand(number_of_iterations)
    #     else:
    #         I[:,i] += 1.0e-2*np.random.rand(number_of_iterations)
    # I[int(number_of_iterations/4):int(number_of_iterations/4*3),:] = 0.09
    v0= np.zeros((number_of_iterations, N))
    rasters = np.zeros((number_of_iterations, N), dtype=bool)
    output = np.zeros((number_of_iterations+int(0.5//dt), N), dtype=np.float32)  # synapse output [nA]
    next_input = np.zeros(N, dtype=np.float32)  # input for next time step [nA]
    resovoir_weight = np.zeros((N, N))
    resovoir_origin = np.zeros((N, N))
    num = 0


    # ---- 重み行列の作成 ----
    resovoir_origin, mask = create_reservoir_matrix(N)
    resovoir_weight = np.copy(resovoir_origin)

    N_S = np.count_nonzero(resovoir_weight)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    synapses_out1 = tsodyks_markram(N_S, dt=dt, tau_rec=0.5, U=0.5)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        if resovoir_weight[r, c] < 0:
            synapses_out1.mask_faci[i] = 1
            synapses_out1.U[i] = 0
        if r == c and c%(N//4) < N//5:
            synapses_out1.U[i] = 0.1
            synapses_out1.tau_rec[i] = 0.1
            
    delays = np.random.randint(8, 13, size=(N,N))  # delay for each neuron
    delays = delays * (mask != 0)

    # ---- 重み行列の可視化 ----
    visualize_matrix(resovoir_origin, num)
    # plt.show(block=False)
    num += 1

    # 正規化と自己結合強化
    # for i in range(N):
    #     if resovoir_weight[i][i] != 0 and i%(N//4) < N//5:
    #         # cell0.R[i] = 1000  #[mV/nA, MΩ]
    #         resovoir_weight[i][i] = 1.5*resovoir_weight[i][i]
    resovoir_weight = resovoir_weight*2000/N


    # ----  NetworkX グラフに変換 ----
    # show_network(resovoir_weight, N, SEED, num)
    # plt.show(block=False)
    # num += 1

    resovoir_weight_calc = np.zeros((N, N_S))
    synapses = np.zeros(N, dtype=int)
    synapses_spike = np.zeros(N_S, dtype=bool)
    delayed_synapses_out = np.zeros((13,N_S), dtype=bool)  # 最大遅延時間分のバッファ
    delay_row = np.zeros(N_S, dtype=int)
    col_indices, row_indices = np.where(resovoir_weight.T != 0)
    for i in range(N_S):
        r = row_indices[i]
        c = col_indices[i]
        resovoir_weight_calc[r, i] = resovoir_weight[r, c]
        delay_row[i] = delays[r, c]
    for i in range(N):
        synapses[i] = np.count_nonzero(resovoir_weight[:, i])
    cols = np.arange(N_S)


    buffer_size = 13
    # ---- RUN SIMULATION ----
    start = time.perf_counter()
    for i in tqdm(range(number_of_iterations)):
        read_idx = int(i%buffer_size)
        I[i] += next_input
        rasters[i], v0[i] = cell0.calc(inputs=I[i], itr=i)  # update cell state
        synapses_spike = np.repeat(rasters[i], repeats=synapses)
        delay_row_ = (delay_row + read_idx) % buffer_size
        delayed_synapses_out[delay_row_, cols] = synapses_spike
        hoge = synapses_out1(delayed_synapses_out[read_idx])  # update synapse state
        next_input = np.dot(resovoir_weight_calc, hoge) # [nA]
        output[i] = next_input
        # if delayed_synapses_out[read_idx,0] != 0:
        # # if rasters[i,0] != 0:
        # # if next_input[0] != 0:
        #     print(hoge[0])
        #     print(next_input[0])
        #     print(i)
    end = time.perf_counter()

    # print(np.where(rasters[:,201]==1))
    # print(delay_row)
    # print(resovoir_weight_calc[0])
    print(f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}")
    print(f"SEED value was {SEED}")


        # --- export simulation results to CSV ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = f"sim_results/SEED{SEED}_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    def fname(base):
        return os.path.join(outdir, f"{base}_seed{SEED}_{timestamp}.csv")

    # # times_vec = times
    # # np.savetxt(fname("times"), times_vec, delimiter=",", header="t_s", comments="", fmt="%.9f",)
    # np.savetxt(fname("I"), I, delimiter=",", header="input", comments="", fmt="%.9f",)
    # np.savetxt(fname("v"), v0, delimiter=",", header="v", comments="", fmt="%.9f",)
    # np.savetxt(fname("syn_output"), output, delimiter=",", header="syn_output", comments="", fmt="%.9f",)
    # np.savetxt(fname("reservoir_weight"), resovoir_weight, delimiter=",", header="reservoir_weight", comments="", fmt="%.9f",)
    # # spikes_sparse = np.vstack((times, neuron_ids)).T
    # # np.savetxt(fname("spikes_sparse"), spikes_sparse, delimiter=",", header="time,neuron_id", comments="", fmt=["%.9f", "%d"],)
    # with open(fname("meta"), "w") as f:
    #     f.write(f"SEED,{SEED}\n")
    #     f.write(f"dt,{dt}\n")
    #     f.write(f"tmax,{tmax}\n")
    #     f.write(f"N,{N}\n")

    # print(f"CSV files written under '{outdir}/' with filenames containing SEED {SEED} and timestamp {timestamp}.")

    # ---- plot simulation result ----
    plot_single_neuron(0, dt, tmax, number_of_iterations, I, v0, num)
    num += 1

    plot_raster(dt, tmax, rasters, N, num)
    num += 1

    # plt.show()


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

def show_network(resovoir_weight, N, SEED, num):
    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            w = resovoir_weight[i][j]
            if w != 0:
                G.add_edge(j, i, weight=w)  # j→i（pre→post）
    pos = nx.spring_layout(G, seed=SEED)    #   ノード配置（円形 or 自動レイアウト） 
    edges = G.edges(data=True)              #   エッジ属性（色と太さ） 
    edge_colors = ['red' if d['weight'] > 0 else 'blue' for (u, v, d) in edges]
    edge_widths = [abs(d['weight']) for (u, v, d) in edges]
    #   可視化 
    plt.figure(num=num, figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowstyle='-|>')
    plt.title("Reservoir Network Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("network.png")

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
    # profiler.add_function(tsodyks_markram.__call__)
    # profiler.add_function(main)
    # profiler.runcall(main)
    # profiler.print_stats()
    main()
