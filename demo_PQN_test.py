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

    temp = int(input("how much times: "))
    N = 1
    
    # ---- initialization ----
    # SEED = int(random.random() * 1000)
    SEED = 678
    random.seed(SEED) # for reproducibility
    np.random.seed(SEED)
    dt = 1e-4  # [s]
    tmax=3    # length of simulation [s]
    cell0=PQNModel(N = N)
    number_of_iterations=int(tmax/dt)
    I=np.zeros((number_of_iterations, N), dtype=np.float32)  # input [nA]
    v0= np.zeros((number_of_iterations, N))
    rasters = np.zeros((number_of_iterations+10000, N), dtype=bool)
    freq = 20   #[Hz]
    span = int((1/freq)//dt)
    rasters[::span, :] = 1
    rasters[0] = 0
    output = np.zeros((number_of_iterations+int(0.5//dt), N), dtype=np.float32)  # synapse output [nA]
    next_input = np.zeros(N, dtype=np.float32)  # input for next time step [nA]
    resovoir_weight = np.ones((N, N))
    num = 0


    N_S = np.count_nonzero(resovoir_weight)
    synapses_out1 = tsodyks_markram(N_S, dt=dt, tau_rec=0.5, U=0.5)
    synapses_out1.mask_faci[0] = 1
    synapses_out1.U[0] = 0
    synapses_out1.tau_rec[0] = 0.1
    synapses_out1.tau_inact[0] = 0.0015

    resovoir_weight_calc = resovoir_weight*temp/1000

    # ---- RUN SIMULATION ----
    start = time.perf_counter()
    for i in tqdm(range(number_of_iterations)):
        I[i] += next_input
        raster, v0[i] = cell0.calc(inputs=I[i], itr=i)  # update cell state
        rasters[i+10] = raster or rasters[i+10]  # store spike
        hoge = synapses_out1(rasters[i])  # update synapse state
        next_input = np.dot(resovoir_weight_calc, hoge) # [nA]
        output[i] = next_input
    end = time.perf_counter()

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
    G = 0.1
    p = 0.05
    while crust_idx != 4:
        i1 = int(crust_idx * N / 4)
        i2 = int((crust_idx + 1) * N / 4)
        resovoir_weight[i1:i2, i1:i2] = ((G * np.random.randn(N//4, N//4)) + 1) * (np.random.rand(N//4, N//4) < p)
        # print(resovoir_weight[i1:i2, i1:i2])
        crust_idx += 1

    #クラスター間の接続
    M = 4
    G = 0.1
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
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = ((G * np.random.randn(N//4, N//4)) + 1) * (np.random.rand(N//4, N//4) < p)

        i_range1 = int(((hoge+1)*N/4)%N)
        i_range2 = int((hoge+2)*N/4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int((hoge*N/4)%N)
        j_range2 = int((hoge+1)*N/4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = ((G * np.random.randn(N//4, N//4)) + 1) * (np.random.rand(N//4, N//4) < p)

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
    plt.figure(num=num, figsize=(9, 5))
    plt.scatter(times, neuron_ids, s=1)
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
