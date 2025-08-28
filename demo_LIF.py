from src import PQNModel
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

if __name__ == "__main__":
    seed = int(random.random() * 1000)
    # seed = 881  # random seed for reproducibility
    ##やることリスト
    #学習アルゴリズム追加

    # set a PQN cell
    N = int(input("number of neurons: "))
    cell0=LIF(N = N)
    dt = 0.0001

    # set a synapse
    synapses_out1 = tsodyks_markram(N, dt=dt, tau_rec=0.5, U=0.5)

    #initialization
    # length of simulation [s]
    tmax=10
    # set the number of iterations
    number_of_iterations=int(tmax/dt)

    v0= np.zeros((number_of_iterations, N))
    spike = np.zeros(N)
    past_spike = np.zeros(N)
    resovoir_weight = np.zeros((N, N))
    random.seed(seed) # for reproducibility

    crust_idx = 0
    G = 0.05
    p = 0.05
    while crust_idx != 4:
        i1 = int(crust_idx * N / 4)
        i2 = int((crust_idx + 1) * N / 4)
        resovoir_weight[i1:i2, i1:i2] = ((G * np.random.randn(N//4, N//4)) / (np.sqrt(N) * p) + 1) * (np.random.rand(N//4, N//4) < p)
        # print(resovoir_weight[i1:i2, i1:i2])
        crust_idx += 1

    # weights = G * np.random.randn(100000) / (np.sqrt(N) * p) + 1
    # plt.figure(figsize=(8, 4))
    # plt.hist(weights, bins=30, color='skyblue', edgecolor='black', density=True)
    # plt.title("Distribution of G * randn(N) / (sqrt(N) * p) + 1")
    # plt.xlabel("Weight Value")
    # plt.ylabel("Probability Density")
    # plt.tight_layout()
    # plt.savefig("weight_distribution.png")
    # plt.show()

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
         
    # weights = G * np.random.randn(100000) / (np.sqrt(N) * p) + 1
    # plt.figure(figsize=(8, 4))
    # plt.hist(weights, bins=30, color='skyblue', edgecolor='black', density=True)
    # plt.title("Distribution of G * randn(N) / (sqrt(N) * p) + 1")
    # plt.xlabel("Weight Value")
    # plt.ylabel("Probability Density")
    # plt.tight_layout()
    # plt.savefig("weight_distribution.png")
    # plt.show()

    for hoge in range(M):
        i_range1 = int(((hoge+1)*N/4)%N)
        i_range2 = int((hoge+2)*N/4)
        if i_range2 > N:
            i_range2 = i_range2 % N
        j_range1 = int((hoge*N/4)%N)
        j_range2 = int((hoge+1)*N/4)
        if j_range2 > N:
            j_range2 = j_range2 % N
        resovoir_weight[i_range1:i_range2, j_range1:j_range2] = ((G * np.random.randn(N//4, N//4)) / (np.sqrt(N) * p) + 1) * (np.random.rand(N//4, N//4) < p)

    base_mask = np.ones((N, int(N/4)))
    base_mask[:, N//5:] = -1
    mask = np.hstack([base_mask for _ in range(4)])
    resovoir_weight = resovoir_weight * mask
    
    # 正規化と自己結合強化
    for i in range(N):
        count = np.count_nonzero(resovoir_weight[i])
        if count > 0:
            resovoir_weight[i] = resovoir_weight[i] / count
        if resovoir_weight[i][i] != 0 and i%(N//4) < N//5:
            resovoir_weight[i][i] = 4 * count * resovoir_weight[i][i]  # 自己結合を強化
            synapses_out1.U[i] = 0.1
            synapses_out1.tau_rec[i] = 0.1

    # print(resovoir_weight)

    # ---  NetworkX グラフに変換 ---
    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            w = resovoir_weight[i][j]
            if w != 0:
                G.add_edge(j, i, weight=w)  # j→i（pre→post）
    #   ノード配置（円形 or 自動レイアウト） 
    pos = nx.spring_layout(G, seed=seed)  # spring_layout / circular_layout など
    #   エッジ属性（色と太さ） 
    edges = G.edges(data=True)
    edge_colors = ['red' if d['weight'] > 0 else 'blue' for (u, v, d) in edges]
    edge_widths = [abs(d['weight']) for (u, v, d) in edges]
    #   可視化 
    plt.figure(num=1, figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowstyle='-|>')
    plt.title("Reservoir Network Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("network.png")
    plt.show(block=False)

    # set step input
    I=np.zeros((number_of_iterations, N))
    for i in range(100):
        temp = 0.9*random.random()/dt
        I[int(temp):int(0.1/dt+temp),random.randint(0,N-1)] = 0.13
    # I[int(0.05/dt):int(0.15/dt),0] = 0.13
    rasters = np.zeros((number_of_iterations, N))
    output = np.zeros((number_of_iterations+int(0.5//dt), N))
    cols = np.arange(output.shape[1])
    next_input = np.zeros(N)
    delays = np.random.randint(0.0008//dt, 0.0012//dt, size=N)  # delay for each neuron
    # delays = np.random.randint(0.001//dt, 0.02//dt, size=N)  # delay for each neuron
    # delay = np.random.randint(0.03//dt, 0.5//dt)  # delay for all neuron
    # delays = np.full(N, delay-1)  # uniform delay for all neurons

    # run simulatiion
    start = time.perf_counter()
    for i in tqdm(range(number_of_iterations)):
        I[i] += next_input
        rasters[i], v0[i] = cell0.calc(inputs=I[i], itr=i)  # update cell state
        rows = delays + i
        output[rows, cols] = 35*synapses_out1(rasters[i])  # [nA]
        next_input = np.dot(resovoir_weight, output[i])  # update input for next iteration
    end = time.perf_counter()

    print(f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}")
    print(f"seed value was {seed}")

    # plot simulation result
    fig = plt.figure(num=2, figsize=(10,4))
    spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, hspace=0.1, height_ratios=[1, 4, 4])
    ax2 = fig.add_subplot(spec[0])
    ax0 = fig.add_subplot(spec[1])
    ax1 = fig.add_subplot(spec[2])
    times, neuron_ids = np.nonzero(rasters)
    ax0.plot([i*dt for i in range(0, number_of_iterations)], v0[:,0])    
    ax0.set_xlim(0, tmax)
    ax0.set_ylabel("v")
    ax0.set_xticks([])
    ax1.plot([i*dt for i in range(0, number_of_iterations)], output[:number_of_iterations,0])
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("synapse output")
    ax2.set_xticks([])
    ax2.plot([i*dt for i in range(0, number_of_iterations)], I[:,0], color="black")
    ax2.set_xlim(0, tmax)
    ax1.set_xlabel("[s]")
    ax2.set_ylabel("I")
    fig.savefig("single_neuron.png")

    times, neuron_ids = np.nonzero(rasters)
    times = times * dt
    neuron_ids = neuron_ids + 1  # Adjust neuron IDs to start from 1
    cluster_colors = ['red', 'blue', 'green', 'orange']
    cluster_id = ((neuron_ids - 1) // (N // 4))  # 0,1,2,3 のクラスタID
    colors = [cluster_colors[c % 4] for c in cluster_id]
    plt.figure(figsize=(9, 5))
    plt.scatter(times, neuron_ids, s=1, color=colors)
    plt.xlabel("time")
    plt.xlim(0, tmax)
    plt.ylabel("neuron ID")
    plt.ylim(0, N)
    plt.title("Raster Plot")
    plt.tight_layout()
    plt.savefig("raster.png")

    plt.figure(figsize=(8, 6))
    max_abs = np.max(np.abs(resovoir_weight))
    im = plt.imshow(resovoir_weight, aspect='auto', cmap='bwr', vmin=-max_abs, vmax=max_abs)  # 0が白になる
    plt.colorbar(im, label='Weight Value')
    plt.title('Reservoir Weight Matrix')
    plt.xlabel('Pre Neuron')
    plt.ylabel('Post Neuron')
    plt.tight_layout()
    plt.savefig("resovoir_weight_matrix.png")
    plt.show()
