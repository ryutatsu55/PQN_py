from src import PQNModel
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
    # seed = 3  # random seed for reproducibility
    ##やることリスト
    #学習アルゴリズム追加

    # set a PQN cell
    # you can use RSexci, RSinhi, FS, LTS, IB, EB, PB, or Class2 mode
    N = int(input("number of neurons: "))
    cell0=PQNModel(mode='RSexci', N = N)
    # set a synapse
    synapses_out1 = tsodyks_markram(N, dt=cell0.PARAM['dt'])
    synapses_out2 = DoubleExponentialSynapse(N, dt=cell0.PARAM['dt'], td=1e-2, tr=5e-3)

    #initialization
    # length of simulation [s]
    tmax=10
    # set the number of iterations
    number_of_iterations=int(tmax/cell0.PARAM['dt'])

    v0= np.zeros((number_of_iterations, N))
    spike = np.zeros(N)
    past_spike = np.zeros(N)
    resovoir_weight = np.zeros((N, N))
    random.seed(seed) # for reproducibility

    crust_idx = 0
    while crust_idx != 4:
        for i in range(int(crust_idx*N/4), int((crust_idx+1)*N/4)):
            for j in range(int(crust_idx*N/4), int((crust_idx+1)*N/4)):
                p = random.random()
                if p < 0.05:  # pの確率で結合
                    if j<N/5+crust_idx*N/4:
                        resovoir_weight[i][j] = 0.1*random.random() + 1
                        # resovoir_weight[i][j] = 1
                    else:
                        resovoir_weight[i][j] = -0.1*random.random() - 1  # 2割　抑制結合
                        # resovoir_weight[i][j] = -1
        crust_idx += 1

    #クラスター間の接続
    M = 4   # クラスタ間の接続数
    for hoge in range(M*4):
        i_range = (hoge*N/4)%N
        j_range = ((hoge+1)*N/4)%N
        p = random.random()
        if p < 0.5:
            i, j = j, i  # 入れ替え
        i = random.randint(int(i_range), int(i_range+N/4-1))
        j = random.randint(int(j_range), int(j_range+N/4-1))
        if (j%(N/4) < N/5):
            resovoir_weight[i][j] = 0.1*random.random()+1
        else:
            resovoir_weight[i][j] = -0.1*random.random()-1

    for i in range(N):
        count = np.count_nonzero(resovoir_weight[i])
        if count > 0:
            resovoir_weight[i] = resovoir_weight[i] / count
    
    # # ---  NetworkX グラフに変換 ---
    # G = nx.DiGraph()
    # for i in range(N):
    #     for j in range(N):
    #         w = resovoir_weight[i][j]
    #         if w != 0:
    #             G.add_edge(j, i, weight=w)  # j→i（pre→post）
    # #   ノード配置（円形 or 自動レイアウト） 
    # pos = nx.spring_layout(G, seed=seed)  # spring_layout / circular_layout など
    # #   エッジ属性（色と太さ） 
    # edges = G.edges(data=True)
    # edge_colors = ['red' if d['weight'] > 0 else 'blue' for (u, v, d) in edges]
    # edge_widths = [abs(d['weight']) for (u, v, d) in edges]
    # #   可視化 
    # plt.figure(num=1, figsize=(8, 8))
    # nx.draw_networkx_nodes(G, pos, node_size=5, node_color="lightgray")
    # nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowstyle='-|>')
    # plt.title("Reservoir Network Graph")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig("network.png")
    # # plt.show(block=False)

    # set step input
    dt = cell0.PARAM['dt']
    I=np.zeros((number_of_iterations, N))
    for i in range(10):
        temp = 0.9*random.random()/dt
        I[int(0.05/dt+temp):int(0.1/dt+temp),random.randint(0,N)] = 0.09
    # I[int(0.05/dt):int(0.1/dt),0] = 0.09
    rasters = np.zeros((number_of_iterations, N))
    output = np.zeros((number_of_iterations+int(0.5//dt), N))
    cols = np.arange(output.shape[1])
    next_input = np.zeros(N)
    delays = np.random.randint(0.03//dt, 0.5//dt, size=N)  # delay for each neuron [ms]
    # delay = np.random.randint(0.03//dt, 0.5//dt)  # delay for all neuron
    # delays = np.full(N, delay-1)  # uniform delay for all neurons
    # run simulatiion
    start = time.perf_counter()
    for i in tqdm(range(number_of_iterations)):
        I[i] += next_input
        cell0.update(I[i])  # update cell state
        v0[i] = (cell0.get_membrane_potential())
        spike = np.where(v0[i] > 4, 1, 0)
        rasters[i] = np.where(spike-past_spike > 0, 1, 0)
        rows = delays + i+1
        output[rows, cols] = 0.0015*synapses_out2(2*synapses_out1(rasters[i]))   #[pA]
        # output[rows, cols] = 15*synapses_out1(rasters[i])  # [nA]
        next_input = np.dot(resovoir_weight, output[i])  # update input for next iteration
        past_spike = spike
    #print(output.shape)



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
    ax0.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], v0[:,0])    
    ax0.set_xlim(0, tmax)
    ax0.set_ylabel("v")
    ax0.set_xticks([])
    ax1.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], output[:number_of_iterations,0])
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("synapse output")
    ax2.set_xticks([])
    ax2.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], I[:,0], color="black")
    ax2.set_xlim(0, tmax)
    ax1.set_xlabel("[s]")
    ax2.set_ylabel("I")
    fig.savefig("single_neuron.png")

    times, neuron_ids = np.nonzero(rasters)
    times = times * cell0.PARAM['dt']
    neuron_ids = neuron_ids + 1  # Adjust neuron IDs to start from 1
    plt.figure(figsize=(9, 2))
    plt.scatter(times, neuron_ids, s=1, color='black')
    plt.xlabel("time")
    plt.xlim(0, tmax)
    plt.ylabel("neuron ID")
    plt.ylim(0, N)
    plt.title("Raster Plot")
    plt.tight_layout()
    plt.show()

    plt.savefig("raster.png")
