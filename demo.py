from src import PQNModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import time
import networkx as nx
from src.Synapses import DoubleExponentialSynapse
import random

if __name__ == "__main__":
    seed = 1
    ##やることリスト
    #学習アルゴリズム追加

    # set a PQN cell
    # you can use RSexci, RSinhi, FS, LTS, IB, EB, PB, or Class2 mode
    N = int(input("number of neurons: "))
    cell0=PQNModel(mode='RSexci', N = N)
    # set a synapse
    synapses_out = DoubleExponentialSynapse(N, dt=cell0.PARAM['dt'], td=2e-2, tr=2e-3)

    #initialization
    # length of simulation [s]
    tmax=1
    # set the number of iterations
    number_of_iterations=int(tmax/cell0.PARAM['dt'])

    v0= np.zeros((number_of_iterations, N))
    output = np.zeros((number_of_iterations, N))
    spike = np.zeros(N)
    past_spike = np.zeros(N)
    resovoir_weight = np.zeros((N, N))
    random.seed(seed) # for reproducibility

    crust_idx = 0
    while crust_idx != 4:
        for i in range(int(crust_idx*N/4), int((crust_idx+1)*N/4)):
            for j in range(int(crust_idx*N/4), int((crust_idx+1)*N/4)):
                p = random.random()
                if p < 0.5:
                    i, j = j, i  # 入れ替え
                p = random.random()
                if p < 0.05:  # pの確率で結合
                    if j<N/5+crust_idx*N/4:
                        resovoir_weight[i][j] = 1*random.random()
                    else:
                        resovoir_weight[i][j] = -1*random.random()  # 2割　抑制結合
        crust_idx += 1

    M = 1   # クラスタ間の接続数
    for hoge in range(M*4):
        i_range = (hoge*N/4)%N
        j_range = (hoge*N/2)%N
        i = random.randint(int(i_range), int(i_range+N/4-1))
        j = random.randint(int((j_range+N/4)%N), int((j_range+N/2-1)%N))
        if (j%(N/4) < N/5):
            resovoir_weight[i][j] = 1*random.random()
        else:
            resovoir_weight[i][j] = -1*random.random()

    #クラスター間の接続
                

    # np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
    # print(resovoir_weight)10
    
    # --- 2. NetworkX グラフに変換 ---
    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            w = resovoir_weight[i][j]
            if w != 0:
                G.add_edge(j, i, weight=w)  # j→i（pre→post）

    # --- 3. ノード配置（円形 or 自動レイアウト） ---
    pos = nx.spring_layout(G, seed=seed)  # spring_layout / circular_layout など

    # --- 4. エッジ属性（色と太さ） ---
    edges = G.edges(data=True)
    edge_colors = ['red' if d['weight'] > 0 else 'blue' for (u, v, d) in edges]
    edge_widths = [abs(d['weight']) for (u, v, d) in edges]

    # --- 5. 可視化 ---
    plt.figure(num=1, figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowstyle='-|>')
    plt.title("Reservoir Network Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("network.png")
    plt.show(block=False)

    # set step input
    dt = cell0.PARAM['dt']
    I=np.zeros((number_of_iterations, N))
    I[int(0.05/dt):int(0.1/dt),2] = 0.09
    next_input = np.zeros(N)

    # run simulatiion
    start = time.perf_counter()
    for i in tqdm(range(number_of_iterations)):
        I[i] += next_input
        cell0.update(I[i])  # update cell state
        v0[i] = (cell0.get_membrane_potential())
        spike = np.where(v0[i] > 4, 1, 0)
        output[i] = synapses_out(np.where(spike-past_spike > 0, 1, 0))
        # tmp = np.where(spike-past_spike > 0, 1, 0)
        # if(tmp[1] == 1):
        #     print(f"Spike at iteration {i}, time {i * cell0.PARAM['dt']:.2f} s")
        next_input = 0.05*np.dot(resovoir_weight, output[i])  # update input for next iteration
        past_spike = spike
    #print(output.shape)



    end = time.perf_counter()
    print(f"processing time for {tmax*1000}ms simulation mas {(end - start)*1000} ms when reservoir_size was {N}")

    # plot simulation result
    fig = plt.figure(num=2, figsize=(10,4))
    spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, hspace=0.1, height_ratios=[4, 4, 1])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax0.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], v0[:,4])
    ax0.set_xlim(0, tmax)
    ax0.set_ylabel("v")
    ax0.set_xticks([])
    ax1.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], output[:,4])
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("output")
    ax1.set_xticks([])
    ax2.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], I[:,4], color="black")
    ax2.set_xlim(0, tmax)
    ax2.set_xlabel("[s]")
    ax2.set_ylabel("I")
    plt.savefig("demo.png")
    plt.show()
