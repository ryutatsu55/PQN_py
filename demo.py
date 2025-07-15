from src import PQNModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import time
import networkx as nx
from src.Synapses import DoubleExponentialSynapse

if __name__ == "__main__":

    start = time.time()
    #りざばーのネットワーク作る
    #network可視化
    #学習アルゴリズム追加
    #8:2で興奮と抑制混ぜる

    # set a PQN cell
    # you can use RSexci, RSinhi, FS, LTS, IB, EB, PB, or Class2 mode
    N = 1000
    cell0=PQNModel(mode='RSexci', N = N)
    # set a synapse
    synapses_out = DoubleExponentialSynapse(N, dt=cell0.PARAM['dt'], td=2e-2, tr=2e-3)

    #initialization
    # length of simulation [s]
    tmax=2
    spike = np.zeros(N)
    past_spike = np.zeros(N)
    # set the number of iterations
    number_of_iterations=int(tmax/cell0.PARAM['dt'])

    # set step input
    I=np.zeros(number_of_iterations)
    I[int(number_of_iterations/4):int(number_of_iterations/4*3)] = 0.09

    # run simulatiion
    v0= np.zeros((number_of_iterations, N))
    output = np.zeros((number_of_iterations, N))
    for i in range(number_of_iterations):
        cell0.update(I[i])
        v0[i] = (cell0.get_membrane_potential())
        spike = np.where(v0[i] > 4, 1, 0)
        output[i] = synapses_out(np.where(spike-past_spike > 0, 1, 0))
        # tmp = np.where(spike-past_spike > 0, 1, 0)
        # if(tmp[1] == 1):
        #     print(f"Spike at iteration {i}, time {i * cell0.PARAM['dt']:.2f} s")
        past_spike = spike
    #print(output.shape)



    end = time.time()
    print(f"Simulation time: {end - start:.2f} seconds")

    # plot simulation result
    fig = plt.figure(figsize=(10,4))
    spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, hspace=0.1, height_ratios=[4, 4, 1])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax0.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], v0[:,2])
    ax0.set_xlim(0, tmax)
    ax0.set_ylabel("v")
    ax0.set_xticks([])
    ax1.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], output[:,2])
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("output")
    ax1.set_xticks([])
    ax2.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], I, color="black")
    ax2.set_xlim(0, tmax)
    ax2.set_xlabel("[s]")
    ax2.set_ylabel("I")
    plt.savefig("demo.png")
    plt.show()
