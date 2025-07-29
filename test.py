from src import PQNModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import time
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
    # N = int(input("number of neurons: "))
    N=1
    cell0=PQNModel(mode='RSexci', N = N)
    # set a synapse
    synapses_out1 = tsodyks_markram(N, dt=cell0.PARAM['dt'])
    synapses_out2 = DoubleExponentialSynapse(N, dt=cell0.PARAM['dt'], td=1e-2, tr=5e-3)

    #initialization
    # length of simulation [s]
    tmax=3
    # set the number of iterations
    number_of_iterations=int(tmax/cell0.PARAM['dt'])

    v0= np.zeros((number_of_iterations, N))
    spike = np.zeros(N)
    past_spike = np.zeros(N)
    random.seed(seed) # for reproducibility


    # set step input
    dt = cell0.PARAM['dt']
    delay = int(0.1/dt)
    I=np.zeros((number_of_iterations, N))
    I[int(0.0/dt):int(0.1/dt),0] = 0.09
    rasters = np.zeros((number_of_iterations, N))
    output = np.zeros((number_of_iterations+delay, N))
    cols = np.arange(output.shape[1])
    next_input = np.zeros(N)
    # delays = np.random.randint(0, delay, size=N)  # delay for each neuron
    delays = np.full(N, delay-1)  # uniform delay for all neurons
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
        next_input = output[i]  # update input for next iteration
        past_spike = spike
    #print(output.shape)



    end = time.perf_counter()
    print(f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}")

    # plot simulation result
    fig = plt.figure(num=2, figsize=(10,4))
    spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, hspace=0.1, height_ratios=[4, 4, 1])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    times, neuron_ids = np.nonzero(rasters)
    ax0.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], v0[:,0])    
    ax0.set_xlim(0, tmax)
    ax0.set_ylabel("v")
    ax0.set_xticks([])
    ax1.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], output[:number_of_iterations,0])
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("synapse output")
    ax1.set_xticks([])
    ax2.plot([i*cell0.PARAM['dt'] for i in range(0, number_of_iterations)], I[:,0], color="black")
    ax2.set_xlim(0, tmax)
    ax2.set_xlabel("[s]")
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
