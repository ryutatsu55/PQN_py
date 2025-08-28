from src import PQNModel
from src import LIF
from src import PSP_test
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
    # cell0=PQNModel(mode='RSexci', N = N)
    cell0=LIF(N=N)
    dt = 0.0001
    # dt = cell0.PARAM['dt']

    # set a synapse
    synapses_out1 = tsodyks_markram(N, dt=dt, tau_rec=0.1, U=0.1)
    synapses_out2 = DoubleExponentialSynapse(N, dt=dt, td=1e-2, tr=5e-3)

    #initialization
    # length of simulation [s]
    tmax=2
    # set the number of iterations
    number_of_iterations=int(tmax/dt)

    v0= np.zeros((number_of_iterations, N))
    spike = np.zeros(N)
    past_spike = np.zeros(N)
    random.seed(seed) # for reproducibility


    # set step input
    delay = int(0.001/dt)
    delays = np.full(N, delay-1)  # uniform delay for all neurons
    # delays = np.random.randint(0, delay, size=N)  # delay for each neuron
    I=np.zeros((number_of_iterations, N))
    I[int(0.0/dt):int(0.075/dt),0] = 0.13
    rasters = np.zeros((number_of_iterations, N))
    output = np.zeros((number_of_iterations+delay, N))
    cols = np.arange(output.shape[1])
    next_input = np.zeros(N)

    # run simulatiion
    start = time.perf_counter()
    for i in tqdm(range(number_of_iterations)):
        I[i] += next_input
        rasters[i], v0[i] = cell0.calc(inputs=I[i], itr=i)  # update cell state
        rows = delays + i+1
        output[rows, cols] = 70*synapses_out1(rasters[i])  # [nA]
        next_input = output[i]  # update input for next iteration
    end = time.perf_counter()
    print(f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}")

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
