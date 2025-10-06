from src import PSP_test
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import time
from src.Synapses import tsodyks_markram
import random

if __name__ == "__main__":
    seed = int(random.random() * 1000)
    N=1
    cell0=PSP_test(N=N)
    dt = 0.0001

    # set a synapse
    synapses_out1 = tsodyks_markram(N, dt=dt, tau_rec=0.5, U=0.5)

    #initialization
    # length of simulation [s]
    tmax=1.5
    # set the number of iterations
    number_of_iterations=int(tmax/dt)

    v0= np.zeros((number_of_iterations, N))
    spike = np.zeros(N)
    past_spike = np.zeros(N)
    random.seed(seed) # for reproducibility


    # set step input
    I=np.zeros((number_of_iterations, N))
    freq = 20   #[Hz]
    rasters = np.zeros((number_of_iterations, N))
    span = int((1/freq)//dt)
    rasters[::span, :] = 1
    rasters[:2000, :] = 0
    rasters[number_of_iterations-2000:, :] = 0
    output = np.zeros((number_of_iterations, N))
    cols = np.arange(output.shape[1])
    next_input = np.zeros(N)
    # synapses_out1.mask_faci[0]=1
    # synapses_out1.U[0]=0
    # synapses_out1.tau_inact[0]=0.0015
    # synapses_out1.tau_rec[0]=0.13

    # run simulatiion
    start = time.perf_counter()
    for i in tqdm(range(number_of_iterations)):
        I[i] += next_input.flatten()
        v0[i] = cell0.calc(inputs=I[i], itr=i)  # update cell state
        next_input = 0.25 * synapses_out1(rasters[i])  # [nA]
    end = time.perf_counter()
    print(f"processing time for {tmax}s simulation mas {(end - start)} s when reservoir_size was {N}")

    # plot simulation result
    fig = plt.figure(num=2, figsize=(10,4))
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.1, height_ratios=[1, 4])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    times, neuron_ids = np.nonzero(rasters)
    ax0.set_xticks([])
    ax0.plot([i*dt for i in range(0, number_of_iterations)], I[:,0], color="black")
    ax0.set_xlim(0, tmax)
    ax1.plot([i*dt for i in range(0, number_of_iterations)], v0[:,0])    
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("v")
    ax0.set_ylabel("I")
    ax1.set_xlabel("[s]")
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
