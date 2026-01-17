import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from pathlib import Path
from src import PQNModel

def run_and_save(mode, I_array, tmax, filename_base):
    cell = PQNModel(mode=mode)
    dt = cell.PARAM['dt']
    num_iters = len(I_array)
    
    time_axis = np.arange(num_iters) * dt

    v0 = []
    for i in range(num_iters):
        cell.update(I_array[i])
        v0.append(cell.get_membrane_potential())

    fig = plt.figure(figsize=(8, 4))
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.1, height_ratios=[4, 1])
    
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])

    ax0.plot(time_axis, v0)
    ax0.set_xlim(0, tmax)
    ax0.set_ylabel("v")
    ax0.set_xticks([])

    ax1.plot(time_axis, I_array, color="black")
    ax1.set_xlim(0, tmax)
    ax1.set_xlabel("[s]")
    ax1.set_ylabel("I")

    plt.savefig(f"data/{mode}/img/{filename_base}.png")
    plt.close(fig)

    df = pd.DataFrame({
        'Time [s]': time_axis,
        'Voltage [v]': v0,
        'Current [I]': I_array
    })
    df.to_csv(f"data/{mode}/csv/{filename_base}.csv")

if __name__ == "__main__":
    classes = ["RSexci", "RSinhi", "FS", "LTS", "IB", "EB", "PB", "Class2"]
    tmax = 2
    I_input = "0.60"

    for mode in classes:
        Path(f"data/{mode}/img").mkdir(parents=True, exist_ok=True)
        Path(f"data/{mode}/csv").mkdir(parents=True, exist_ok=True)

        cell_tmp = PQNModel(mode=mode)
        num_iters = int(tmax / cell_tmp.PARAM['dt'])

        I_step = np.zeros(num_iters)
        I_step[num_iters // 4 : num_iters // 4 * 3] = float(I_input)
        run_and_save(mode, I_step, tmax, f"{I_input}_step")

        I_const = np.full(num_iters, float(I_input))
        run_and_save(mode, I_const, tmax, f"{I_input}")
