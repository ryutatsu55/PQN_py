import numpy as np
import random
import os

dt = 0.01  # シミュレーションのタイムステップ (例えば10ms)
n_neurons = 40
duration_stim = int(0.1 / dt) # 100ms
duration_interval = int(0.5 / dt) # 2.5s
Ntrain = 100
Ntest = 100
dirs_to_create = [
    "RNN_analyze/reservoir_inputs/train/top",
    "RNN_analyze/reservoir_inputs/train/middle",
    "RNN_analyze/reservoir_inputs/train/bottom",
    "RNN_analyze/reservoir_inputs/test/top",
    "RNN_analyze/reservoir_inputs/test/middle",
    "RNN_analyze/reservoir_inputs/test/bottom"
]
for directory in dirs_to_create:
    os.makedirs(directory, exist_ok=True)

for i in range(Ntrain):
    input = np.zeros((duration_interval, 3*n_neurons), dtype=float)
    target = np.random.randint(0,3)
    input[:duration_stim, target*n_neurons:(target+1)*n_neurons] = 0.8
    # print(input.shape)
    if target == 0:
        np.save(f"RNN_analyze/reservoir_inputs/train/top/{i}.npy", input)
    elif target == 1:
        np.save(f"RNN_analyze/reservoir_inputs/train/middle/{i}.npy", input)
    elif target == 2:
        np.save(f"RNN_analyze/reservoir_inputs/train/bottom/{i}.npy", input)


for i in range(Ntest):
    input = np.zeros((duration_interval, 3*n_neurons), dtype=float)
    target = np.random.randint(0,3)
    input[:duration_stim, target*n_neurons:(target+1)*n_neurons] = 0.8
    if target == 0:
        np.save(f"RNN_analyze/reservoir_inputs/test/top/{i}.npy", input)
    elif target == 1:
        np.save(f"RNN_analyze/reservoir_inputs/test/middle/{i}.npy", input)
    elif target == 2:
        np.save(f"RNN_analyze/reservoir_inputs/test/bottom/{i}.npy", input)


# 入力スケジュールの生成例
# inputs: (Time, Input_Dimension=3)
# targets: (Time, Output_Dimension=3)