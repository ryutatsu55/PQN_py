import argparse
import numpy as np
import glob
import os
from tqdm import tqdm

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
import src.PQN_RNN_onGPU as PQN_RNN

os.makedirs("reservoir_outputs/train", exist_ok=True)
os.makedirs("reservoir_outputs/test", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cells",
    "-c",
    type=int,
    default=100,
    help="number of reservoir cells (default: 100)",
)
args = parser.parse_args()


# ================================
# SNN の出力特徴を保存する関数
# ================================
def process_and_save(
    input_dir: str, output_dir: str, label: str, num_of_cells: int
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))

    print(f"[{label}] {len(paths)} files found.")

    for i, path in enumerate(tqdm(paths)):
        coch = np.load(path)

        density = 0.1
        # ---- SNN 実行（特徴ベクトルを得る） ----
        is_last_iteration = (i == len(paths) - 1)
        # feat = GPU_SNN_simulation.main(
        #     input_data=coch,
        #     label=label,
        #     return_feature=True,
        #     isDebugPrint=False,
        #     Nin=num_of_cells,
        #     density,
        #     N = num_of_cells,
        #     record=is_last_iteration,
        # )  # shape = (100,) or (200,)

        M = coch.shape[1]
        W_in = np.random.uniform(0.0, 1.0, size=(M, num_of_cells)).astype(np.float32)
        ch_mask = np.random.choice([0, 1], size=(M, num_of_cells),p=[1-density, density]).astype(np.float32)
        W_in = W_in * ch_mask
        projected_input = coch @ W_in
        feat = coch.mean(axis=0)  # ダミー実装

        # ---- 保存するファイル名 ----
        filename = os.path.basename(path)  # ex: zero_01.npy
        save_path = os.path.join(output_dir, filename)

        # ---- 保存 ----
        if feat is not None:
            np.save(save_path, feat)

    print(f"[{label}] Saved features to: {output_dir}")


# ================================
# メイン処理
# ================================
def main(num_of_cells: int) -> None:
    # TRAIN
    process_and_save(
        input_dir="audio_rc/reservoir_inputs/train/coch_zero",
        output_dir="audio_rc/reservoir_outputs/train/features_zero",
        label="zero",
        num_of_cells=num_of_cells,
    )
    process_and_save(
        input_dir="audio_rc/reservoir_inputs/train/coch_one",
        output_dir="audio_rc/reservoir_outputs/train/features_one",
        label="one",
        num_of_cells=num_of_cells,
    )

    # TEST
    process_and_save(
        input_dir="audio_rc/reservoir_inputs/test/coch_zero",
        output_dir="audio_rc/reservoir_outputs/test/features_zero",
        label="zero",
        num_of_cells=num_of_cells,
    )
    process_and_save(
        input_dir="audio_rc/reservoir_inputs/test/coch_one",
        output_dir="audio_rc/reservoir_outputs/test/features_one",
        label="one",
        num_of_cells=num_of_cells,
    )


if __name__ == "__main__":
    num_of_cells = args.cells
    main(num_of_cells)
