import numpy as np
import glob
import os
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)  # HACK: 親ディレクトリをパスに追加
import GPU_SNN_simulation

os.makedirs("reservoir_outputs/train", exist_ok=True)
os.makedirs("reservoir_outputs/test", exist_ok=True)


# ================================
# SNN の出力特徴を保存する関数
# ================================
def process_and_save(
    input_dir: str, output_dir: str, label: str, num_of_cells: int
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))

    print(f"[{label}] {len(paths)} files found.")

    for path in tqdm(paths):
        coch = np.load(path)

        # ---- SNN 実行（特徴ベクトルを得る） ----
        feat = GPU_SNN_simulation.main(
            input_data=coch,
            label=label,
            return_feature=True,
            isDebugPrint=False,
            N=num_of_cells,
        )  # shape = (100,) or (200,)

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
def main() -> None:
    num_of_cells = 100
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
    main()
