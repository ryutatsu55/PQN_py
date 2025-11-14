import numpy as np
import glob
import os
from tqdm import tqdm
import GPU_SNN_simulation  # あなたの SNN main()


# ================================
# SNN の出力特徴を保存する関数
# ================================
def process_and_save(input_dir, output_dir, label):
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
            isDebugPrint=False
        )  # shape = (100,) or (200,)

        # ---- 保存するファイル名 ----
        filename = os.path.basename(path)          # ex: zero_01.npy
        save_path = os.path.join(output_dir, filename)

        # ---- 保存 ----
        if feat is not None:
            np.save(save_path, feat)

    print(f"[{label}] Saved features to: {output_dir}")


# ================================
# メイン処理
# ================================
def main():
    process_and_save(
        input_dir="inputs/coch_zero",
        output_dir="outputs/features_zero",
        label="zero"
    )

    process_and_save(
        input_dir="inputs/coch_one",
        output_dir="outputs/features_one",
        label="one"
    )


if __name__ == "__main__":
    main()
