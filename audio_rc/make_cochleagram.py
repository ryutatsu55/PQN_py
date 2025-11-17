import glob
import subprocess
import librosa
from lyon.calc import LyonCalc
from multiprocessing import Pool, cpu_count
import os
import numpy as np


def compute_cochleagram(audio_path: str, decimation_factor: int = 64) -> np.ndarray:
    """Load an audio file and compute its cochleagram."""

    y, sr = librosa.load(audio_path, sr=None)
    y = y.astype(np.float64)  # LyonCalc expects float64 input

    calc = LyonCalc()
    coch = calc.lyon_passive_ear(
        y,
        int(sr),
        decimation_factor=decimation_factor,
        step_factor=0.195,  # 微調整して100chにした
    )
    return coch


def preprocess_with_lyon(audio_path: str, decimation_factor: int = 64) -> np.ndarray:
    """
    Load a wav file, run Lyon's auditory model, and return the cochleagram.
    Parameters:
        wav_path: path to the .wav file
        decimation_factor: integer factor to decimate time axis (for computational cost)
    Returns:
        coch: np.ndarray of shape (T’, N_channels)
            where T’ ≈ (num_samples / decimation_factor), N_channels ≈ 86 (in default)
    """
    # 1. 音声読み込み　Lyon モデルの計算
    coch = compute_cochleagram(audio_path, decimation_factor)
    # coch の形状例： (T’, 86) など :contentReference[oaicite:5]{index=5}
    print(f"Cochleagram shape: {coch.shape}")

    # 2. 必要に応じて正規化や切り出しなどを実施
    # 例：チャネルごと平均0／分散1 にする
    coch_norm = (coch - np.mean(coch, axis=0, keepdims=True)) / (
        np.std(coch, axis=0, keepdims=True) + 1e-9
    )

    return coch_norm


def sph_to_wav(sph_path: str) -> str:
    """
    Convert .sph file to .wav using 'sox' command.
    Returns path to the generated .wav file.
    """
    wav_path = sph_path[:-4] + ".wav"
    if not os.path.exists(wav_path):
        # Run sox only if wac does not exist
        subprocess.run(["sox", sph_path, wav_path], check=True)
    return wav_path


def _process_sph_worker(args):
    """
    Worker function for multiprocessing.
    args: (sph_path, out_dir, decimation_factor, delete_wav)
    """
    sph_path, out_dir, decimation_factor, delete_wav = args
    basename = os.path.basename(sph_path).replace(".sph", "")
    wav_path = None
    try:
        print(f"[WORKER] Processing {basename}")
        wav_path = sph_to_wav(sph_path)
        coch = preprocess_with_lyon(wav_path, decimation_factor=decimation_factor)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"coch_{basename}.npy")
        np.save(out_path, coch)
        return out_path
    except Exception as e:
        print(f"[ERROR] Failed to process {sph_path}: {e}")
        return None
    finally:
        # Optionally delete wav file after processing
        if delete_wav and wav_path is not None and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError as e:
                print(f"[WARN] Failed to remove {wav_path}: {e}")


def build_dataset_tensor(npy_paths, out_path: str):
    """
    Build a single big tensor for reservoir input from multiple cochleagram npy files.
    Padding in time dimension is applied to match the maximum length.
    Saves npz file with:
        X: shape (num_files, T_max, N_channels)
        lengths: shape (num_files,) original time lengths
        filenames: shape (num_files,) basename of each file
    """
    if len(npy_paths) == 0:
        raise ValueError("No npy files are provided to build dataset.")

    # Load first file to get N_channels
    first = np.load(npy_paths[0])
    if first.ndim != 2:
        raise ValueError("Each cochleagram array must be 2D (T, N_channels).")

    num_files = len(npy_paths)
    N_channels = first.shape[1]
    lengths = []
    max_T = first.shape[0]

    # First pass: compute lengths and max_T
    lengths.append(first.shape[0])

    for path in npy_paths[1:]:
        arr = np.load(path)
        if arr.shape[1] != N_channels:
            raise ValueError(
                f"Channel dimension mismatch in {path}: "
                f"expected {N_channels}, got {arr.shape[1]}"
            )
        T = arr.shape[0]
        lengths.append(T)
        if T > max_T:
            max_T = T

    # Allocate big tensor
    X = np.zeros((num_files, max_T, N_channels), dtype=np.float32)
    filenames = []

    # Second pass: fill tensor
    for idx, path in enumerate(npy_paths):
        arr = np.load(path)
        T = arr.shape[0]
        X[idx, :T, :] = arr  # zero padding for the rest
        filenames.append(os.path.basename(path))

    lengths = np.array(lengths, dtype=np.int32)
    filenames = np.array(filenames, dtype=object)

    # Save as npz
    np.savez(out_path, X=X, lengths=lengths, filenames=filenames)
    print(f"[DATASET] Saved dataset to {out_path}")
    print(f"          shape: X={X.shape}, lengths.shape={lengths.shape}")


def preprocess_all_sph_for_words(
    train_root: str,
    out_dir: str = "coch_output",
    word_codes: list[str] | None = None,
    decimation_factor: int = 64,
    delete_wav: bool = True,
    n_jobs: int | None = None,
    build_big_tensor: bool = True,
    big_tensor_name: str = "dataset_all_words.npz",
):
    """
    Preprocess all .sph files under TI46/TI20/TRAIN for given word codes.

    Parameters:
        train_root:
            Root directory for TI20 TRAIN. e.g. ".../TI46/TI20/TRAIN"
        out_dir:
            Directory where individual cochleagram npy files will be saved.
        word_codes:
            List of 2-character word codes to filter.
            e.g. ["00"] for ZERO,
                 [f"{i:02d}" for i in range(10)] for digits 0-9.
            If None, all words are processed.
        decimation_factor:
            Decimation factor passed to Lyon model.
        delete_wav:
            If True, delete intermediate wav files after processing.
        n_jobs:
            Number of parallel processes. If None, use cpu_count().
        build_big_tensor:
            If True, build a single big tensor npz from all npy files.
        big_tensor_name:
            Filename of the npz dataset (saved under out_dir).
    """
    # Discover all .sph files: TRAIN/(F1..F8,M1..M8)/*.sph
    pattern = os.path.join(train_root, "*", "*.sph")
    sph_files = sorted(glob.glob(pattern))

    if len(sph_files) == 0:
        raise FileNotFoundError(f"No .sph files found under: {pattern}")

    # Filter by word codes if specified
    if word_codes is not None:
        codes_lower = {c.lower() for c in word_codes}
        filtered = []
        for path in sph_files:
            base = os.path.basename(path)
            word_code = base[:2].lower()  # first 2 chars indicate word code
            if word_code in codes_lower:
                filtered.append(path)
        sph_files = filtered

    if len(sph_files) == 0:
        raise ValueError("No .sph files matched the given word_codes.")

    print(f"[INFO] Found {len(sph_files)} .sph files to process.")
    os.makedirs(out_dir, exist_ok=True)

    # Prepare args for worker
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)

    worker_args = [
        (sph_path, out_dir, decimation_factor, delete_wav) for sph_path in sph_files
    ]

    print(f"[INFO] Using {n_jobs} parallel workers.")

    # Run multiprocessing
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_process_sph_worker, worker_args)

    # Collect successfully processed npy paths
    npy_paths = [p for p in results if p is not None]
    print(f"[INFO] Successfully processed {len(npy_paths)} files.")

    # Build big tensor for reservoir if requested
    if build_big_tensor and len(npy_paths) > 0:
        dataset_path = os.path.join(out_dir, big_tensor_name)
        build_dataset_tensor(npy_paths, dataset_path)

    return npy_paths


if __name__ == "__main__":
    audio_root = "./audio/"
    folders = ["train", "test"]

    configs = [
        ("00", "zero"),
        ("01", "one"),
    ]

    for folder in folders:
        for code, name in configs:
            preprocess_all_sph_for_words(
                train_root=audio_root + folder,
                out_dir=f"reservoir_inputs/{folder}/coch_{name}",  # 出力先フォルダ
                word_codes=[code],  # 指定した単語コードのみ
                decimation_factor=64,
                delete_wav=True,  # .wav は一時ファイルとして削除
                n_jobs=None,  # CPU数 - 1 を自動で使用
                build_big_tensor=True,
                big_tensor_name=f"dataset_{name}.npz",
            )

    print("Done preprocessing.")
