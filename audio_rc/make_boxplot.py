import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_acc_values(jsonl_path: Path) -> list[float]:
    """Read acc_test values from a jsonl file."""
    values: list[float] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "acc_test" in record:
                values.append(record["acc_test"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make boxplots of acc_test from jsonl files in audio_rc/results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a jsonl file or a directory containing jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "figs" / "acc_test_boxplot.png",
        help="Output path for the boxplot image.",
    )
    args = parser.parse_args()

    input_path: Path = args.input

    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"{input_path} is not a jsonl file.")
        jsonl_files = [input_path]

    elif input_path.is_dir():
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No jsonl files found in {input_path}")
    else:
        raise FileNotFoundError(f"{input_path} does not exist.")

    labels: list[str] = []
    data: list[list[float]] = []

    for jf in jsonl_files:
        accs = load_acc_values(jf)
        if not accs:
            continue
        labels.append(jf.stem)
        data.append(accs)

    if not data:
        raise ValueError("No acc_test values found in provided jsonl files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(max(6, 1.2 * len(data)), 5))
    plt.boxplot(
        data,
        patch_artist=True,
        boxprops=dict(facecolor="#a7c7e7", edgecolor="#1f4f82"),
        medianprops=dict(color="#1f4f82"),
        whiskerprops=dict(color="#1f4f82"),
        capprops=dict(color="#1f4f82"),
    )
    plt.ylim(0, 1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ylabel("acc_test")
    plt.title("acc_test Boxplot")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved boxplot to {args.output}")


if __name__ == "__main__":
    main()
