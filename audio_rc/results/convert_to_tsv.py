import json
from pathlib import Path

# 対象の jsonl ファイル（順番＝列の順番）
files = [
    "results_RSexci.jsonl",
    "results_RSinhi.jsonl",
    "results_FS.jsonl",
    "results_LTS.jsonl",
    "results_IB.jsonl",
    "results_EB.jsonl",
    "results_PB.jsonl",
]

# 各ファイルの acc_test を読み込む
all_acc = []

for file in files:
    acc_list = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "acc_test" in d:
                acc_list.append(d["acc_test"])
    all_acc.append(acc_list)

# 最大行数に合わせる
max_len = max(len(col) for col in all_acc)

# ヘッダー（ファイル名）
print("\t".join(Path(f).stem for f in files))

# 行ごとに出力
for i in range(max_len):
    row = []
    for col in all_acc:
        row.append(str(col[i]) if i < len(col) else "")
    print("\t".join(row))
