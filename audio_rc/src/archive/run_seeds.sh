#!/bin/bash

CELLS=48

# デフォルトのシード範囲
DEFAULT_SEEDS=(0 1 2 3 4 5 6 7 8 9)

# 引数で範囲指定があれば利用
if [[ -n "${1:-}" && -n "${2:-}" ]]; then
    START_SEED=$1
    END_SEED=$2

    # 数値チェック
    if ! [[ "$START_SEED" =~ ^-?[0-9]+$ && "$END_SEED" =~ ^-?[0-9]+$ ]]; then
        echo "Error: start と end は整数で指定してください。" >&2
        exit 1
    fi

    if (( START_SEED >= END_SEED )); then
        echo "Error: end は start より大きい値を指定してください (end は非包含)。" >&2
        exit 1
    fi

    LAST_SEED=$((END_SEED - 1))
    mapfile -t SEEDS < <(seq "$START_SEED" "$LAST_SEED")
else
    SEEDS=("${DEFAULT_SEEDS[@]}")
fi

LOGDIR="audio_rc/logs/run_seeds"
mkdir -p "${LOGDIR}"

for SEED in "${SEEDS[@]}"; do
    echo "===== Running seed $SEED ====="

    # 日時の取得
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # ログファイル名
    LOGFILE="${LOGDIR}/run_${SEED}_${TIMESTAMP}.log"

    echo "Logging to: $LOGFILE"

    # Python 実行＋標準出力と標準エラーをログ保存
    python audio_rc/train_snn_readout.py \
        --mode snn \
        --cells $CELLS \
        --seed $SEED \
        >  "$LOGFILE" 2>&1

    echo "Finished seed $SEED"
done

echo "All seeds finished!"
