#!/bin/bash

CELLS_LIST=(24 52 80 100 120 160 200)

MODE="snn"

LOG_DIR="audio_rc/logs"
mkdir -p "$LOG_DIR"

for N in "${CELLS_LIST[@]}"; do
    echo "========================================"
    echo " Running experiment: N = $N"
    echo " Mode: $MODE"
    echo "========================================"

    LOGFILE="${LOG_DIR}/run_N${N}_$(date '+%Y%m%d_%H%M%S').log"

    python audio_rc/train_snn_readout.py \
        --mode "$MODE" \
        --cells "$N" \
        2>&1 | tee "$LOGFILE"

    echo "Saved log to: $LOGFILE"
    echo ""
done

echo "All experiments completed."
