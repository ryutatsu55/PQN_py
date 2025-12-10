#!/bin/bash

# === SETTINGS ==========================================================
WORKDIR="/home/tanii/yamada/PQN_py"   # 実際のプロジェクトの絶対パスに変更
LOGDIR="$WORKDIR/audio_rc/logs"
mkdir -p "$LOGDIR"

timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/run_background_${timestamp}.log"
PIDFILE="$LOGDIR/run_background_${timestamp}.pid"

# === RUN ===============================================================
echo "Starting job at $(date)" | tee -a "$LOGFILE"

cd "$WORKDIR"

# setsid + nohup で VSCode セッションから完全に切り離して実行
setsid nohup python3 ./audio_rc/train_snn_readout.py \
    --mode snn -c 48 \
    > "$LOGFILE" 2>&1 < /dev/null &

PID=$!
echo $PID > "$PIDFILE"

echo "Started background job."
echo "  PID: $PID"
echo "  LOG: $LOGFILE"
echo "  PID FILE: $PIDFILE"
echo "You can close VSCode or terminal safely."
