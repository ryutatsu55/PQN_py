#!/bin/bash

# ==============================================================================
#  RNN Reservoir Computing Experiment Runner
#  Usage: ./run_experiment.sh [Optional: Experiment_Description]
# ==============================================================================

# --- 1. 設定 & 前準備 (Configuration) ---

# エラーが起きたら即停止 (set -e), 未定義変数はエラー扱い (set -u)
set -euo pipefail

# カラー定義 (見た目をかっこよくする)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Pythonインタプリタの指定 (必要なら .venv/bin/python などに変更)
PYTHON_EXEC="python"

# タイムスタンプ取得 (実験IDとする)
DATESTAMP=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%H%M")

# 引数がある場合は実験名に追加
if [ -z "${1:-}" ]; then
    EXP_NAME="${DATESTAMP}/${TIMESTAMP}"
else
    EXP_NAME="${DATESTAMP}/${TIMESTAMP}_${1}"
fi

# 結果保存ディレクトリ
BASE_DIR="RNN_analyze"
RESULTS_DIR="${BASE_DIR}/archive/${EXP_NAME}"
CONFIG_SRC="config.py" # 環境に合わせてパスを調整してください

# --- 2. ユーティリティ関数 (Helper Functions) ---

log_info() {
    echo -e "${GREEN}[INFO] $(date +'%H:%M:%S') ➜ $1${RESET}"
}

log_warn() {
    echo -e "${YELLOW}[WARN] $(date +'%H:%M:%S') ➜ $1${RESET}"
}

log_error() {
    echo -e "${RED}[ERROR] $(date +'%H:%M:%S') ➜ $1${RESET}"
}

section_header() {
    echo -e "\n${CYAN}======================================================${RESET}"
    echo -e "${CYAN}   $1${RESET}"
    echo -e "${CYAN}======================================================${RESET}"
}

# エラー時のトラップ処理
error_handler() {
    log_error "スクリプトが異常終了しました。直前のコマンドを確認してください。"
}
trap error_handler ERR

# --- 3. メイン処理 (Main Execution) ---

section_header "実験を開始します: ${EXP_NAME}"
log_info "結果出力先: ${RESULTS_DIR}"

# ディレクトリ作成
mkdir -p "${RESULTS_DIR}"

# Step 1: 相関行列の解析
section_header "Step 1: Analyzing Correlation Matrix"
log_info "Running corr_neuron.py..."
${PYTHON_EXEC} ${BASE_DIR}/src/corr_neuron.py

# Step 2: データ生成
section_header "Step 2: Input Data Generation"
log_info "Running make_spatial_input.py..."
${PYTHON_EXEC} ${BASE_DIR}/src/make_spatial_input.py

# Step 3: 時空間認識 (SNNモード)
section_header "Step 3: Spatial Recognition (SNN Mode)"
log_info "Running recognition_test.py..."
# tqdmの表示が崩れないようにPYTHONUNBUFFERED=1をつけるのがコツ
PYTHONUNBUFFERED=1 ${PYTHON_EXEC} ${BASE_DIR}/src/recognition_test.py \
    --mode snn \
    --classifier both

# # Step 3: 時空間認識 (Featureモード)
# section_header "Step 3: Spatiotemporal Recognition (Feature Mode)"
# log_info "Running spatiotemp_recognition.py..."
# PYTHONUNBUFFERED=1 ${PYTHON_EXEC} RNN_analyze/spatiotemp_recognition.py \
#     --mode feature\
#     --classifier both

# Step 4: 結果の集約
section_header "Step 4: Archiving Results"

# 生成されたデータや画像を結果フォルダに移動/コピー
# (実際の保存先パスに合わせて調整してください)
DATA_SRC="${BASE_DIR}/result"

if [ -d "${DATA_SRC}" ]; then
    log_info "データを結果フォルダにアーカイブ中..."
    cp -r "${DATA_SRC}/." "${RESULTS_DIR}" 2>/dev/null || true
fi
# 設定ファイルのバックアップ (再現性の確保)
if [ -f "${CONFIG_SRC}" ]; then
    cp "${CONFIG_SRC}" "${RESULTS_DIR}/RNN_config_snapshot.py"
    log_info "設定ファイルをスナップショットとして保存しました"
else
    log_warn "設定ファイル (${CONFIG_SRC}) が見つかりません。バックアップをスキップします。"
fi

# --- 4. 完了 (Completion) ---

echo -e "\n${GREEN}✔ すべての処理が正常に完了しました！${RESET}"
echo -e "  📂 結果ディレクトリ: ${RESULTS_DIR}"
echo -e "  📄 設定ファイル: ${RESULTS_DIR}/RNN_config_snapshot.py\n"