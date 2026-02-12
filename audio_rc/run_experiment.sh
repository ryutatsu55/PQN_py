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
BASE_DIR="audio_rc"
ARCHIVE_DIR="${BASE_DIR}/archive/${EXP_NAME}"
CONFIG_SRC="config.py" # 環境に合わせてパスを調整してください
RESULT_SRC="${BASE_DIR}/result"

# ディレクトリ作成
mkdir -p "${ARCHIVE_DIR}"

# ログファイルパスの定義
LOG_FILE="${ARCHIVE_DIR}/execution.log"

# 以降の全出力を「画面」と「ファイル」の両方に出力する設定
# >(tee ...) プロセス置換を使って出力を分岐
# 2>&1 でエラー出力もログに含める
# exec > >(tee -a "${LOG_FILE}") 2>&1
exec > >(tee >(sed "s/"$'\033'"\[[0-9;]*m//g" >> "${LOG_FILE}")) 2>&1

echo -e "${GREEN}[INFO] ログの記録を開始します: ${LOG_FILE}${RESET}"


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

if [ -f "${CONFIG_SRC}" ]; then
    cp "${CONFIG_SRC}" "${ARCHIVE_DIR}/RNN_config_snapshot.py"
    log_info "設定ファイルをスナップショットとして保存しました"
else
    log_warn "設定ファイル (${CONFIG_SRC}) が見つかりません。バックアップをスキップします。"
fi

# --- タスク定義 ---
TASKS=(
    "f2f_digit" 
    "m2m_digit" 
    "z2z_gender" 
    "o2o_gender" 
    "m2f_digit" 
    "f2m_digit" 
    "z2o_gender" 
    "o2z_gender" 
    "reverse_f_z"
)

section_header "実験を開始します: ${EXP_NAME}"
log_info "対象タスク数: ${#TASKS[@]}"
log_info "結果出力先: ${RESULT_SRC}"
log_info "結果保存先: ${ARCHIVE_DIR}"

# --- メインループ ---
for task in "${TASKS[@]}"; do
    section_header "Running Task: ${task}"
    
    # 1. Pythonスクリプト実行
    # 結果フォルダはスクリプト内で毎回初期化(削除)されます
    PYTHONUNBUFFERED=1 ${PYTHON_EXEC} ${BASE_DIR}/src/speech_recognition.py \
        --mode snn \
        --task "${task}"

    # 2. 結果の退避
    TASK_ARCHIVE="${ARCHIVE_DIR}/${task}"
    mkdir -p "${TASK_ARCHIVE}"
    
    if [ -d "${RESULT_SRC}" ]; then
        log_info "結果をアーカイブに保存中: ${TASK_ARCHIVE}"
        cp -r "${RESULT_SRC}/." "${TASK_ARCHIVE}"
    else
        echo -e "${RED}[ERROR] 結果ディレクトリが見つかりません。${RESET}"
    fi
done

# --- 4. 完了 (Completion) ---

echo -e "\n${GREEN}✔ すべての処理が正常に完了しました！${RESET}"
echo -e "  📂 結果ディレクトリ: ${ARCHIVE_DIR}"
echo -e "  📄 設定ファイル: ${ARCHIVE_DIR}/RNN_config_snapshot.py\n"