#!/bin/bash

# ==============================================================================
#  RNN Reservoir Computing Experiment Runner (Batch Seed Processing)
# ==============================================================================

# エラーが起きたら即停止 (set -e), 未定義変数はエラー扱い (set -u)
set -euo pipefail

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

PYTHON_EXEC="python"

DATESTAMP=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%H%M")

if [ -z "${1:-}" ]; then
    EXP_NAME="${DATESTAMP}/${TIMESTAMP}"
else
    EXP_NAME="${DATESTAMP}/${1}"
fi

# これが「親」ディレクトリになります (例: archive/20240101/1200_ExpName)
PARENT_ARCHIVE_DIR="archive/${EXP_NAME}"
SPEECH_DIR="audio_rc"
SPATIO_DIR="RNN_analyze"
CONFIG_SRC="./config.py"
SPEECH_RESULT="${SPEECH_DIR}/result"
SPATIO_RESULT="${SPATIO_DIR}/result"
# 親ディレクトリ作成
mkdir -p "${PARENT_ARCHIVE_DIR}"

# ログ設定: 親ディレクトリに execution.log を作成
LOG_FILE="${PARENT_ARCHIVE_DIR}/execution.log"
exec > >(tee >(sed "s/"$'\033'"\[[0-9;]*m//g" >> "${LOG_FILE}")) 2>&1

echo -e "${GREEN}[INFO] ログの記録を開始します: ${LOG_FILE}${RESET}"

# --- ユーティリティ関数 ---
log_info() { echo -e "${GREEN}[INFO] $(date +'%H:%M:%S') ➜ $1${RESET}"; }
log_warn() { echo -e "${YELLOW}[WARN] $(date +'%H:%M:%S') ➜ $1${RESET}"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S') ➜ $1${RESET}"; }
section_header() { 
    echo -e "\n${CYAN}======================================================${RESET}"
    echo -e "${CYAN}   $1${RESET}"
    echo -e "${CYAN}======================================================${RESET}"
}
error_handler() { log_error "スクリプトが異常終了しました。直前のコマンドを確認してください。"; }
trap error_handler ERR

# --- メイン処理 ---

# 設定ファイルのスナップショット保存 (親ディレクトリに1つだけ保存)
if [ -f "${CONFIG_SRC}" ]; then
    cp "${CONFIG_SRC}" "${PARENT_ARCHIVE_DIR}/config_snapshot.py"
    log_info "設定ファイルをスナップショットとして保存しました: ${PARENT_ARCHIVE_DIR}/config_snapshot.py"
else
    log_warn "設定ファイル (${CONFIG_SRC}) が見つかりません。"
fi

# 対象タスク
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

section_header "実験を開始します: ${EXP_NAME} (Seed 1-21)"
log_info "結果保存用ベースディレクトリ: ${PARENT_ARCHIVE_DIR}"

# ==========================================
#  SEED LOOP (1 to 21)
# ==========================================
for SEED in {1..21}; do
    section_header "Processing SEED: ${SEED}"
    
    # シードごとの保存先フォルダ (例: .../seed_1)
    SEED_ARCHIVE_DIR="${PARENT_ARCHIVE_DIR}/seed_${SEED}"
    mkdir -p "${SEED_ARCHIVE_DIR}"

    # clean up previous results
    rm -rf "${SPATIO_RESULT}"/*

    # Step 1: 相関行列の解析
    log_info "Running corr_neuron.py..."
    PYTHONUNBUFFERED=1 ${PYTHON_EXEC} ${SPATIO_DIR}/src/corr_neuron.py \
        --seed "${SEED}"

    # Step 2: データ生成
    log_info "Running make_spatial_input.py..."
    PYTHONUNBUFFERED=1 ${PYTHON_EXEC} ${SPATIO_DIR}/src/make_spatial_input.py \
        --seed "${SEED}"

    # Step 3: 時空間認識 (SNNモード)
    log_info "Running recognition_test.py..."
    PYTHONUNBUFFERED=1 ${PYTHON_EXEC} ${SPATIO_DIR}/src/recognition_test.py \
        --mode snn \
        --classifier both \
        --seed "${SEED}"

    ANALYZE_ARCHIVE="${SEED_ARCHIVE_DIR}/spatio_temp"
    mkdir -p "${ANALYZE_ARCHIVE}"
    
    if [ -d "${SPATIO_RESULT}" ]; then
        # log_info "${task} の結果を保存中..."
        cp -r "${SPATIO_RESULT}/." "${ANALYZE_ARCHIVE}"
    else
        echo -e "${RED}[ERROR] 結果ディレクトリが見つかりません。${RESET}"
    fi

    # タスクごとのループ
    for task in "${TASKS[@]}"; do
        section_header "Running Task: ${task} (Seed: ${SEED})"
        
        # Python実行 (--seed を渡す)
        PYTHONUNBUFFERED=1 ${PYTHON_EXEC} ${SPEECH_DIR}/src/speech_recognition.py \
            --mode snn \
            --task "${task}" \
            --seed "${SEED}"

        # 結果の退避
        TASK_ARCHIVE="${SEED_ARCHIVE_DIR}/${task}"
        mkdir -p "${TASK_ARCHIVE}"
        
        if [ -d "${SPEECH_RESULT}" ]; then
            # log_info "${task} の結果を保存中..."
            cp -r "${SPEECH_RESULT}/." "${TASK_ARCHIVE}"
        else
            echo -e "${RED}[ERROR] 結果ディレクトリが見つかりません。${RESET}"
        fi
    done
done

echo -e "\n${GREEN}✔ 全シード・全タスクの処理が完了しました！${RESET}"
echo -e "  📂 結果ディレクトリ: ${PARENT_ARCHIVE_DIR}"