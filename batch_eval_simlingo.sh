#!/bin/bash
# ============================================================
# SimLingo Bench2Drive 批次驗證腳本（本機多 GPU）
#
# 使用方式:
#   bash batch_eval_simlingo.sh [選項]
#
# 選項:
#   --gpus "0 1 2 3 4 5 6 7"   指定使用的 GPU 編號（預設: 全部 8 張）
#   --output /path/to/dir       輸出目錄（預設: eval_results/batch_YYYYMMDD_HHMMSS）
#   --model /path/to/model.pt   模型路徑（預設: outputs/simlingo/checkpoints/pytorch_model.pt）
#   --routes /path/to/split/    路線目錄（預設: leaderboard/data/bench2drive_split）
#   --save-video                啟用影格儲存以生成影片
#   --generate-video            評估完成後自動生成影片（需搭配 --save-video）
#   --base-port 20000           CARLA 起始 port（預設: 20000）
#   --port-step 200             各 GPU 間的 port 間隔（預設: 200）
#
# 範例:
#   # 使用全部 8 GPU 跑完整 220 條路線
#   bash batch_eval_simlingo.sh
#
#   # 只用 4 張 GPU，並儲存影片
#   bash batch_eval_simlingo.sh --gpus "0 1 2 3" --save-video --generate-video
#
#   # 指定輸出目錄
#   bash batch_eval_simlingo.sh --output /mnt/SSD7/dow904/simlingo/eval_results/my_run
# ============================================================

set -euo pipefail

# ============================================================
# 預設設定（可透過命令列參數覆蓋）
# ============================================================
SIMLINGO_ROOT="/mnt/SSD7/dow904/simlingo"
CARLA_ROOT="/mnt/SSD7/dow904/carla"
MODEL_PATH="${SIMLINGO_ROOT}/outputs/simlingo/checkpoints/pytorch_model.pt"
ROUTES_DIR="${SIMLINGO_ROOT}/leaderboard/data/bench2drive_split"
GPU_LIST_STR="4 5 6 7"
OUTPUT_DIR=""
SAVE_VIDEO=0
GENERATE_VIDEO=0
BASE_PORT=20000
PORT_STEP=200

# ============================================================
# 解析命令列參數
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPU_LIST_STR="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2";   shift 2 ;;
        --model)      MODEL_PATH="$2";   shift 2 ;;
        --routes)     ROUTES_DIR="$2";   shift 2 ;;
        --save-video)     SAVE_VIDEO=1;  shift   ;;
        --generate-video) GENERATE_VIDEO=1; shift ;;
        --base-port)  BASE_PORT="$2";    shift 2 ;;
        --port-step)  PORT_STEP="$2";    shift 2 ;;
        -h|--help)
            head -n 40 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "未知參數: $1"; exit 1 ;;
    esac
done

# 預設輸出目錄（含時間戳）
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SIMLINGO_ROOT}/eval_results/batch_$(date +%Y%m%d_%H%M%S)"
fi

# 轉換 GPU list 為陣列
read -ra GPU_LIST <<< "${GPU_LIST_STR}"
NUM_GPUS=${#GPU_LIST[@]}

# ============================================================
# 檢查必要檔案
# ============================================================
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: 找不到模型檔案: ${MODEL_PATH}"
    exit 1
fi

if [[ ! -d "${ROUTES_DIR}" ]]; then
    echo "ERROR: 找不到路線目錄: ${ROUTES_DIR}"
    exit 1
fi

if [[ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
    echo "ERROR: 找不到 CarlaUE4.sh: ${CARLA_ROOT}/CarlaUE4.sh"
    exit 1
fi

# ============================================================
# 建立輸出目錄
# ============================================================
mkdir -p "${OUTPUT_DIR}/res"
mkdir -p "${OUTPUT_DIR}/viz"
mkdir -p "${OUTPUT_DIR}/logs"

# ============================================================
# 顯示設定摘要
# ============================================================
echo "========================================================"
echo "  SimLingo Bench2Drive 批次驗證"
echo "========================================================"
echo "  SIMLINGO_ROOT : ${SIMLINGO_ROOT}"
echo "  CARLA_ROOT    : ${CARLA_ROOT}"
echo "  MODEL_PATH    : ${MODEL_PATH}"
echo "  ROUTES_DIR    : ${ROUTES_DIR}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  GPUs          : ${GPU_LIST[*]}"
echo "  NUM_GPUS      : ${NUM_GPUS}"
echo "  BASE_PORT     : ${BASE_PORT}"
echo "  PORT_STEP     : ${PORT_STEP}"
echo "  SAVE_VIDEO    : ${SAVE_VIDEO}"
echo "  GENERATE_VIDEO: ${GENERATE_VIDEO}"
echo "========================================================"

# ============================================================
# 取得排序後的路線 XML 清單
# ============================================================
mapfile -t ROUTES < <(ls "${ROUTES_DIR}"/bench2drive_*.xml 2>/dev/null | sort)
TOTAL_ROUTES=${#ROUTES[@]}

if [[ ${TOTAL_ROUTES} -eq 0 ]]; then
    echo "ERROR: 在 ${ROUTES_DIR} 找不到路線 XML 檔案"
    exit 1
fi

echo "  總路線數     : ${TOTAL_ROUTES}"
echo "  每 GPU 路線數: $((TOTAL_ROUTES / NUM_GPUS)) ~ $(((TOTAL_ROUTES + NUM_GPUS - 1) / NUM_GPUS))"
echo "========================================================"

# ============================================================
# 分配路線給各 GPU（round-robin 分配）
# ============================================================
for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    GPU_ID=${GPU_LIST[$gpu_idx]}
    WORKER_ROUTES_FILE="${OUTPUT_DIR}/gpu${GPU_ID}_routes.txt"
    > "${WORKER_ROUTES_FILE}"

    for ((i=gpu_idx; i<TOTAL_ROUTES; i+=NUM_GPUS)); do
        echo "${ROUTES[$i]}" >> "${WORKER_ROUTES_FILE}"
    done

    ROUTE_COUNT=$(wc -l < "${WORKER_ROUTES_FILE}")
    echo "  GPU ${GPU_ID}: 分配 ${ROUTE_COUNT} 條路線 (port: $((BASE_PORT + gpu_idx * PORT_STEP)))"
done
echo "========================================================"

# ============================================================
# 並行啟動各 GPU Worker
# ============================================================
PIDS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    GPU_ID=${GPU_LIST[$gpu_idx]}
    PORT=$((BASE_PORT + gpu_idx * PORT_STEP))
    TM_PORT=$((PORT + 100))
    WORKER_ROUTES_FILE="${OUTPUT_DIR}/gpu${GPU_ID}_routes.txt"
    LOG_FILE="${OUTPUT_DIR}/logs/gpu${GPU_ID}.log"

    echo "啟動 GPU ${GPU_ID} Worker (CARLA port: ${PORT}, TM port: ${TM_PORT})..."

    bash "${SCRIPT_DIR}/run_eval_worker.sh" \
        "${GPU_ID}" \
        "${PORT}" \
        "${TM_PORT}" \
        "${WORKER_ROUTES_FILE}" \
        "${OUTPUT_DIR}" \
        "${MODEL_PATH}" \
        "${CARLA_ROOT}" \
        "${SIMLINGO_ROOT}" \
        "${SAVE_VIDEO}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)

    # 錯開啟動時間，避免 CARLA 同時競爭資源
    sleep 5
done

# ============================================================
# 等待所有 Worker 完成
# ============================================================
echo ""
echo "所有 Worker 已啟動，等待完成..."
echo "可以透過以下指令監控進度："
echo "  tail -f ${OUTPUT_DIR}/logs/gpu0.log"
echo "  ls ${OUTPUT_DIR}/res/ | wc -l  (已完成路線數)"
echo ""

FAIL_WORKERS=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU_ID=${GPU_LIST[$i]}
    wait "${PID}"
    EXIT=${?}
    if [[ ${EXIT} -eq 0 ]]; then
        echo "GPU ${GPU_ID} Worker 完成 ✓"
    else
        echo "GPU ${GPU_ID} Worker 異常退出 (exit code: ${EXIT}) ✗"
        FAIL_WORKERS=$((FAIL_WORKERS + 1))
    fi
done

echo ""
echo "========================================================"
echo "  所有評估完成！失敗 Worker: ${FAIL_WORKERS}"
echo "========================================================"

# ============================================================
# 生成影片（若啟用）
# ============================================================
if [[ "${SAVE_VIDEO}" -eq 1 && "${GENERATE_VIDEO}" -eq 1 ]]; then
    echo ""
    echo "生成影片中..."
    VIDEO_DIR="${OUTPUT_DIR}/videos"
    mkdir -p "${VIDEO_DIR}"

    python3 "${SCRIPT_DIR}/generate_eval_videos.py" \
        --input_dir "${OUTPUT_DIR}/viz" \
        --output_dir "${VIDEO_DIR}" \
        --fps 10

    echo "影片已儲存至: ${VIDEO_DIR}"
fi

# ============================================================
# 彙整結果
# ============================================================
echo ""
echo "彙整評估結果..."

RES_COUNT=$(ls "${OUTPUT_DIR}/res/"*_res.json 2>/dev/null | wc -l)
echo "已完成路線數: ${RES_COUNT} / ${TOTAL_ROUTES}"

if [[ ${RES_COUNT} -gt 0 ]]; then
    python3 "${SIMLINGO_ROOT}/tools/result_parser_new.py" \
        --xml "${SIMLINGO_ROOT}/leaderboard/data/bench2drive220.xml" \
        --results "${OUTPUT_DIR}/res" \
        2>&1 | tail -30

    echo ""
    echo "詳細結果 CSV: ${OUTPUT_DIR}/res/results.csv"
fi

echo ""
echo "========================================================"
echo "  輸出目錄: ${OUTPUT_DIR}"
echo "  結果 JSON: ${OUTPUT_DIR}/res/*_res.json"
echo "  可視化:    ${OUTPUT_DIR}/viz/"
if [[ "${SAVE_VIDEO}" -eq 1 && "${GENERATE_VIDEO}" -eq 1 ]]; then
    echo "  影片:      ${OUTPUT_DIR}/videos/"
fi
echo "========================================================"
