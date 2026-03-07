#!/bin/bash
# ============================================================
# SimLingo 單 GPU 評估 Worker（無頭模式 / Headless）
# 由 batch_eval_simlingo.sh 呼叫，請勿直接執行
#
# 使用 Xvfb 虛擬顯示器讓 CARLA 能在無 GUI 環境下執行
# ============================================================

GPU_ID=$1          # GPU 編號 (e.g. 0)
PORT=$2            # CARLA RPC port
TM_PORT=$3         # Traffic Manager port
ROUTES_FILE=$4     # 含 route XML 路徑的清單檔
OUTPUT_DIR=$5      # 結果輸出根目錄
MODEL_PATH=$6      # pytorch_model.pt 路徑
CARLA_ROOT=$7      # CARLA 安裝目錄
SIMLINGO_ROOT=$8   # simlingo repo 根目錄
SAVE_VIDEO=${9:-0} # 是否儲存影格 (0/1)

LOG_PREFIX="[GPU ${GPU_ID}]"

# 每個 GPU 使用不同的虛擬 display number（:10, :11, ..., :17）
DISPLAY_NUM=$((10 + GPU_ID))
export DISPLAY=":${DISPLAY_NUM}"

echo "${LOG_PREFIX} ===== Worker 啟動 ====="
echo "${LOG_PREFIX} PORT=${PORT}, TM_PORT=${TM_PORT}"
echo "${LOG_PREFIX} DISPLAY=${DISPLAY}"
echo "${LOG_PREFIX} ROUTES_FILE=${ROUTES_FILE}"
echo "${LOG_PREFIX} OUTPUT_DIR=${OUTPUT_DIR}"
echo "${LOG_PREFIX} SAVE_VIDEO=${SAVE_VIDEO}"

# ---- Python 環境設定 ----
SIMLINGO_ENV="/mnt/SSD7/dow904/.local/share/mamba/envs/simlingo"
export PATH="${SIMLINGO_ENV}/bin:${PATH}"
export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH="${PYTHONPATH:-}:${SIMLINGO_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${PYTHONPATH}:${SIMLINGO_ROOT}/leaderboard"
export PYTHONPATH="${PYTHONPATH}:${SIMLINGO_ROOT}/scenario_runner"
export SCENARIO_RUNNER_ROOT="${SIMLINGO_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${SIMLINGO_ROOT}/leaderboard"

mkdir -p "${OUTPUT_DIR}/res"
mkdir -p "${OUTPUT_DIR}/viz"
mkdir -p "${OUTPUT_DIR}/logs"

# ============================================================
# 函式：啟動 Xvfb 虛擬顯示器
# ============================================================
start_xvfb() {
    # 若該 display 已存在，先清除
    pkill -f "Xvfb ${DISPLAY}" 2>/dev/null
    sleep 1
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" 2>/dev/null

    echo "${LOG_PREFIX} 啟動 Xvfb (${DISPLAY})..."
    Xvfb "${DISPLAY}" -screen 0 1280x720x24 -ac +extension GLX +render -noreset \
        &> "${OUTPUT_DIR}/logs/xvfb_gpu${GPU_ID}.log" &
    XVFB_PID=$!

    sleep 2

    if ! kill -0 "${XVFB_PID}" 2>/dev/null; then
        echo "${LOG_PREFIX} ERROR: Xvfb 啟動失敗！"
        echo "${LOG_PREFIX} 請確認已安裝: sudo apt install xvfb"
        return 1
    fi
    echo "${LOG_PREFIX} Xvfb 已就緒 (PID: ${XVFB_PID})"
    return 0
}

# ============================================================
# 函式：啟動 CARLA
# ============================================================
start_carla() {
    local log_suffix="${1:-}"
    echo "${LOG_PREFIX} 啟動 CARLA (off-screen, port=${PORT})..."

    VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json" \
    DISPLAY="${DISPLAY}" \
        "${CARLA_ROOT}/CarlaUE4.sh" \
        -RenderOffScreen \
        -nosound \
        -graphicsadapter="${GPU_ID}" \
        -carla-rpc-port="${PORT}" \
        -carla-streaming-port=0 \
        &> "${OUTPUT_DIR}/logs/carla_gpu${GPU_ID}${log_suffix}.log" &

    CARLA_PID=$!
    echo "${LOG_PREFIX} CARLA PID: ${CARLA_PID}，等待初始化 (30s)..."
    sleep 30

    if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
        echo "${LOG_PREFIX} ERROR: CARLA 啟動失敗！"
        echo "${LOG_PREFIX} 請查看: ${OUTPUT_DIR}/logs/carla_gpu${GPU_ID}${log_suffix}.log"
        tail -20 "${OUTPUT_DIR}/logs/carla_gpu${GPU_ID}${log_suffix}.log" 2>/dev/null
        return 1
    fi
    echo "${LOG_PREFIX} CARLA 已就緒"
    return 0
}

# ============================================================
# 函式：停止 CARLA
# ============================================================
stop_carla() {
    echo "${LOG_PREFIX} 停止 CARLA (PID: ${CARLA_PID})..."
    kill "${CARLA_PID}" 2>/dev/null
    sleep 3
    kill -9 "${CARLA_PID}" 2>/dev/null
    wait "${CARLA_PID}" 2>/dev/null
    pkill -f "carla-rpc-port=${PORT}" 2>/dev/null
    pkill -f "CarlaUE4.*${PORT}" 2>/dev/null
}

# ============================================================
# 啟動 Xvfb + CARLA
# ============================================================
start_xvfb || exit 1
start_carla || exit 1

# ============================================================
# 逐條路線執行評估
# ============================================================
ROUTE_COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r route_xml; do
    [[ -z "$route_xml" ]] && continue

    route_base=$(basename "${route_xml}" .xml)
    result_file="${OUTPUT_DIR}/res/${route_base}_res.json"
    viz_path="${OUTPUT_DIR}/viz/"

    ROUTE_COUNT=$((ROUTE_COUNT + 1))
    echo ""
    echo "${LOG_PREFIX} --- 路線 ${ROUTE_COUNT}: ${route_base} ---"

    # ---- 檢查是否已完成（支援 resume）----
    if [[ -f "${result_file}" ]]; then
        IS_DONE=$(python3 -c "
import json
try:
    with open('${result_file}') as f:
        d = json.load(f)
    p = d.get('_checkpoint', {}).get('progress', [0, 1])
    if len(p) >= 2 and p[0] >= p[1]:
        records = d.get('_checkpoint', {}).get('records', [])
        failed = any(r.get('status', '') in [
            'Failed', 'Failed - Simulation crashed',
            'Failed - Agent crashed', \"Failed - Agent couldn't be set up\"]
            for r in records)
        print('incomplete' if failed else 'done')
    else:
        print('incomplete')
except Exception:
    print('incomplete')
" 2>/dev/null)

        if [[ "${IS_DONE}" == "done" ]]; then
            echo "${LOG_PREFIX} 跳過 ${route_base}（已完成）"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            continue
        else
            echo "${LOG_PREFIX} ${route_base} 未完成，重新執行..."
        fi
    fi

    # ---- 執行評估 ----
    echo "${LOG_PREFIX} 執行評估: ${route_base}"

    DISPLAY="${DISPLAY}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    SAVE_PATH="${viz_path}" \
    SAVE_VIDEO="${SAVE_VIDEO}" \
    python -u "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
        --host=localhost \
        --port="${PORT}" \
        --traffic-manager-port="${TM_PORT}" \
        --routes="${route_xml}" \
        --repetitions=1 \
        --track=SENSORS \
        --checkpoint="${result_file}" \
        --resume=True \
        --agent="${SIMLINGO_ROOT}/team_code/agent_simlingo.py" \
        --agent-config="${MODEL_PATH}+" \
        --timeout=600 \
        2>&1

    EXIT_CODE=$?

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "${LOG_PREFIX} 完成: ${route_base}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "${LOG_PREFIX} 警告: ${route_base} 結束碼 ${EXIT_CODE}"
        FAIL_COUNT=$((FAIL_COUNT + 1))

        # 若 CARLA 崩潰，重啟 Xvfb + CARLA
        if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            echo "${LOG_PREFIX} CARLA 已停止，重新啟動..."
            # 重啟 Xvfb（可能也崩潰了）
            if ! kill -0 "${XVFB_PID}" 2>/dev/null; then
                start_xvfb || { echo "${LOG_PREFIX} Xvfb 重啟失敗，退出"; break; }
            fi
            start_carla "_restart_$(date +%H%M%S)" || { echo "${LOG_PREFIX} CARLA 重啟失敗，退出"; break; }
        fi
    fi

    # 路線間短暫間隔
    sleep 5

done < "${ROUTES_FILE}"

# ============================================================
# 清理
# ============================================================
echo ""
echo "${LOG_PREFIX} 所有路線處理完畢"
echo "${LOG_PREFIX} 成功: ${SUCCESS_COUNT}, 失敗: ${FAIL_COUNT}"

stop_carla

# 停止 Xvfb
echo "${LOG_PREFIX} 停止 Xvfb (PID: ${XVFB_PID})..."
kill "${XVFB_PID}" 2>/dev/null
pkill -f "Xvfb ${DISPLAY}" 2>/dev/null

echo "${LOG_PREFIX} Worker 完成！"
