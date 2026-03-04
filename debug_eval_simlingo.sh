#!/bin/bash
# ============================================================
# SimLingo Debug 驗證腳本
# 只跑少數幾條路線，用來確認整個 pipeline 是否正常運作
#
# 使用方式:
#   bash debug_eval_simlingo.sh [GPU_ID] [路線數] [SAVE_VIDEO]
#
# 範例:
#   bash debug_eval_simlingo.sh 4 2 0   # 用 GPU 4 跑 2 條路線，不存影片
#   bash debug_eval_simlingo.sh 4 4 1   # 用 GPU 4 跑 4 條路線，存影片
# ============================================================

set -eu

GPU_ID=${1:-4}
NUM_ROUTES=${2:-2}
SAVE_VIDEO=${3:-0}

SIMLINGO_ROOT="/mnt/SSD7/dow904/simlingo"
CARLA_ROOT="/mnt/SSD7/dow904/carla"
SIMLINGO_ENV="/mnt/SSD7/dow904/.local/share/mamba/envs/simlingo"
MODEL_PATH="${SIMLINGO_ROOT}/outputs/simlingo/checkpoints/pytorch_model.pt"
ROUTES_DIR="${SIMLINGO_ROOT}/leaderboard/data/bench2drive_split"
OUTPUT_DIR="${SIMLINGO_ROOT}/eval_results/debug_$(date +%Y%m%d_%H%M%S)"
PORT=$((20000 + GPU_ID * 200))

# 使用 simlingo micromamba 環境
export PATH="${SIMLINGO_ENV}/bin:${PATH}"
TM_PORT=$((PORT + 100))
DISPLAY_NUM=$((10 + GPU_ID))

echo "========================================================"
echo "  SimLingo Debug 驗證"
echo "========================================================"
echo "  GPU           : ${GPU_ID}"
echo "  路線數        : ${NUM_ROUTES}"
echo "  SAVE_VIDEO    : ${SAVE_VIDEO}"
echo "  PORT          : ${PORT} / TM: ${TM_PORT}"
echo "  DISPLAY       : :${DISPLAY_NUM}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "========================================================"

# ---- 建立輸出目錄 ----
mkdir -p "${OUTPUT_DIR}/res" "${OUTPUT_DIR}/viz" "${OUTPUT_DIR}/logs"

# ---- 建立 debug 路線清單（取前 N 條）----
ROUTES_FILE="${OUTPUT_DIR}/debug_routes.txt"
mapfile -t ALL_ROUTES < <(ls "${ROUTES_DIR}"/bench2drive_*.xml | sort)
printf '%s\n' "${ALL_ROUTES[@]:0:${NUM_ROUTES}}" > "${ROUTES_FILE}"

echo "測試路線："
cat "${ROUTES_FILE}"
echo ""

# ---- 設定 Python 環境 ----
export PYTHONPATH="${PYTHONPATH:-}:${SIMLINGO_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${PYTHONPATH}:${SIMLINGO_ROOT}/leaderboard"
export PYTHONPATH="${PYTHONPATH}:${SIMLINGO_ROOT}/scenario_runner"
export SCENARIO_RUNNER_ROOT="${SIMLINGO_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${SIMLINGO_ROOT}/leaderboard"
export DISPLAY=":${DISPLAY_NUM}"

# ---- 啟動 Xvfb ----
echo "啟動 Xvfb (DISPLAY=${DISPLAY})..."
pkill -f "Xvfb ${DISPLAY}" 2>/dev/null || true
sleep 1
rm -f "/tmp/.X${DISPLAY_NUM}-lock" 2>/dev/null || true

Xvfb "${DISPLAY}" -screen 0 1280x720x24 -ac +extension GLX +render -noreset \
    &> "${OUTPUT_DIR}/logs/xvfb.log" &
XVFB_PID=$!
sleep 2

if ! kill -0 "${XVFB_PID}" 2>/dev/null; then
    echo "ERROR: Xvfb 啟動失敗！"
    echo "請確認已安裝: sudo apt install xvfb"
    exit 1
fi
echo "Xvfb 已就緒 (PID: ${XVFB_PID})"

# ---- 啟動 CARLA ----
echo ""
echo "啟動 CARLA (off-screen, port=${PORT})..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" DISPLAY="${DISPLAY}" \
    "${CARLA_ROOT}/CarlaUE4.sh" \
    -RenderOffScreen \
    -nosound \
    -graphicsadapter="${GPU_ID}" \
    -carla-rpc-port="${PORT}" \
    -carla-streaming-port=0 \
    &> "${OUTPUT_DIR}/logs/carla.log" &
CARLA_PID=$!

echo "CARLA PID: ${CARLA_PID}，等待初始化 (30s)..."
sleep 30

if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
    echo "ERROR: CARLA 啟動失敗！請查看："
    echo "  ${OUTPUT_DIR}/logs/carla.log"
    echo ""
    echo "最後 30 行 log："
    tail -n 30 "${OUTPUT_DIR}/logs/carla.log" 2>/dev/null
    kill "${XVFB_PID}" 2>/dev/null
    exit 1
fi
echo "CARLA 已就緒"

# ---- 逐條路線執行評估 ----
echo ""
SUCCESS=0
FAIL=0

while IFS= read -r route_xml; do
    [[ -z "$route_xml" ]] && continue

    route_base=$(basename "${route_xml}" .xml)
    result_file="${OUTPUT_DIR}/res/${route_base}_res.json"

    echo "------------------------------------------------------"
    echo "評估: ${route_base}"
    echo "------------------------------------------------------"

    DISPLAY="${DISPLAY}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    SAVE_PATH="${OUTPUT_DIR}/viz/" \
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
        echo "完成: ${route_base} ✓"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "失敗: ${route_base} ✗ (exit ${EXIT_CODE})"
        FAIL=$((FAIL + 1))
    fi

    sleep 3
done < "${ROUTES_FILE}"

# ---- 清理 ----
echo ""
echo "========================================================"
echo "  Debug 完成！成功: ${SUCCESS}, 失敗: ${FAIL}"
echo "========================================================"

echo "停止 CARLA..."
kill "${CARLA_PID}" 2>/dev/null || true
sleep 2
kill -9 "${CARLA_PID}" 2>/dev/null || true
pkill -f "carla-rpc-port=${PORT}" 2>/dev/null || true

echo "停止 Xvfb..."
kill "${XVFB_PID}" 2>/dev/null || true
pkill -f "Xvfb ${DISPLAY}" 2>/dev/null || true

# ---- 顯示結果 ----
echo ""
echo "結果 JSON: ${OUTPUT_DIR}/res/"
ls "${OUTPUT_DIR}/res/"*.json 2>/dev/null && \
    python3 -c "
import json, glob
for f in sorted(glob.glob('${OUTPUT_DIR}/res/*.json')):
    with open(f) as fp:
        d = json.load(fp)
    for r in d.get('_checkpoint', {}).get('records', []):
        print(f'  {r[\"route_id\"]}: {r[\"status\"]}')
        scores = r.get('scores', {})
        print(f'    route_completion={scores.get(\"score_route\",\"N/A\"):.1f}%  driving_score={scores.get(\"score_composed\",\"N/A\"):.2f}')
" 2>/dev/null || echo "(結果解析失敗，請手動查看 JSON)"

echo ""
echo "完整 log: ${OUTPUT_DIR}/logs/"
echo "  CARLA log: ${OUTPUT_DIR}/logs/carla.log"
echo "  Xvfb log:  ${OUTPUT_DIR}/logs/xvfb.log"
