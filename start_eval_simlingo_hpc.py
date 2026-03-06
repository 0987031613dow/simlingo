"""
SimLingo HPC SLURM 批次評估腳本

相較於原始 start_eval_simlingo.py 的改進：
1. 在每個 SLURM job 中自動啟動/停止 CARLA（off-screen 模式）
2. 修正路徑設定（請更新 configs 區塊中的路徑）
3. 加入 port 衝突避免機制
4. 支援 SAVE_VIDEO=1 開啟影格儲存

使用方式:
    # 在 HPC 節點上（需有 SLURM 環境）
    python start_eval_simlingo_hpc.py

設定區域在 configs 清單中，請依實際路徑修改。
"""

# %%
import os
import subprocess
import time
import ujson
import shutil
from tqdm.autonotebook import tqdm

# %%
def get_num_jobs(job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        ).decode('utf-8').replace('\n', ''))
    try:
        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except Exception:
        max_num_parallel_jobs = 1

    return num_running_jobs, max_num_parallel_jobs

# %%
def bash_file_bench2drive(job, carla_port, tm_port, partition_name):
    """產生 SLURM job 腳本（含 CARLA 自動啟動/停止）"""
    cfg = job["cfg"]
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]
    save_video = cfg.get("save_video", 0)

    with open(job_file, 'w', encoding='utf-8') as rsh:
        rsh.write(f'''#!/bin/bash
#SBATCH --job-name={cfg["agent"]}_{seed}_{cfg["benchmark"]}_{route_id}
#SBATCH --partition={partition_name}
#SBATCH -o {log_file}
#SBATCH -e {err_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:1

echo "JOB ID $SLURM_JOB_ID"
echo "HOSTNAME: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
. ~/software/anaconda3/etc/profile.d/conda.sh
conda activate {cfg["conda_env"]}

cd {cfg["repo_root"]}

# ---- 環境變數 ----
export CARLA_ROOT={cfg["carla_root"]}
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/leaderboard
export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/scenario_runner
export SCENARIO_RUNNER_ROOT={cfg["repo_root"]}/scenario_runner
export LEADERBOARD_ROOT={cfg["repo_root"]}/leaderboard
export SAVE_PATH={viz_path}/
export SAVE_VIDEO={save_video}

# ---- 啟動 Xvfb 虛擬顯示器（無 GUI 環境必要）----
DISPLAY_NUM=$((10 + SLURM_LOCALID))
export DISPLAY=":${{DISPLAY_NUM}}"
rm -f "/tmp/.X${{DISPLAY_NUM}}-lock" 2>/dev/null
Xvfb "${{DISPLAY}}" -screen 0 1280x720x24 -ac +extension GLX +render -noreset \\
    &> {log_file}.xvfb.log &
XVFB_PID=$!
sleep 2
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb 啟動失敗！請安裝: sudo apt install xvfb"
    exit 1
fi
echo "Xvfb 已就緒 (DISPLAY=${{DISPLAY}}, PID=$XVFB_PID)"

# ---- 啟動 CARLA (off-screen) ----
echo "啟動 CARLA (port={carla_port}, TM={tm_port})..."
DISPLAY="${{DISPLAY}}" \\
{cfg["carla_root"]}/CarlaUE4.sh \\
    -RenderOffScreen \\
    -nosound \\
    -carla-rpc-port={carla_port} \\
    -carla-streaming-port=0 \\
    &> {log_file}.carla.log &
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"

# 等待 CARLA 初始化
echo "等待 CARLA 初始化 (30s)..."
sleep 30

# 確認 CARLA 存活
if ! kill -0 $CARLA_PID 2>/dev/null; then
    echo "ERROR: CARLA 啟動失敗！"
    tail -30 {log_file}.carla.log
    exit 1
fi
echo "CARLA 已就緒"

# ---- 執行評估 ----
echo "開始評估路線 {route_id}..."
python -u {cfg["repo_root"]}/leaderboard/leaderboard/leaderboard_evaluator.py \\
    --host=localhost \\
    --port={carla_port} \\
    --traffic-manager-port={tm_port} \\
    --routes={route} \\
    --repetitions=1 \\
    --track=SENSORS \\
    --checkpoint={result_file} \\
    --resume=True \\
    --agent={cfg["agent_file"]} \\
    --agent-config={cfg["checkpoint"]}+ \\
    --traffic-manager-seed={seed} \\
    --timeout=600
EVAL_EXIT=$?

# ---- 停止 CARLA & Xvfb ----
echo "停止 CARLA (PID: $CARLA_PID)..."
kill $CARLA_PID 2>/dev/null
sleep 3
kill -9 $CARLA_PID 2>/dev/null
pkill -f "carla-rpc-port={carla_port}" 2>/dev/null
kill $XVFB_PID 2>/dev/null
pkill -f "Xvfb ${{DISPLAY}}" 2>/dev/null

echo "評估完成，exit code: $EVAL_EXIT"
exit $EVAL_EXIT
''')


# %%
def get_running_jobs():
    running_jobs = subprocess.check_output(f'squeue --me', shell=True).decode('utf-8').splitlines()
    running_jobs = set(x.strip().split(" ")[0] for x in running_jobs[1:])
    return running_jobs

# %%
def filter_completed(jobs):
    filtered_jobs = []
    running_jobs = get_running_jobs()

    for job in jobs:
        # 如果 job 正在執行中，保留
        if "job_id" in job:
            if job["job_id"] in running_jobs:
                filtered_jobs.append(job)
                continue

        result_file = job["result_file"]
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    evaluation_data = ujson.load(f)
            except Exception:
                if job["tries"] > 0:
                    filtered_jobs.append(job)
                    continue

            progress = evaluation_data['_checkpoint']['progress']

            need_to_resubmit = False
            if len(progress) < 2 or progress[0] < progress[1]:
                need_to_resubmit = True
            else:
                for record in evaluation_data['_checkpoint']['records']:
                    if record['status'] in [
                        'Failed - Agent couldn\'t be set up',
                        'Failed',
                        'Failed - Simulation crashed',
                        'Failed - Agent crashed'
                    ]:
                        need_to_resubmit = True

            if need_to_resubmit and job["tries"] > 0:
                filtered_jobs.append(job)
        elif job["tries"] > 0:
            filtered_jobs.append(job)

    return filtered_jobs

# %%
def kill_dead_jobs(jobs):
    running_jobs = get_running_jobs()

    for job in jobs:
        if "job_id" in job:
            job_id = job["job_id"]
        elif os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                job_id = f.readline().strip().replace("JOB ID ", "")
        else:
            continue

        if job_id not in running_jobs:
            continue

        log = job["log_file"]
        if not os.path.exists(log):
            continue

        with open(log) as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue

        if (any("Watchdog exception" in line for line in lines) or
                "Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.\n" in lines or
                "[91mStopping the route, the agent has crashed:\n" in lines):
            subprocess.Popen(f"scancel {job_id}", shell=True)


# ============================================================
# !! 請修改以下 configs 為你的實際路徑 !!
# ============================================================
configs = [
    {
        # === 必填：請依實際情況修改 ===
        "agent": "simlingo",

        # SimLingo 模型 checkpoint 路徑（HPC 上的路徑）
        "checkpoint": "/PATH/TO/REPO/outputs/simlingo/checkpoints/pytorch_model.pt",

        # 評估基準
        "benchmark": "bench2drive",

        # bench2drive_split 目錄（每條路線一個 XML）
        "route_path": "/PATH/TO/REPO/leaderboard/data/bench2drive_split",

        # 評估種子（論文使用 seeds=[1,2,3] 各跑一次，通常先用 1）
        "seeds": [1],

        # 失敗重試次數
        "tries": 2,

        # 輸出根目錄
        "out_root": "/PATH/TO/REPO/eval_results/HPC_Bench2Drive",

        # CARLA 安裝路徑（HPC 上）
        "carla_root": "/PATH/TO/carla",

        # simlingo repo 根目錄（HPC 上）
        "repo_root": "/PATH/TO/REPO",

        # agent 腳本路徑
        "agent_file": "/PATH/TO/REPO/team_code/agent_simlingo.py",

        # HPC 用戶名（用於 squeue 過濾）
        "username": "YOUR_HPC_USERNAME",

        # Conda 環境名稱
        "conda_env": "simlingo",

        # 是否儲存影格（0=否, 1=是）
        "save_video": 0,
    }
]
# ============================================================

# %%
job_queue = []
for cfg_idx, cfg in enumerate(configs):
    route_path = cfg["route_path"]
    routes = [x for x in os.listdir(route_path) if x[-4:] == ".xml"]

    if cfg["benchmark"] == "bench2drive":
        fill_zeros = 3
    else:
        fill_zeros = 2

    for seed in cfg["seeds"]:
        seed = str(seed)
        base_dir = os.path.join(cfg["out_root"], cfg["agent"], cfg["benchmark"], seed)
        os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)

        for route in sorted(routes):
            route_id = route.split("_")[-1][:-4].zfill(fill_zeros)
            route = os.path.join(route_path, route)

            viz_path = os.path.join(base_dir, "viz", route_id)
            os.makedirs(viz_path, exist_ok=True)

            result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
            log_file    = os.path.join(base_dir, "out", f"{route_id}_out.log")
            err_file    = os.path.join(base_dir, "err", f"{route_id}_err.log")
            job_file    = os.path.join(base_dir, "run", f'eval_{route_id}.sh')

            job = {
                "cfg": cfg,
                "route": route,
                "route_id": route_id,
                "seed": seed,
                "viz_path": viz_path,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "job_file": job_file,
                "tries": cfg["tries"]
            }
            job_queue.append(job)

# %%
# Port 池（每個 job 取一組）
# HPC 節點每個 job 獨佔一張 GPU，所以 port 固定即可
# 若多個 job 在同一節點，需要不同 port
carla_ports    = list(range(10000, 20000, 50))
tm_ports       = list(range(30000, 40000, 50))

# %%
jobs = len(job_queue)
progress = tqdm(total=jobs)

# TODO: 修改為你的 SLURM partition 名稱
partition_name = "YOUR_PARTITION_NAME"

while job_queue:
    kill_dead_jobs(job_queue)
    job_queue = filter_completed(job_queue)
    progress.update(jobs - len(job_queue) - progress.n)

    running_jobs = get_running_jobs()

    used_ports = set()
    for job in job_queue:
        if "job_id" in job and job["job_id"] in running_jobs:
            used_ports.update(job.get("ports", set()))

    with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
        max_num_parallel_jobs = int(f.read())

    if len(running_jobs) >= max_num_parallel_jobs:
        time.sleep(5)
        continue

    for job in job_queue:
        if job["tries"] <= 0:
            continue

        if "job_id" in job and job["job_id"] in running_jobs:
            continue

        if os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                first_line = f.readline().strip()
                job_id = first_line.replace("JOB ID ", "")
                if job_id in running_jobs:
                    print(f"{job['log_file']} 已在執行中。")
                    continue

        # 分配 port（避免衝突）
        carla_port = next(p for p in carla_ports if p not in used_ports)
        tm_port    = next(p for p in tm_ports if p not in used_ports)

        if job["cfg"]["benchmark"].lower() == "bench2drive":
            bash_file_bench2drive(job, carla_port, tm_port, partition_name)
            job["ports"] = {carla_port, tm_port}
        else:
            raise NotImplementedError(f"Benchmark {job['cfg']['benchmark']} 尚未實作")

        # 清空 viz 目錄後重新執行
        shutil.rmtree(job["viz_path"])
        os.mkdir(job["viz_path"])

        job_id = subprocess.check_output(
            f'sbatch {job["job_file"]}', shell=True
        ).decode('utf-8').strip().rsplit(' ', maxsplit=1)[-1]

        job["job_id"] = job_id
        job["tries"] -= 1

        print(f'提交 {job["job_file"]} (SLURM job: {job_id})')
        print(f'待處理 jobs: {len(job_queue)}')
        break

    time.sleep(10)
