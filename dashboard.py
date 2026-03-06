import streamlit as st
import pandas as pd
import subprocess
import os
import json
import glob
import time
import psutil
import io
import csv

# Set page config
st.set_page_config(layout="wide", page_title="Bench2Drive Local Dashboard")

st.title("Bench2Drive Local Dashboard 🚗 (No Slurm)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    auto_refresh = st.checkbox("Auto Refresh Job Monitor (5s)", value=False)
    
    st.markdown("---")
    st.info("Navigate using the tabs above.")

def get_gpu_info():
    try:
        # Query nvidia-smi for specific metrics in CSV format
        cmd = "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        gpus = []
        reader = csv.reader(io.StringIO(output))
        for row in reader:
            if len(row) >= 6:
                gpus.append({
                    "ID": int(row[0]),
                    "Name": row[1].strip(),
                    "Temp (°C)": int(row[2]),
                    "Util (%)": int(row[3]),
                    "Mem Used (MiB)": int(row[4]),
                    "Mem Total (MiB)": int(row[5]),
                    "Mem Util (%)": round(int(row[4]) / int(row[5]) * 100, 1)
                })
        return gpus
    except Exception as e:
        return []

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Process & GPU Monitor", "📈 Results Viewer", "🚀 Job Launcher"])

# --- TAB 1: PROCESS & GPU MONITOR ---
with tab1:
    st.header("System Monitor")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Refresh System", type="primary"):
            pass 
    
    if auto_refresh:
        time.sleep(2) # Faster refresh for GPU
        st.rerun()

    # GPU Section
    st.subheader("GPU Status 🖥️")
    gpus = get_gpu_info()
    if gpus:
        df_gpu = pd.DataFrame(gpus)
        st.dataframe(df_gpu, use_container_width=True)
        
        # Simple metric display for quick glance
        cols = st.columns(max(1, len(gpus)))
        for i, gpu in enumerate(gpus):
            with cols[i]:
                st.metric(label=f"GPU {gpu['ID']}", value=f"{gpu['Util (%)']}%", delta=f"{gpu['Temp (°C)']}°C")
    else:
        st.warning("Could not retrieve GPU info (nvidia-smi failed).")

    st.markdown("---")
    st.subheader("Process Status ⚙️")

    # Check for relevant processes: python running the eval script, or CarlaUE4
    relevant_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'username']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline:
                cmd_str = " ".join(cmdline)
                # Check for our eval script or CARLA
                if ("no_slurm_eval.py" in cmd_str) or ("CarlaUE4" in cmd_str) or ("leaderboard_evaluator.py" in cmd_str):
                    relevant_procs.append({
                        "PID": proc.info['pid'],
                        "Name": proc.info['name'],
                        "Command": cmd_str[:100] + "..." if len(cmd_str) > 100 else cmd_str,
                        "Started": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(proc.info['create_time'])),
                        "Status": proc.status()
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if relevant_procs:
        st.dataframe(pd.DataFrame(relevant_procs), use_container_width=True)
    else:
        st.info("No active Evaluation or CARLA processes found.")

# --- TAB 2: RESULTS VIEWER ---
with tab2:
    st.header("Evaluation Results")
    
    # Path selection (defaulting to the path in no_slurm_eval.py)
    default_path = os.path.join(os.getcwd(), "eval_results", "Bench2Drive")
    results_dir = st.text_input("Results Directory", value=default_path)
    
    if st.button("Load Results"):
        if not os.path.exists(results_dir):
            st.warning(f"Directory not found: {results_dir}")
        else:
            pattern = os.path.join(results_dir, "**", "*.json") # Relaxed pattern
            json_files = glob.glob(pattern, recursive=True)
            
            if not json_files:
                st.info("No JSON files found in directory.")
            else:
                results_data = []
                progress_bar = st.progress(0)
                
                for i, fpath in enumerate(json_files):
                    try:
                        # Skip typically non-result jsons if we can identify them
                        if "metrics.json" in fpath or "sensors.json" in fpath:
                            continue
                            
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                            
                            row = {"File": os.path.basename(fpath)}
                            
                            # Extract typical leaderboard metrics
                            # Structure: { "entry_status": "Finished", "scores": {"score_route": 100, ...} }
                            if 'entry_status' in data:
                                row['Status'] = data['entry_status']
                            
                            if 'scores' in data:
                                for k, v in data['scores'].items():
                                    row[k] = v
                                    
                            # Some formats might have it top level
                            if 'score_route' in data:
                                row['score_route'] = data['score_route']
                            
                            results_data.append(row)
                            
                    except:
                        pass
                    progress_bar.progress((i + 1) / len(json_files))
                
                if results_data:
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Metrics
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Metrics Distribution")
                        metric_to_plot = st.selectbox("Select Metric", numeric_cols)
                        st.bar_chart(df[metric_to_plot])
                else:
                    st.warning("Found JSON files but failed to extract recognizable result data.")

# --- TAB 3: JOB LAUNCHER ---
with tab3:
    st.header("Launch Evaluation")
    
    mode = st.radio("Execution Mode", ["Local Wrapper (no_slurm_eval.py)", "Expert Direct (leaderboard_evaluator.py)"])
    
    if mode == "Local Wrapper (no_slurm_eval.py)":
        st.markdown("### Configuration (Wrapper)")
        with st.expander("Environment Variables & Paths", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                carla_root = st.text_input("CARLA_ROOT", value="/mnt/SSD7/dow904/carla")
                work_dir = st.text_input("WORK_DIR (Repo Root)", value="/mnt/SSD7/Bench2Drive-VL")
                checkpoint = st.text_input("CHECKPOINT_PATH", value="/mnt/SSD7/dow904/simlingo/pytorch_model.pt")
                
            with col2:
                agent_file = st.text_input("AGENT_FILE", value=os.path.join(work_dir, "team_code/agent_simlingo.py"))
                route_path = st.text_input("ROUTE_PATH", value=os.path.join(work_dir, "leaderboard/data/bench2drive_split"))
                out_root = st.text_input("OUT_ROOT", value="./eval_results/Bench2Drive")

            col3, col4 = st.columns(2)
            with col3:
                vqa_gen = st.checkbox("VQA_GEN", value=True)
                strict_mode = st.checkbox("STRICT_MODE", value=True)
            with col4:
                st.text("System Paths usually derived derived from WORK_DIR")
        
        st.markdown("---")
        
        script_path = os.path.join(os.getcwd(), "no_slurm_eval.py")
        
        if os.path.exists(script_path):
            if st.button("🚀 Run Local Evaluation"):
                try:
                    # Prepare environment
                    custom_env = os.environ.copy()
                    custom_env["CARLA_ROOT"] = carla_root
                    custom_env["WORK_DIR"] = work_dir
                    custom_env["CHECKPOINT_PATH"] = checkpoint
                    custom_env["AGENT_FILE"] = agent_file
                    custom_env["ROUTE_PATH"] = route_path
                    custom_env["OUT_ROOT"] = out_root
                    
                    if vqa_gen: custom_env["VQA_GEN"] = "1"
                    if strict_mode: custom_env["STRICT_MODE"] = "1"
                    
                    # Also set derived variables requested by user
                    custom_env["SCENARIO_RUNNER_ROOT"] = os.path.join(work_dir, "scenario_runner")
                    custom_env["LEADERBOARD_ROOT"] = os.path.join(work_dir, "leaderboard")
                    
                    # Update PYTHONPATH to include these
                    existing_pythonpath = custom_env.get("PYTHONPATH", "")
                    new_paths = [
                        f"{carla_root}/PythonAPI",
                        f"{carla_root}/PythonAPI/carla",
                        f"{carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg",
                        f"{work_dir}/scenario_runner",
                        f"{work_dir}/leaderboard",
                        f"{work_dir}/B2DVL_Adapter",
                        existing_pythonpath
                    ]
                    custom_env["PYTHONPATH"] = ":".join(new_paths)

                    cmd = f"python -u {script_path}"
                    
                    log_file = "dashboard_local_run.log"
                    with open(log_file, "w") as out:
                        proc = subprocess.Popen(
                            cmd.split(), 
                            stdout=out, 
                            stderr=subprocess.STDOUT, 
                            cwd=os.getcwd(),
                            env=custom_env
                        )
                    
                    st.success(f"Started local evaluation! PID: {proc.pid}")
                    st.info(f"Logs are being written to {log_file}")
                    
                except Exception as e:
                    st.error(f"Failed to launch script: {e}")
            
            # Log Tail
            st.subheader("Log Output")
            log_file = "dashboard_local_run.log"
            if st.checkbox("Show Logs", value=True):
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        last_lines = lines[-20:]
                        st.code("".join(last_lines))
                else:
                    st.text("No log file found yet.")
        else:
            st.error(f"Script not found: {script_path}")

    elif mode == "Expert Direct (leaderboard_evaluator.py)":
        st.markdown("### Configuration (Direct)")
        
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host IP", value="140.113.203.91")
            port = st.number_input("Port", value=2000)
            gpu_id = st.text_input("CUDA_VISIBLE_DEVICES", value="3")
        
        with col2:
            agent = st.text_input("Agent Path", value="team_code/agent_simlingo.py")
            routes = st.text_input("Routes XML", value="/mnt/SSD7/dow904/Bench2Drive/leaderboard/data/bench2drive220.xml")
            checkpoint_json = st.text_input("Checkpoint JSON", value="simlingo_results.json")
            
        agent_config = st.text_input("Agent Config", value="$(pwd)/outputs/simlingo/checkpoints/pytorch_model.pt+test_v3")

        st.markdown("---")
        
        if st.button("🚀 Run Direct Evaluator"):
            try:
                # Construct command
                # Resolve $(pwd) if present
                if "$(pwd)" in agent_config:
                    agent_config = agent_config.replace("$(pwd)", os.getcwd())
                
                cmd = [
                    f"python", "leaderboard/leaderboard/leaderboard_evaluator.py",
                    f"--host={host}",
                    f"--port={port}",
                    f"--routes={routes}",
                    "--repetitions=1",
                    "--track=SENSORS",
                    f"--checkpoint={checkpoint_json}",
                    f"--agent={agent}",
                    f"--agent-config={agent_config}"
                ]
                
                full_cmd = " ".join(cmd)
                st.code(f"CUDA_VISIBLE_DEVICES={gpu_id} {full_cmd}", language="bash")
                
                # Env
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                
                log_file = "dashboard_direct_run.log"
                with open(log_file, "w") as out:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=out, 
                        stderr=subprocess.STDOUT, 
                        cwd=os.getcwd(),
                        env=env
                    )
                
                st.success(f"Started Direct Evaluator! PID: {proc.pid}")
                st.info(f"Logs: {log_file}")
                
            except Exception as e:
                st.error(f"Failed to launch: {e}")

        st.subheader("Log Output")
        log_file = "dashboard_direct_run.log"
        if st.checkbox("Show Logs (Direct)", value=True):
             if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    st.code("".join(lines[-20:]))
