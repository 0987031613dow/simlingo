export CARLA_ROOT=/mnt/SSD7/dow904/carla  
export WORK_DIR=/mnt/SSD7/dow904/simlingo
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla  
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner  
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard  
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}