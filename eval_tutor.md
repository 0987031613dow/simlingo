
On carla server:


CUDA_VISIBLE_DEVICES=3 python leaderboard/leaderboard/leaderboard_evaluator.py \
--host 140.113.203.91 \
--port 2000 \
--routes /mnt/SSD7/dow904/Bench2Drive/leaderboard/data/bench2drive220.xml \
--repetitions 1 \
--track SENSORS \
--checkpoint simlingo_results.json \
--agent team_code/agent_simlingo.py \
--agent-config $(pwd)/outputs/simlingo/checkpoints/pytorch_model.pt+test_v3



watch -n 1 nvidia-smi




========= Results of RouteScenario_1711 (repetition 0) ------ SUCCESS =========

╒═══════════════════════╤═════════════════════╕
│ Start Time            │ 2026-01-30 06:03:40 │
├───────────────────────┼─────────────────────┤
│ End Time              │ 2026-01-30 06:20:29 │
├───────────────────────┼─────────────────────┤
│ System Time           │ 1008.89s            │
├───────────────────────┼─────────────────────┤
│ Game Time             │ 18.35s              │
├───────────────────────┼─────────────────────┤
│ Ratio (Game / System) │ 0.018               │
╘═══════════════════════╧═════════════════════╛

╒═══════════════════════╤═════════╤══════════╕
│ Criterion             │ Result  │ Value    │
├───────────────────────┼─────────┼──────────┤
│ RouteCompletionTest   │ SUCCESS │ 100 %    │
├───────────────────────┼─────────┼──────────┤
│ OutsideRouteLanesTest │ SUCCESS │ 0 %      │
├───────────────────────┼─────────┼──────────┤
│ CollisionTest         │ SUCCESS │ 0 times  │
├───────────────────────┼─────────┼──────────┤
│ RunningRedLightTest   │ SUCCESS │ 0 times  │
├───────────────────────┼─────────┼──────────┤
│ RunningStopTest       │ SUCCESS │ 0 times  │
├───────────────────────┼─────────┼──────────┤
│ MinSpeedTest          │ SUCCESS │ 103.48 % │
├───────────────────────┼─────────┼──────────┤
│ InRouteTest           │ SUCCESS │          │
├───────────────────────┼─────────┼──────────┤
│ AgentBlockedTest      │ SUCCESS │          │
├───────────────────────┼─────────┼──────────┤
│ Timeout               │ SUCCESS │          │
╘═══════════════════════╧═════════╧══════════╛

