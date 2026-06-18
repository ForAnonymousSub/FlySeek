# # 简单：UAV 高空全程稳跟
# # --seed 固定随机种子，保证每次运行的目标小汽车初始化位置可复现，
# # 避免随机种子偶发命中 PCD 中的水面/桥外区域（标记为 drivable 的伪道路点）。
# python flyseek_extend/scripts/demo_hide_and_seek.py \
#   --env env_airsim_16 --init-profile standard \
#   --target SM_ClassicCar02_Drivable6_4 \
#   --tracking-difficulty easy \
#   --seed 42

# # 中等：经历一次遮挡 → reacquire
# python flyseek_extend/scripts/demo_hide_and_seek.py \
#   --env env_airsim_16 --init-profile standard \
#   --target SM_ClassicCar02_Drivable6_4 \
#   --tracking-difficulty medium \
#   --seed 43

# # 困难：跟丢
# python flyseek_extend/scripts/demo_hide_and_seek.py \
#   --env env_airsim_16 --init-profile standard \
#   --target SM_ClassicCar02_Drivable6_4 \
#   --tracking-difficulty hard \
#   --seed 44

# python flyseek_extend/scripts/demo_adversary_chase.py --auto-from-scout --env env_airsim_16 \
#   --target-behavior direct_escape     --target-policy-difficulty medium --seed 42

# python flyseek_extend/scripts/demo_adversary_chase.py --auto-from-scout --env env_airsim_16 \
#   --target-behavior sharp_turn        --target-policy-difficulty hard   --seed 42

# python flyseek_extend/scripts/demo_adversary_chase.py --auto-from-scout --env env_airsim_16 \
#   --target-behavior detour_feint      --target-policy-difficulty medium --seed 42

# python flyseek_extend/scripts/demo_adversary_chase.py --auto-from-scout --env env_airsim_16 \
#   --target-behavior occlusion_seeking --target-policy-difficulty hard --seed 42
# (occlusion_seeking uses PCD-validated open_then_hide route: main road → alley hide)

# # Generate one episode (real, AirVLN running):
# python flyseek_extend/flyseek_bench/run_generate_episodes.py \
#   --scene_id env_airsim_16 --difficulty medium --behavior sharp_turn \
#   --seed 42 --num_episodes 1 --output_dir flyseek_extend/output/bench

# # Generate a small batch (offline, no simulator — fully exercises structure + sanity):
# python -m flyseek_bench.run_generate_episodes \
#   --scene_id env_airsim_16 --difficulty hard --behavior occlusion_seeking \
#   --seed 42 --num_episodes 3 --output_dir flyseek_extend/output/bench --dry_run

# # Compute metrics for an existing episode:
# python -m flyseek_bench.metrics --episode_dir flyseek_extend/output/bench/<episode_id>

# for AirSim — occlusion_seeking: allow ~25s of hutong/building hide on screen
python flyseek_extend/scripts/demo_adversary_chase.py \
  --env env_airsim_16 \
  --auto-from-scout \
  --init-profile standard \
  --target-behavior occlusion_seeking \
  --target-policy-difficulty hard \
  --seed 66 \
  --duration 75 \
  --open-road-frac 0.45 \
  --route-len-m 150 \
  --min-building-height-m 20 \
  --min-building-footprint-cells 12 \
  --hide-search-radius-m 55 \
  --route-search-radius-m 240 \
  --route-max-attempts 20 \
  --min-building-occluded-frac 0.65 \
  --building-probe-dist-m 9 \
  --require-adjacent-building


# Hutong demo: car drives into narrow gap between annotated buildings
python flyseek_extend/scripts/demo_alley_chase.py \
  --env env_airsim_16 \
  --auto-from-scout \
  --seed 66 \
  --duration 55 \
  --open-approach-m 35


# Paired videos: drone tracking SUCCESS (open escape) + FAIL (hide behind seg_map buildings)
# Requires AirSim env_airsim_16 running.
python flyseek_extend/scripts/demo_chase_pair.py \
  --env env_airsim_16 \
  --auto-from-scout \
  --init-profile standard \
  --shared-seed 66 \
  --duration 75 \
  --seg-building-jsonl scene_data/seg_map/env_airsim_16.jsonl \
  --route-len-m 120 \
  --open-road-frac 0.4


# Single episode — annotated building hide only (drone should lose track):
python flyseek_extend/scripts/demo_adversary_chase.py \
  --env env_airsim_16 \
  --auto-from-scout \
  --init-profile standard \
  --target-behavior occlusion_seeking \
  --target-policy-difficulty hard \
  --seg-building-jsonl scene_data/seg_map/env_airsim_16.jsonl \
  --seed 66 \
  --duration 75 \
  --route-len-m 120 \
  --open-road-frac 0.4


# for UE5
python flyseek_extend/scripts/demo_unrealcv_chase.py --env env_ue_smallcity --target-behavior direct_escape     --target-policy-difficulty easy   --duration 30
python flyseek_extend/scripts/demo_unrealcv_chase.py --env env_ue_smallcity --target-behavior sharp_turn        --target-policy-difficulty medium --duration 30
python flyseek_extend/scripts/demo_unrealcv_chase.py --env env_ue_smallcity --target-behavior detour_feint      --target-policy-difficulty medium --duration 30
python flyseek_extend/scripts/demo_unrealcv_chase.py --env env_ue_smallcity --target-behavior occlusion_seeking --target-policy-difficulty hard   --duration 30

 bash envs/airsim/env_airsim_16/LinuxNoEditor/start.sh 

 cd "$(dirname "$0")/../.."   # OpenFly-Platform root (hosts flyseek_extend/)
# 仅几何(CPU)
bash flyseek_extend/shell/demo_gs_ommo_urban.sh
# 几何 + 渲染合成(GPU)
bash flyseek_extend/shell/demo_gs_ommo_urban.sh --render