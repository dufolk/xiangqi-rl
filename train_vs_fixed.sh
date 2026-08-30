#!/usr/bin/env bash
# 训练智能体对抗「固定的本地最新模型」(frozen teacher)。
# 当近 100 局智能体胜率稳定达到 100% 时自动停止并保存模型。
# 运行： bash train_vs_fixed.sh   （或直接在 PowerShell 执行下面的 python 命令）
set -euo pipefail
cd "$(dirname "$0")"

LATEST=checkpoints/xiangqi_ppo_continued.pt

python -m rl.train \
  --resume "$LATEST" \
  --opponent model \
  --opponent-model "$LATEST" \
  --updates 2000 \
  --num-envs 16 \
  --rollout-steps 128 \
  --output checkpoints/xiangqi_ppo_vs_fixed.pt \
  --log-dir runs/vs_fixed \
  --save-interval 10 \
  --target-win-rate 1.0 \
  --win-rate-window 100 \
  --min-updates-before-stop 10
