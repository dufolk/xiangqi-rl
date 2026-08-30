#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/runs/ppo_${STAMP}"
CHECKPOINT="$ROOT/checkpoints/xiangqi_ppo_${STAMP}.pt"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法：./train_ppo_tb.sh

可选环境变量：
  UPDATES=1000          PPO 更新次数
  NUM_ENVS=8            并行环境进程数
  ROLLOUT_STEPS=128     每个环境每次更新采样半回合数
  EPOCHS=4              每批 PPO 训练轮数
  LEARNING_RATE=2.5e-4  学习率
  SAVE_INTERVAL=25      检查点保存间隔
  OPPONENT=selfplay     对手模式：selfplay（自对弈）、model（固定模型）或 search（传统搜索）
  OPPONENT_MODEL=路径   固定模型路径（仅 OPPONENT=model 时必填）
  OPPONENT_STOCHASTIC=0 设为 1 时固定模型按概率采样，否则贪心走子
  SEARCH_DEPTH=3        搜索算法深度（仅 OPPONENT=search 时生效）
  SEARCH_QUIESCENCE=0   设为 1 开启静态搜索（吃子延伸），更强但明显更慢

恢复训练：
  ./train_ppo_tb.sh --resume checkpoints/xiangqi_ppo_YYYYMMDD_HHMMSS.pt
  # 或 RESUME=checkpoints/xxx.pt ./train_ppo_tb.sh

固定模型对战训练（随机红黑）：
  OPPONENT=model OPPONENT_MODEL=checkpoints/frozen_pointer.pt ./train_ppo_tb.sh

对抗搜索模式训练（随机红黑，搜索深度 4）：
  OPPONENT=search SEARCH_DEPTH=4 ./train_ppo_tb.sh
EOF
  exit 0
fi

RESUME_PATH="${RESUME:-}"
if [[ "${1:-}" == "--resume" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "--resume 需要提供 checkpoint 文件路径" >&2
    exit 2
  fi
  RESUME_PATH="$2"
  shift 2
fi
if [[ $# -ne 0 ]]; then
  echo "未知参数：$1" >&2
  echo "使用 ./train_ppo_tb.sh --help 查看用法" >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "未找到虚拟环境：$PYTHON"
  echo "请先执行：python3 -m venv .venv"
  exit 1
fi

"$PYTHON" -m pip install -r "$ROOT/requirements-train.txt"
mkdir -p "$LOG_DIR" "$(dirname "$CHECKPOINT")"

if [[ -n "$RESUME_PATH" ]]; then
  if [[ ! -f "$RESUME_PATH" ]]; then
    echo "找不到 resume checkpoint：$RESUME_PATH" >&2
    exit 2
  fi
fi

OPPONENT_MODE="${OPPONENT:-selfplay}"
OPPONENT_MODEL_PATH="${OPPONENT_MODEL:-}"
case "$OPPONENT_MODE" in
  selfplay|search) ;;
  model)
    if [[ -z "$OPPONENT_MODEL_PATH" ]]; then
      echo "OPPONENT=model 时必须设置 OPPONENT_MODEL=检查点路径" >&2
      exit 2
    fi
    if [[ ! -f "$OPPONENT_MODEL_PATH" ]]; then
      echo "找不到固定对手模型：$OPPONENT_MODEL_PATH" >&2
      exit 2
    fi
    ;;
  *)
    echo "OPPONENT 必须是 selfplay、model 或 search，当前为：$OPPONENT_MODE" >&2
    exit 2
    ;;
esac

echo "TensorBoard 日志：$LOG_DIR"
echo "模型检查点：$CHECKPOINT"
if [[ -n "$RESUME_PATH" ]]; then
  echo "恢复模型：$RESUME_PATH"
else
  echo "恢复模型：无（从头训练）"
fi
echo "对手模式：$OPPONENT_MODE"
if [[ "$OPPONENT_MODE" == "model" ]]; then
  echo "固定对手模型：$OPPONENT_MODEL_PATH"
fi
echo "训练启动后，可在另一个终端运行："
echo "  $PYTHON -m tensorboard.main --logdir $ROOT/runs --port 6006"

set -- \
  --updates "${UPDATES:-1000}" \
  --num-envs "${NUM_ENVS:-8}" \
  --rollout-steps "${ROLLOUT_STEPS:-128}" \
  --epochs "${EPOCHS:-4}" \
  --learning-rate "${LEARNING_RATE:-2.5e-4}" \
  --save-interval "${SAVE_INTERVAL:-25}" \
  --opponent "$OPPONENT_MODE" \
  --search-depth "${SEARCH_DEPTH:-3}" \
  $( [ "${SEARCH_QUIESCENCE:-0}" = "1" ] && echo "--search-quiescence" ) \
  --log-dir "$LOG_DIR" \
  --output "$CHECKPOINT"
if [[ "$OPPONENT_MODE" == "model" ]]; then
  set -- "$@" --opponent-model "$OPPONENT_MODEL_PATH"
  if [[ "${OPPONENT_STOCHASTIC:-0}" == "1" ]]; then
    set -- "$@" --opponent-stochastic
  fi
fi
if [[ -n "$RESUME_PATH" ]]; then
  set -- "$@" --resume "$RESUME_PATH"
fi
exec "$PYTHON" -m rl.train "$@"
