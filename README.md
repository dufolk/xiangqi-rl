# 赤霄 · 中国象棋 PPO 实验场

一个从零重构的中国象棋桌面项目：包含完整规则引擎、现代化 GUI、自定义棋局编辑器、红黑视角翻转、Alpha-Beta 基线对手，以及可直接扩展的 Masked PPO 自博弈训练平台。

## 功能

- 完整中国象棋走子规则：蹩马腿、塞象眼、炮架、过河兵、九宫、将帅照面、将军与困毙；
- 人机对弈与本地双人模式，AI 搜索深度可在 GUI 内切换；
- 一键翻转红黑视角、悔棋、中文着法记录；
- 内置“日系棋娘”和“峡谷英雄”双皮肤，可在对局中无缝切换；
- 开局、吃子、捉双、将军、绝杀与胜利均有当前皮肤专属角色播报和贴图；
- 局面编辑器：任意摆子、右键擦除、切换先手、标准布局/清空；
- 中国象棋 FEN 导入和剪贴板导出；
- 当前行棋方规范化的 16×10×9 状态张量、8100 固定动作空间、合法动作掩码；
- 卷积残差 Actor-Critic 与 PPO/GAE 自博弈训练入口。

## 运行 GUI

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

快捷键：`F` 翻转棋盘，`U` 悔棋，`R` 新对局，`S` 切换皮肤，`Esc` 退出。

### 自定义棋局

点击“局面编辑”，从右侧棋子工具箱选择棋子，再点击棋盘放置；右键点击棋盘可擦除。棋局至少需要红帅和黑将各一个，且不能直接照面。也可以导入如下格式的 FEN：

```text
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w
```

## PPO 训练

```bash
.venv/bin/python -m pip install -r requirements-train.txt
./train_ppo_tb.sh
```

脚本默认以 **8 个 spawned 子进程环境**并行收集数据；每个环境采样 128 个半回合，因此一次 PPO 更新的总批次为 \(8 \times 128 = 1024\) 个局面。策略网络仍在主进程的 MPS/CUDA 上批量推理，子进程只负责规则计算和合法着法生成。

可通过环境变量调整，例如：

```bash
NUM_ENVS=8 UPDATES=100 ROLLOUT_STEPS=128 EPOCHS=4 ./train_ppo_tb.sh
```

通过 `OPPONENT` 切换训练模式：默认 `selfplay` 为同一策略自对弈；设置为 `model` 则对抗一个冻结模型，并必须提供 `OPPONENT_MODEL`：

```bash
# 自对弈（默认）
OPPONENT=selfplay ./train_ppo_tb.sh

# 对抗固定模型；学习中的模型从头训练或通过 --resume 指定
OPPONENT=model OPPONENT_MODEL=checkpoints/frozen_pointer.pt ./train_ppo_tb.sh
```

策略采用两步自回归动作：交叉注意力先从 90 个起点中选择可走棋子，Pointer 再在 90 个目标格中选择落点；环境接口仍使用兼容的扁平动作编号。旧版 8100 维策略检查点与该架构不兼容，需从新训练或使用新版检查点。

对当前机器建议先使用 `NUM_ENVS=8`；若系统响应正常，可尝试 `NUM_ENVS=12` 或 `NUM_ENVS=16`。

默认每次启动都会**从头训练**。只有显式指定 checkpoint 才恢复训练；`UPDATES` 始终表示“在恢复点之后额外执行多少次更新”：

```bash
./train_ppo_tb.sh --resume checkpoints/xiangqi_ppo_20260828_120000.pt

# 也可使用环境变量
RESUME=checkpoints/xiangqi_ppo_20260828_120000.pt UPDATES=500 ./train_ppo_tb.sh
```

恢复时会加载模型权重、Adam 优化器状态、已完成更新次数、全局半回合数和已完成对局数；日志仍写入新的带时间戳目录，新的 checkpoint 也不会覆盖旧模型。

启动后，在另一个终端查看 TensorBoard：

```bash
.venv/bin/python -m tensorboard.main --logdir runs --port 6006
```

浏览器打开 `http://localhost:6006`。每次训练使用带时间戳的 `runs/ppo_*` 日志目录与 `checkpoints/xiangqi_ppo_*.pt` 检查点。

TensorBoard 包含：总策略/价值损失、熵、近似 KL、裁剪比例、梯度范数、解释方差、学习率、平均合法着法数量、采样奖励、吃子数、将军数、对局步数、红黑胜率、截断率、吃子价值、将死/困毙终局类型、并行环境数、PPO 批次大小、每次更新耗时和并行采样吞吐。当前训练器已采用多进程采样基线；后续可继续加入对手池、Elo 联赛或 MCTS。

### 强化学习接口

- `rl.environment.XiangqiEnv`：`reset()`、`step(action)`、`legal_actions()`、`action_mask()`；
- 观测：当前行棋方视角下的 14 个己敌棋子平面 + 己方/敌方受将平面；
- 动作：规范视角中的 `起点索引 × 90 + 终点索引`，总计 8100 个槽位；
- `rl.model.build_model()`：六层 GroupNorm 残差骨干 + 起终点配对策略头 + 合法动作 mask；
- 结构图：见 `docs/ppo-network.md`；
- `rl.train`：PPO clip、GAE、熵正则、价值损失和梯度裁剪。

## 项目结构

| 路径 | 职责 |
|---|---|
| `main.py` / `app.py` | 桌面入口与 Pygame GUI |
| `game_engine.py` | 无 UI 规则引擎、FEN、走子与撤销 |
| `ai_player.py` | Alpha-Beta 基线 AI |
| `state_codec.py` | 状态序列化 |
| `rl/environment.py` | PPO 环境、观测和动作掩码 |
| `rl/model.py` | Actor-Critic 网络 |
| `rl/train.py` | 多进程采样 PPO 自博弈训练入口与 TensorBoard 指标 |
| `rl/vector_env.py` | spawn 子进程并行环境采集器 |
| `tests/` | 规则、训练接口与并行环境测试 |

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
