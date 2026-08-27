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
- 16×10×9 状态张量、8100 固定动作空间、合法动作掩码；
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
.venv/bin/python -m rl.train --updates 100 --rollout-steps 512
```

检查点默认保存到 `checkpoints/xiangqi_ppo.pt`。当前训练器是清晰、可运行的单进程基线，接口已将 GUI 与训练完全解耦，后续可直接加入并行环境、对手池、Elo 联赛或 MCTS。

### 强化学习接口

- `rl.environment.XiangqiEnv`：`reset()`、`step(action)`、`legal_actions()`、`action_mask()`；
- 观测：14 个红黑棋子平面 + 先手平面 + 最后落点平面；
- 动作：`起点索引 × 90 + 终点索引`，总计 8100 个槽位；
- `rl.model.build_model()`：带合法动作 mask 的卷积残差 Actor-Critic；
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
| `rl/train.py` | PPO 自博弈训练入口 |
| `tests/` | 规则与训练接口测试 |

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
