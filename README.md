# 璀璨宝石：四人标准局

一个使用 `pygame` 制作的璀璨宝石游戏。采用标准基础版 90 张发展卡（I/II/III 级分别为 40/30/20）和标准贵族条件，支持本地热座、生成式 AI/本地策略 AI、自定义玩家位次和获胜分数，并预留好友联网大厅接口。游戏包含 AI 生成的高清宝石、贵族与三级发展卡插画。

## 启动

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

当前项目目录已准备好 `.venv` 时，直接执行最后一行即可。

## 开局设置

启动后可输入玩家名称，并选择本局 AI 数量、自己的行动位次和获胜分数。可以选择单人竞技或 2v2 组队；组队时 1、3 号位为一队，2、4 号位为一队，团队率先达到 30 分后完成本轮并按团队总分结算。未配置外部模型时会自动使用本地策略 AI。

如需接入 OpenAI 兼容的生成式模型，可在启动前设置：

```bash
export SPLENDOR_AI_ENDPOINT="https://你的服务/v1/chat/completions"
export SPLENDOR_AI_API_KEY="你的密钥"
export SPLENDOR_AI_MODEL="模型名称"
```

## 局域网联机

1. 房主点击“创建局域网房间”，把大厅显示的 IP 发给朋友。
2. 客人在“房主 IP”中只输入该 IP，点击“加入房间”。程序固定使用端口 `48527`。
3. 房主在大厅用左右箭头给房主和已加入的客人安排座次，其余席位由 AI 接管。
4. 房主点击“开始联机游戏”。对局由房主权威执行并实时同步，客人右上角显示往返延迟。

设备需位于同一局域网；若系统防火墙询问是否允许 Python 接收入站连接，请选择允许。

## 项目结构

- `main.py`：桌面程序入口
- `app.py`：Pygame 界面、输入与会话协调
- `game_engine.py`：无 UI 的游戏模型和规则
- `game_ai.py`：合法动作、策略观测和动作执行
- `ai_player.py`：团队启发式与生成式模型策略
- `state_codec.py`：联机状态序列化和隐私过滤
- `network.py`：局域网 TCP 传输
- `rl/`：强化学习训练框架
  - `rl/environment.py`、`rl/policy.py`、`rl/ppo_policy.py`：单智能体视角的 PPO 环境与策略协议（历史接口）
  - `rl/mappo/`：**MAPPO / PPO 自博弈训练框架**（多智能体、CTDE、分布式 headless 收集、TensorBoard）

`rl.environment.SplendorEnv` 可在不启动 Pygame 的情况下执行 `reset()`、`legal_actions()`、`step()`，为后续自博弈和 PPO 训练提供稳定接口。PPO 依赖为可选项，不影响正常游戏运行。

## 操作

- 点击左侧彩色宝石选择：可拿 3 枚不同色，或在该颜色库存不少于 4 时双击同色拿 2 枚；然后点击“确认拿取”。
- 点击发展卡，再点击“购买所选”或“保留所选”。也可点击每级旁的“从牌库盲抽保留”；保留卡会获得 1 枚黄金（若库存仍有）。
- 购买后，符合条件的贵族会高亮，点击选择一位。
- 若宝石超过 10 枚，按提示点击左侧宝石归还。
- 有玩家达到本局设定分数后，本轮所有玩家完成行动，分数最高者获胜；同分时发展卡较少者获胜。
- 按 `R` 可随时重开当前配置的对局；按 `Esc` 返回开局设置。

---

# 强化学习训练（MAPPO / PPO 自博弈）

针对四人 / 2v2 对局的**多智能体强化学习**训练框架，位于 `rl/mappo/`。支持两种模式：

| 模式 | 参数 | 算法 | 奖励来源 | 对手 |
|------|------|------|----------|------|
| `--mode team` | 2v2 组队，目标 30 分 | **MAPPO（CTDE）** | 团队总分 `team_score` | 当前策略（self-play） |
| `--mode standard` | 4 人自由竞争，目标 15 分 | **PPO（参数共享自博弈）** | 个人 `points` | 当前策略（self-play） |

两种模式复用同一套 `ActorCritic` 网络与 `MAPPO` 更新逻辑，仅通过 `team_mode` 切换奖励与胜负判定。

## 依赖

- `torch`（已验证 2.11 + CUDA 12.8，RTX 5070 可用）
- `gymnasium`、`tensorboard`（可选，仅用于查看曲线）
- 训练完全 **headless**，无需 Pygame 窗口（`SDL_VIDEODRIVER=dummy`）。

## 一键训练脚本

`train_mappo.bat`（Windows）已封装全部参数：

```bat
train_mappo.bat                :: team 模式 (MAPPO/CTDE)，默认 200 万步
train_mappo.bat standard       :: standard 模式 (PPO/self-play)
train_mappo.bat team 5000000   :: 自定义总步数
```

脚本默认 7 个 worker（`cpu_count-1`）、horizon 64、每轮聚合 8 个 rollout、PPO 4 epoch、每 10 次更新做一次 self-play 评估。模型保存到 `checkpoints/`，TensorBoard 日志写到 `runs/`。

查看训练曲线：

```bash
tensorboard --logdir runs
```

也可直接调用（全部参数可见）：

```bash
python -m rl.mappo.main_train --mode team --timesteps 2000000 \
    --workers 7 --target-score 30 --save checkpoints/mappo_team.pt --tb-runs runs/team
```

## 训练管线

```
 多进程 Worker ×7 (CPU, headless)              主进程 (GPU / RTX 5070)
┌───────────────────────────────┐            ┌──────────────────────────────────┐
│ 各自跑一局 SplendorMAEnv       │ rollout    │ MAPPO.update：                  │
│ 当前策略 self-play 收集：       │ ─────────► │  PPO-clip 策略损失              │
│  obs, mask, action, logp,     │ batches    │  + 集中式 critic 价值损失       │
│  reward, value, done          │            │  + 熵正则                       │
│  + episode 级（总奖励/长度）   │ ◄───────── │  → 算 GAE 优势 → 多 epoch 优化  │
└───────────────────────────────┘ 权重广播   │  → 新权重 pickle 回传 worker    │
                                            └──────────────────────────────────┘
                                                          │
                                            TensorBoard：loss / value / entropy /
                                            kl / reward / length / win_rate
```

- **CTDE（集中式训练 / 去中心化执行）**：训练时 critic 接收**全局状态**（4 个 agent 观察的拼接），actor 只接收各自编码后的观察；执行时每个 agent 独立决策。
- **分布式收集**：7 个 worker 进程各自 headless 跑环境、用 CPU 做推理，通过 `multiprocessing.Queue` 把 rollout 回传主进程；主进程在 GPU 上做 MAPPO 更新后，把新权重广播给所有 worker。
- **Self-play**：收集阶段 4 个 seat 全部使用同一份（参数共享 + `agent_id` embedding 的）策略，即始终与"自己"对打。每 `eval_every` 次更新，`SelfPlayArena` 让当前策略坐 seat0、其余 seat 用随机基线，统计胜率写入 TensorBoard。

## 状态空间（observation）

`encoders.encode_observation` 把任意全局状态压成**定长 278 维** float 向量（`obs_dim`）。布局如下：

| 区块 | 维度 | 内容 |
|------|------|------|
| bank | 6 | 6 色宝石银行存量 |
| 全局标量 | 10 | 目标分、team_mode、两队分数(2)、round_end、当前玩家（座位号、宝石6、保留数、分数） |
| 4 名玩家 | 4×15 | 每玩家：分数 + 宝石(6) + 折扣(6) + 卡数 + 保留数 |
| 桌面卡 | 12×14 | 每张：tier(1) + bonus 颜色 one-hot(6) + points(1) + cost(6)，最多 12 张 |
| 贵族 | 5×6 | 每个贵族的宝石需求（最多 5 个） |

合计 `6 + 10 + 60 + 168 + 30 = 278`。

- **Actor 输入**：单个 agent 的 278 维观察 + `agent_id` one-hot(4)。
- **Critic 输入（全局状态）**：4 个 agent 观察拼接 = **1112 维**（`OBS_DIM * 4`）。

> 说明：当前每个 agent 观察均为"全局信息编码"，差异仅体现在 `current_player` 区块；critic 进一步拼接 4 份以符合 CTDE 的集中式输入约定，未来可在此处加入隐私屏蔽（仅暴露公开信息）。

## 动作空间（action）

游戏原生动作是变长的字符串 id（如 `buy:table:0:2`），不便直接做固定维度策略。这里**枚举成固定 62 维离散空间**（`ACTION_SPACE = 62`），并用 **action mask** 屏蔽非法动作（softmax 前对非法槽位置 `-1e9`）：

| 槽位区间 | 数量 | 动作 |
|----------|------|------|
| 0 – 11 | 12 | 买桌面卡：`tier×4 + pos` |
| 12 – 14 | 3 | 保留桌面卡 |
| 15 – 17 | 3 | 从牌库盲抽并保留（按 tier） |
| 18 – 20 | 3 | 买保留区卡 |
| 21 – (21+N_TAKE) | 40 | 拿宝石组合（3 枚异色 / 双同色，预枚举） |
| +5 | 5 | 选择贵族 |
| +6 | 6 | 归还某色宝石（超额时） |

执行时由 `slot_to_action_id` 把选中的槽位映射回游戏可识别的 action id。`build_action_mask` 依据 `game_ai.legal_actions` 生成当前合法掩码。

## 奖励设计

在 `envs.SplendorMAEnv.step` 中实现：

- **逐步奖励** = 该 agent 行动后分数增量
  - team 模式：本队 `team_score` 增量
  - standard 模式：个人 `points` 增量
- **终局奖励** = 胜 `+10` / 负 `−10`（按 `winner_team` / `winner` 判定）
- 信号较稀疏（仅得分/终局时非零），依赖 **GAE（γ=0.99, λ=0.95）** 把信用回溯到先前的拿宝石、保留、布局等决策。

## 网络结构（`networks.py`）

**Actor（去中心化，参数共享）**
```
输入: obs(278) 拼接 agent_id one-hot(4)  →  [278+4]
线性 → ReLU → 256 → 256 → 128 → 线性 → 62 (logits)
                                      │
                            mask 屏蔽非法 → Categorical 采样
```

**Centralized Critic（集中式）**
```
输入: global_state(1112 = 4×278)
线性 → ReLU → 512 → 256 → 128 → 线性 → 1 (V)
```

- Actor / Critic 各自独立 Adam（`lr_actor=3e-4`、`lr_critic=1e-3`），梯度裁剪 `max_grad_norm=0.5`。
- 损失：`L = L_clip^actor − c2·entropy + c1·value_loss`，PPO 裁剪 ε=0.2。
- 全部 `.to(cuda)` 运行在 5070；worker 用 CPU 推理。

## TensorBoard 曲线

日志前缀为 `team/` 或 `standard/`，包含：

- 训练：`policy_loss`、`value_loss`、`entropy`、`approx_kl`、`mean_reward_per_step`
- 整局：`mean_episode_reward`、`mean_episode_length`、`collected_steps`
- 自博弈评估（`eval/`）：`win_rate_seat0`（standard）或 `win_rate_team0`/`win_rate_team1`（team）、`mean_score`、`mean_length`

## 文件导航（`rl/mappo/`）

| 文件 | 职责 |
|------|------|
| `encoders.py` | 状态/动作编码、action mask、维度常量 |
| `envs.py` | `SplendorMAEnv` 回合制多智能体环境封装（复用 `game_engine`） |
| `networks.py` | 共享 Actor + 集中式 Critic |
| `buffer.py` | Rollout 缓冲 + GAE 优势计算 |
| `algo.py` | MAPPO 更新（PPO-clip + 价值 + 熵） |
| `collector.py` | 多进程 headless 分布式数据收集器 |
| `selfplay.py` | `SelfPlayArena` 自博弈胜率评估 |
| `main_train.py` | 训练入口（CLI + TensorBoard + 评估） |

## 后续可扩展

- **Fictitious Self-Play**：把 `SelfPlayArena` 的 `opponent_pool` 接成历史 checkpoint 对手池（目前评估用随机基线），提升训练稳定性。
- **隐私屏蔽**：在 `encode_observation` 中加入对手保留卡等隐蔽信息的可见性控制，更贴近真实 CTDE 假设。
- **课程学习**：先用较小目标分（如 15）预热，再逐步提升到 30。

