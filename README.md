# 木韵棋院 · 中国象棋 PPO 实验场

一个从零重构的中国象棋桌面项目：包含完整规则引擎、现代化 GUI、自定义棋局编辑器、红黑视角翻转、Pikafish 引擎对手，以及可直接扩展的 Masked PPO 自博弈训练平台。

## 功能

- 完整中国象棋走子规则：蹩马腿、塞象眼、炮架、过河兵、九宫、将帅照面、将军与困毙；
- 人机对弈与本地双人模式；AI 由 [Pikafish](https://github.com/official-pikafish/Pikafish)（NNUE 象棋引擎）驱动，`棋力` 按钮在 萌新 / 入门 / 进阶 / 高手 / 大师 五档间切换，AI 一侧会显示引擎评估与搜索深度；未安装引擎时自动退回 PPO 网络或 Alpha-Beta；
- 一键翻转红黑视角、悔棋、中文着法记录；
- 2.5D 木质棋盘：透视相机把整张木棋盘摆在桌面上，棋子是有厚度的木质圆棋；天空随局势变化（黑方占优时乌云闪电、相持时黄昏、红方占优时朝阳），两侧帅/將旗随士气升起或低垂；
- 棋盘上的剧情演出：轮到谁动，谁就“醒来”变成手绘卡通角色，在真实的格子上走位、出招、受击，演完再变回木棋。所有特效都锚定在棋盘坐标上，随棋子位置、翻转视角而变，关键时刻镜头会自动推近：
  - 吃子：42 种攻防组合各有专属招式与受击反应（车被打掉轮子、马被踢晕、炮被轰得满脸黑灰、象被压扁掉牙、士帽被打飞、小卒被弹上天）；
  - 局势决定吃子的戏码：大优时“乘胜追击”从容收割，相持时双方先拼刀“铛!”再分胜负，大劣时“绝地反击”火焰气场 + 慢动作 + 速度线；另有“扳回一城”“打破均势”标签；
  - 战术：捉双、闪击、闪将、双将、将军抽X（自动识别被抽的子，如“将军抽车”），各有光束、钩线、印章等专属表现；
  - 22 种开局各有短剧（当头炮的准星、过宫炮的九宫金光、巡河炮的小船探照灯、金钩炮的金钩钓鱼、仙人指路的云中仙手、飞相局的翅膀、屏风马的立体屏风、顺炮的雷电、列炮的交剑等）；
  - 12 种经典杀法各有一幕（双车错、卧槽马、挂角马、钓鱼马、马后炮、重炮、天地炮、铁门栓、大胆穿心、白脸将、海底捞月、二鬼拍门），王冠飞落、“将死”印章、胜方欢呼；外加困毙的铁笼；
- “观战”模式让两个 AI 对弈，配合剧情演出可以直接当动画看；`演出` 按钮在全程 / 精彩 / 关闭之间切换；`演出图鉴` 用 80 个精心摆好的局面逐个上演所有开局、吃子、局势、战术与杀法；
- 局面编辑器：任意摆子、右键擦除、切换先手、标准布局/清空；
- 中国象棋 FEN 导入和剪贴板导出；
- 当前行棋方规范化的 16×10×9 状态张量、8100 固定动作空间、合法动作掩码；
- 卷积残差 Actor-Critic 与 PPO/GAE 自博弈训练入口。

## 运行 GUI

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/fetch_pikafish.py   # 下载 Pikafish 引擎与 NNUE 权重到 engines/pikafish/
.venv/bin/python main.py
```

`tools/fetch_pikafish.py` 会从 Pikafish 官方 GitHub Release 下载最新版本，解压出当前平台（Windows / macOS / Linux x86-64）的通用二进制和 `pikafish.nnue`；加 `--all` 可一次装好三个平台，`--tag` 指定版本。也可以用环境变量 `PIKAFISH_PATH` 指向自己编译的引擎（`pikafish.nnue` 需与之同目录）。

| 棋力 | 搜索方式 |
| --- | --- |
| 萌新 | 深度 1，在前 6 条候选里按分差随机挑选 |
| 入门 | 深度 4，前 4 条候选随机 |
| 进阶 | 深度 8，前 3 条候选小幅随机 |
| 高手（默认） | 每步思考 0.8 秒，走最佳着法 |
| 大师 | 每步思考 2.5 秒，走最佳着法 |

快捷键：`F` 翻转棋盘，`U` 悔棋，`R` 新对局，`空格`/点击 跳过当前演出，`G` 打开演出图鉴（`←`/`→` 切换），`Esc` 退出。

### 离线录制与预览

```bash
# 无窗口录制一局 AI 观战（需要 ffmpeg）
python3 tools/record_match.py --out match.mp4 --seconds 90
# 连续录制演出图鉴，或只录指定条目
python3 tools/record_match.py --gallery --out gallery.mp4 --seconds 120
python3 tools/record_match.py --gallery-keys tactic/fork,mate/wocao_horse --out picks.mp4 --seconds 60
# 把单个演出渲染成 4×3 关键帧拼图（--at 输出指定时刻的整帧，all 渲染全部图鉴）
python3 tools/contact_sheet.py mate wocao_horse /tmp/wocao.png
python3 tools/contact_sheet.py all /tmp/sheets
```

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
| `ai_player.py` | Pikafish UCI 引擎封装、PPO 对手与 Alpha-Beta 基线 |
| `tools/fetch_pikafish.py` | 下载 Pikafish 引擎与 NNUE 权重 |
| `state_codec.py` | 状态序列化 |
| `cinema/detect.py` | 无 UI 的剧情识别：开局、吃子组合、局势/语气、战术（捉双/闪击/闪将/双将/抽子）、杀法图形 |
| `cinema/view3d.py` | 2.5D 透视相机、木质棋盘与棋子、天空/旗帜、深度排序与镜头推近 |
| `cinema/show.py` | 每一步棋的棋盘演出时间线：醒来、走位、吃子、剧情段落、入睡 |
| `cinema/board_story.py` | 开局、战术、将军、杀法、困毙的棋盘场景 |
| `cinema/demos.py` | 演出图鉴与测试使用的示范局面 |
| `cinema/actors.py` / `cinema/fx.py` | 程序化手绘角色与特效 |
| `tools/` | 离线录制与关键帧拼图工具 |
| `rl/environment.py` | PPO 环境、观测和动作掩码 |
| `rl/model.py` | Actor-Critic 网络 |
| `rl/train.py` | 多进程采样 PPO 自博弈训练入口与 TensorBoard 指标 |
| `rl/vector_env.py` | spawn 子进程并行环境采集器 |
| `tests/` | 规则、训练接口与并行环境测试 |

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
