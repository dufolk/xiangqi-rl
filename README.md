# 璀璨宝石：四人标准局

一个使用 `pygame` 制作的璀璨宝石游戏。支持本地热座、生成式 AI/本地策略 AI、自定义玩家位次和获胜分数，并预留好友联网大厅接口。游戏包含 AI 生成的高清宝石、贵族与三级发展卡插画。

## 启动

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

当前项目目录已准备好 `.venv` 时，直接执行最后一行即可。

## 开局设置

启动后可输入玩家名称，并选择本局 AI 数量、自己的行动位次和获胜分数。未配置外部模型时会自动使用本地策略 AI。

如需接入 OpenAI 兼容的生成式模型，可在启动前设置：

```bash
export SPLENDOR_AI_ENDPOINT="https://你的服务/v1/chat/completions"
export SPLENDOR_AI_API_KEY="你的密钥"
export SPLENDOR_AI_MODEL="模型名称"
```

“创建好友大厅”已经接入独立的网络网关接口。默认展示本地预览邀请码；配置 `SPLENDOR_LOBBY_ENDPOINT` 后，将按 JSON/HTTP 协议调用真实大厅服务。

## 操作

- 点击左侧彩色宝石选择：可拿 3 枚不同色，或在该颜色库存不少于 4 时双击同色拿 2 枚；然后点击“确认拿取”。
- 点击发展卡，再点击“购买所选”或“保留所选”。保留卡会获得 1 枚黄金（若库存仍有）。
- 购买后，符合条件的贵族会高亮，点击选择一位。
- 若宝石超过 10 枚，按提示点击左侧宝石归还。
- 有玩家达到本局设定分数后，本轮所有玩家完成行动，分数最高者获胜；同分时发展卡较少者获胜。
- 按 `R` 可随时重开当前配置的对局；按 `Esc` 返回开局设置。
