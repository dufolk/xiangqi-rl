import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


class GenerativeAIPlayer:
    """OpenAI-compatible model adapter with a deterministic local strategy fallback."""

    def __init__(self):
        self.endpoint = os.getenv("SPLENDOR_AI_ENDPOINT", "").strip()
        self.api_key = os.getenv("SPLENDOR_AI_API_KEY", "").strip()
        self.model = os.getenv("SPLENDOR_AI_MODEL", "gpt-4o-mini").strip()
        self.last_mode = "本地策略"

    @property
    def model_enabled(self) -> bool:
        return bool(self.endpoint and self.api_key)

    def choose_action(self, snapshot: Dict[str, Any], actions: List[Dict[str, Any]]) -> str:
        if not actions:
            return ""
        if self.model_enabled:
            try:
                action_id = self._ask_model(snapshot, actions)
                if action_id in {action["id"] for action in actions}:
                    self.last_mode = "生成式模型"
                    return action_id
            except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError):
                pass
        self.last_mode = "团队策略" if snapshot.get("team_mode") else "本地策略"
        return self.local_choice(snapshot, actions)

    def _ask_model(self, snapshot: Dict[str, Any], actions: List[Dict[str, Any]]) -> str:
        prompt = {
            "role": "你是璀璨宝石桌游高手。根据局面只从 legal_actions 选择一个 id，兼顾得分、折扣、贵族和阻止对手。",
            "state": snapshot,
            "legal_actions": actions,
            "output": {"action_id": "必须精确复制一个合法 id"},
        }
        body = json.dumps({
            "model": self.model,
            "temperature": 0.35,
            "max_tokens": 80,
            "messages": [
                {"role": "system", "content": "只输出 JSON，不要解释。"},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        return str(json.loads(content)["action_id"])

    @staticmethod
    def local_choice(snapshot: Dict[str, Any], actions: List[Dict[str, Any]]) -> str:
        """Score legal actions using personal progress plus 2v2 team context."""
        team_mode = bool(snapshot.get("team_mode"))
        target = int(snapshot.get("target_score", 15))
        team_scores = snapshot.get("team_scores") or [0, 0]
        current = snapshot.get("current_player", {})
        seat = int(current.get("seat", 0))
        own_score = int(current.get("score", 0))
        own_team = seat % 2
        team_gap = target - int(team_scores[own_team]) if team_mode else target - own_score
        enemy_gap = target - int(team_scores[1 - own_team]) if team_mode else target

        def score(action: Dict[str, Any]) -> float:
            kind = action.get("type")
            points = float(action.get("points", 0))
            noble_progress = float(action.get("noble_progress", 0))
            enemy_threat = float(action.get("enemy_threat", 0))
            teammate_interest = float(action.get("teammate_interest", 0))
            if kind == "noble":
                return 2000 + points * 100
            if kind == "buy":
                value = 500 + points * 90 + action.get("tier", 0) * 8 + noble_progress * 28
                if points >= team_gap:
                    value += 5000
                if team_mode and enemy_gap <= 5:
                    value += points * 35
                return value
            if kind == "reserve":
                value = 100 + points * 18 + action.get("tier", 0) * 5 + enemy_threat * 75
                if team_mode:
                    value -= teammate_interest * 45
                    if enemy_gap <= 5:
                        value += enemy_threat * 80
                return value
            if kind == "take":
                value = 180 + float(action.get("usefulness", 0)) * 8
                if team_mode:
                    value += float(action.get("deny_value", 0)) * 4
                    value -= float(action.get("teammate_need", 0)) * 2
                return value
            if kind == "return":
                return 80 - float(action.get("usefulness", 0)) * 8
            return 0

        return max(actions, key=lambda action: (score(action), action["id"]))["id"]
