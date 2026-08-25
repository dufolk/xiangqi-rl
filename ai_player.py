import json
import os
import random
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
        self.last_mode = "本地策略"
        return self.local_choice(actions)

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
    def local_choice(actions: List[Dict[str, Any]]) -> str:
        def score(action: Dict[str, Any]) -> float:
            kind = action.get("type")
            if kind == "buy":
                return 100 + action.get("points", 0) * 30 + action.get("tier", 0) * 5 + random.random()
            if kind == "noble":
                return 200 + random.random()
            if kind == "take":
                return 30 + action.get("usefulness", 0) * 4 + random.random()
            if kind == "reserve":
                return 20 + action.get("points", 0) * 4 + action.get("tier", 0) * 2 + random.random()
            if kind == "return":
                return 10 - action.get("usefulness", 0) + random.random()
            return random.random()

        return max(actions, key=score)["id"]
