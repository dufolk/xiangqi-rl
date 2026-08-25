import json
import os
import secrets
import urllib.request
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class LobbyConfig:
    player_name: str
    ai_count: int
    human_seat: int
    target_score: int


@dataclass
class LobbyInfo:
    lobby_id: str
    invite_code: str
    status: str
    config: LobbyConfig


class LobbyGateway:
    """Lobby transport boundary. Replace this adapter without changing game rules or UI."""

    def create_lobby(self, config: LobbyConfig) -> LobbyInfo:
        raise NotImplementedError

    def join_lobby(self, invite_code: str, player_name: str) -> LobbyInfo:
        raise NotImplementedError

    def send_action(self, lobby_id: str, action: Dict) -> None:
        raise NotImplementedError

    def poll_state(self, lobby_id: str) -> Optional[Dict]:
        raise NotImplementedError


class PreviewLobbyGateway(LobbyGateway):
    """Local placeholder used until a real lobby service endpoint is configured."""

    def create_lobby(self, config: LobbyConfig) -> LobbyInfo:
        code = secrets.token_hex(3).upper()
        return LobbyInfo(f"preview-{code.lower()}", code, "接口预留模式", config)

    def join_lobby(self, invite_code: str, player_name: str) -> LobbyInfo:
        config = LobbyConfig(player_name, 0, 0, 15)
        return LobbyInfo(f"preview-{invite_code.lower()}", invite_code.upper(), "接口预留模式", config)

    def send_action(self, lobby_id: str, action: Dict) -> None:
        return None

    def poll_state(self, lobby_id: str) -> Optional[Dict]:
        return None


class HttpLobbyGateway(LobbyGateway):
    """Minimal JSON/HTTP contract: POST /lobbies, POST /lobbies/{id}/join/actions, GET state."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def _request(self, method: str, path: str, payload: Optional[Dict] = None) -> Dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_lobby(self, config: LobbyConfig) -> LobbyInfo:
        data = self._request("POST", "/lobbies", asdict(config))
        return LobbyInfo(data["lobby_id"], data["invite_code"], data.get("status", "waiting"), config)

    def join_lobby(self, invite_code: str, player_name: str) -> LobbyInfo:
        data = self._request("POST", "/lobbies/join", {"invite_code": invite_code, "player_name": player_name})
        return LobbyInfo(data["lobby_id"], data["invite_code"], data.get("status", "waiting"), LobbyConfig(**data["config"]))

    def send_action(self, lobby_id: str, action: Dict) -> None:
        self._request("POST", f"/lobbies/{lobby_id}/actions", action)

    def poll_state(self, lobby_id: str) -> Optional[Dict]:
        return self._request("GET", f"/lobbies/{lobby_id}/state")


def create_lobby_gateway() -> LobbyGateway:
    endpoint = os.getenv("SPLENDOR_LOBBY_ENDPOINT", "").strip()
    return HttpLobbyGateway(endpoint) if endpoint else PreviewLobbyGateway()
