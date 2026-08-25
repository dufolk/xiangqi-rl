import json
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

LAN_PORT = 48527
MAX_MESSAGE_BYTES = 2_000_000


@dataclass
class PeerInfo:
    peer_id: str
    name: str


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _send(sock: socket.socket, message: Dict, lock: Optional[threading.Lock] = None) -> None:
    payload = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("network message is too large")
    if lock:
        with lock:
            sock.sendall(payload)
    else:
        sock.sendall(payload)


def _messages(sock: socket.socket):
    buffer = b""
    while True:
        data = sock.recv(65536)
        if not data:
            return
        buffer += data
        if len(buffer) > MAX_MESSAGE_BYTES:
            raise ValueError("network receive buffer is too large")
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line:
                yield json.loads(line.decode("utf-8"))


class LanHost:
    def __init__(self, player_name: str, port: int = LAN_PORT):
        self.player_name = player_name
        self.port = port
        self.host_peer_id = "host"
        self.events: queue.Queue = queue.Queue()
        self._server: Optional[socket.socket] = None
        self._clients: Dict[str, socket.socket] = {}
        self._client_locks: Dict[str, threading.Lock] = {}
        self._peers: Dict[str, PeerInfo] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()

    @property
    def address(self) -> str:
        return f"{local_ip()}:{self.port}"

    @property
    def peers(self) -> List[PeerInfo]:
        with self._lock:
            return list(self._peers.values())

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(4)
        server.settimeout(0.5)
        self._server = server
        self._running.set()
        threading.Thread(target=self._accept_loop, name="splendor-lan-host", daemon=True).start()

    def _accept_loop(self) -> None:
        while self._running.is_set():
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.settimeout(None)
            threading.Thread(target=self._client_loop, args=(client,), daemon=True).start()

    def _client_loop(self, client: socket.socket) -> None:
        peer_id = ""
        try:
            iterator = _messages(client)
            hello = next(iterator)
            if hello.get("type") != "hello":
                raise ValueError("expected hello")
            peer_id = uuid.uuid4().hex[:8]
            name = str(hello.get("name", "客人"))[:12] or "客人"
            send_lock = threading.Lock()
            with self._lock:
                self._clients[peer_id] = client
                self._client_locks[peer_id] = send_lock
                self._peers[peer_id] = PeerInfo(peer_id, name)
            _send(client, {"type": "welcome", "peer_id": peer_id}, send_lock)
            self.events.put({"type": "connected", "peer_id": peer_id, "name": name})
            for message in iterator:
                message["peer_id"] = peer_id
                if message.get("type") == "ping":
                    _send(client, {"type": "pong", "sent_at": message.get("sent_at")}, send_lock)
                else:
                    self.events.put(message)
        except (OSError, ValueError, StopIteration, json.JSONDecodeError):
            pass
        finally:
            if peer_id:
                with self._lock:
                    self._clients.pop(peer_id, None)
                    self._client_locks.pop(peer_id, None)
                    self._peers.pop(peer_id, None)
                self.events.put({"type": "disconnected", "peer_id": peer_id})
            try:
                client.close()
            except OSError:
                pass

    def send_to(self, peer_id: str, message: Dict) -> None:
        with self._lock:
            client = self._clients.get(peer_id)
            send_lock = self._client_locks.get(peer_id)
        if client:
            try:
                _send(client, message, send_lock)
            except OSError:
                pass

    def broadcast(self, message: Dict) -> None:
        with self._lock:
            peer_ids = list(self._clients)
        for peer_id in peer_ids:
            self.send_to(peer_id, message)

    def poll(self) -> List[Dict]:
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        self._running.clear()
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass


class LanClient:
    def __init__(self, player_name: str):
        self.player_name = player_name
        self.peer_id = ""
        self.events: queue.Queue = queue.Queue()
        self.rtt_ms: Optional[float] = None
        self.connected = False
        self.error = ""
        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._running = threading.Event()

    def connect(self, host: str, port: int = LAN_PORT, timeout: float = 4.0) -> None:
        if ":" in host:
            host, port_text = host.rsplit(":", 1)
            port = int(port_text)
        sock = socket.create_connection((host.strip(), port), timeout=timeout)
        sock.settimeout(None)
        self._sock = sock
        self._running.set()
        _send(sock, {"type": "hello", "name": self.player_name}, self._send_lock)
        threading.Thread(target=self._receive_loop, name="splendor-lan-client", daemon=True).start()
        threading.Thread(target=self._ping_loop, name="splendor-lan-ping", daemon=True).start()

    def _receive_loop(self) -> None:
        try:
            for message in _messages(self._sock):
                kind = message.get("type")
                if kind == "welcome":
                    self.peer_id = message["peer_id"]
                    self.connected = True
                elif kind == "pong":
                    sent_at = float(message.get("sent_at", time.monotonic()))
                    self.rtt_ms = max(0.0, (time.monotonic() - sent_at) * 1000)
                else:
                    self.events.put(message)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        finally:
            self.connected = False
            if self._running.is_set():
                self.events.put({"type": "disconnected"})

    def _ping_loop(self) -> None:
        while self._running.is_set():
            if self.connected:
                self.send({"type": "ping", "sent_at": time.monotonic()})
            time.sleep(1)

    def send(self, message: Dict) -> None:
        if not self._sock:
            return
        try:
            _send(self._sock, message, self._send_lock)
        except OSError:
            self.connected = False

    def poll(self) -> List[Dict]:
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        self._running.clear()
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
