"""Premium animated Pygame interface for Chinese chess."""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "0")

import pygame

from ai_player import PPOAI
from game_engine import (
    ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, KINDS, PAWN, RED, ROOK,
    Move, Piece, XiangqiGame,
)

VW, VH = 1440, 900
BOARD_X, BOARD_Y, CELL = 96, 96, 76
GRID_W, GRID_H = CELL * 8, CELL * 9
SIDE_X = 790

INK = (19, 13, 39)
INK_2 = (31, 21, 56)
PANEL = (39, 29, 67)
PANEL_LIGHT = (61, 43, 91)
TEXT = (255, 246, 255)
MUTED = (188, 169, 207)
CYAN = (113, 205, 255)
CYAN_LIGHT = (202, 235, 255)
CYAN_DIM = (73, 128, 174)
RED_C = (234, 72, 126)
RED_LIGHT = (255, 137, 180)
RED_DARK = (104, 28, 68)
GOLD = (247, 194, 103)
GOLD_LIGHT = (255, 232, 174)
JADE = (124, 225, 194)
BLACK_C = (31, 28, 62)
BOARD_INK = (195, 160, 234)

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
ANIME_DIR = os.path.join(ASSET_DIR, "jpanime")
CHARACTER_DIR = os.path.join(ASSET_DIR, "jpanime2")
RIFT_DIR = os.path.join(ASSET_DIR, "lol")
CHARACTER_NAMES = {
    KING: "主将姬", ADVISOR: "近卫姬", ELEPHANT: "灵象姬", HORSE: "骑战姬",
    ROOK: "战车姬", CANNON: "炮术姬", PAWN: "先锋姬",
}
RIFT_CHARACTER_NAMES = {
    (RED, KING): "嘉文四世", (BLACK, KING): "斯维因",
    (RED, ADVISOR): "拉克丝", (BLACK, ADVISOR): "乐芙兰",
    (RED, ELEPHANT): "加里奥", (BLACK, ELEPHANT): "赛恩",
    (RED, HORSE): "瑟庄妮", (BLACK, HORSE): "赫卡里姆",
    (RED, ROOK): "波比", (BLACK, ROOK): "厄加特",
    (RED, CANNON): "崔丝塔娜", (BLACK, CANNON): "金克丝",
    (RED, PAWN): "盖伦", (BLACK, PAWN): "德莱厄斯",
}
RIFT_QUOTES = {
    (RED, KING): ("为了德玛西亚！", "长枪所向，阵线随我推进。", "王者从不把胜利交给偶然。"),
    (BLACK, KING): ("胜局早已写定。", "我看见的，比你想象得更远。", "真正的力量来自全局。"),
    (RED, ADVISOR): ("光明会指引我们。", "让这束光照亮破绽。", "别眨眼，精彩才刚开始。"),
    (BLACK, ADVISOR): ("真假，只在一念之间。", "你看到的，未必是真相。", "这一步，也在计划之中。"),
    (RED, ELEPHANT): ("巨像，挺身而战！", "只要我在，防线就在。", "该让石翼舒展一下了。"),
    (BLACK, ELEPHANT): ("死亡也无法阻止我。", "撞碎他们的防线。", "疼痛，只会让我更强。"),
    (RED, HORSE): ("凛冬正在逼近！", "踏过冰原，也踏过阵线。", "没有人能躲过寒冬。"),
    (BLACK, HORSE): ("践踏他们的阵线！", "暗影之蹄已经锁定目标。", "逃吧，我喜欢追逐。"),
    (RED, ROOK): ("锤子可不是摆设！", "这一锤，专治坚固防线。", "英雄也要从小事做起。"),
    (BLACK, ROOK): ("弱者终将被碾碎。", "恐惧是最精确的武器。", "改造完成，开始清场。"),
    (RED, CANNON): ("瞄准，然后——轰！", "炮口已经校准！", "这一发会很响亮。"),
    (BLACK, CANNON): ("规则？炸掉就好啦！", "三、二、一——惊喜！", "别跑，火箭喜欢会动的目标。"),
    (RED, PAWN): ("正义，永不动摇！", "稳住阵线，步步向前。", "无畏之心，永不退后。"),
    (BLACK, PAWN): ("诺克萨斯，无可阻挡！", "再向前一步，就是征服。", "只有强者能留下。"),
}
ANIME_QUOTES = {
    KING: ("全员听令，按计划推进。", "棋盘的节奏，由我掌握。", "别担心，我会带大家回去。"),
    ADVISOR: ("你的破绽，我已经看见了。", "这一步，请交给我吧。", "守护与反击，我都很擅长。"),
    ELEPHANT: ("先稳住阵线，再寻找机会。", "结界展开，休想越界。", "耐心也是制胜的一部分。"),
    HORSE: ("从侧翼切进去！", "跟得上我的速度吗？", "下一站，就是你的死角。"),
    ROOK: ("让开，战车要加速了！", "正面突破才够痛快。", "这条线路，现在归我。"),
    CANNON: ("弹道确认，准备发射！", "漂亮的爆炸要选好角度。", "听见倒计时了吗？"),
    PAWN: ("一步也不会退。", "小角色也能改变战局。", "前进，就是我的答案。"),
}
EVENT_QUOTES = {
    "opening": ("序幕已经拉开。", "先声夺人，漂亮的开场。", "布局完成，接下来才是正戏。", "就从这一手开始改写局面。"),
    "fork": ("两个目标，一个都别想走。", "选择吧——无论哪个都要付出代价。", "一箭双雕，机会抓住了。", "同时锁定，战术成立。"),
    "check": ("将军。现在轮到你回答。", "王城告急，这一步必须应对。", "退路正在一格格消失。", "这一击，直指中枢。"),
    "capture": ("关键目标已经清除。", "交换成立，优势到手。", "这枚棋子，我就收下了。", "突破口已经打开。"),
    "mate": ("胜负已定。", "终局，就写在这一手。", "没有退路了。", "到此为止——漂亮的终章。"),
}


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    candidates = {
        "darwin": ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"],
        "win32": [os.path.join(os.environ.get("SystemRoot", "C:/Windows"), "Fonts", "msyh.ttc")],
    }.get(sys.platform, ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"])
    for path in candidates:
        if os.path.exists(path):
            font = pygame.font.Font(path, size)
            font.set_bold(bold)
            return font
    return pygame.font.SysFont("Arial Unicode MS", size, bold=bold)


def ease_out_cubic(value: float) -> float:
    return 1.0 - (1.0 - value) ** 3


def ease_out_back(value: float) -> float:
    c1, c3 = 1.70158, 2.70158
    return 1.0 + c3 * (value - 1.0) ** 3 + c1 * (value - 1.0) ** 2


@dataclass
class UIButton:
    key: str
    rect: pygame.Rect
    label: str
    active: bool = False
    accent: bool = False


@dataclass
class MoveAnimation:
    move: Move
    piece: Piece
    started_at: int
    duration: int = 300

    def progress(self, now: int) -> float:
        return min(1.0, max(0.0, (now - self.started_at) / self.duration))


@dataclass
class TacticalCallout:
    title: str
    subtitle: str
    commentary: str
    color: tuple[int, int, int]
    started_at: int
    actor: Piece
    victim: Optional[Piece] = None
    duration: int = 2600

    def progress(self, now: int) -> float:
        return min(1.0, max(0.0, (now - self.started_at) / self.duration))


class XiangqiApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("赤霄棋院 · 中国象棋")
        self.window = pygame.display.set_mode((VW, VH), pygame.RESIZABLE)
        self.canvas = pygame.Surface((VW, VH)).convert()
        self.clock = pygame.time.Clock()
        self.fonts = {
            "tiny": load_font(14), "small": load_font(17), "body": load_font(20),
            "button": load_font(17, True), "title": load_font(31, True),
            "piece": load_font(35, True), "hero": load_font(43, True),
            "seal": load_font(23, True), "river": load_font(30, True),
            "callout": load_font(64, True), "callout_small": load_font(18, True),
        }
        self.piece_cache: dict[tuple[str, str, int], pygame.Surface] = {}
        self.portrait_cache: dict[tuple[str, str, int], pygame.Surface] = {}
        self.skin_assets = {
            "anime": {
                "background": self._load_nested_asset("background"),
                "board": self._load_nested_asset("board2"),
                "characters": self._load_character_art(CHARACTER_DIR),
                "victory": {
                    RED: self._load_nested_asset("victory_red"),
                    BLACK: self._load_nested_asset("victory_black"),
                },
            },
            "rift": {
                "background": self._load_nested_asset("background2", RIFT_DIR),
                "board": self._load_nested_asset("board2", RIFT_DIR),
                "characters": self._load_character_art(RIFT_DIR),
                "victory": {
                    RED: self._load_nested_asset("victory_red", RIFT_DIR),
                    BLACK: self._load_nested_asset("victory_black2", RIFT_DIR),
                },
            },
        }
        self.skin = "anime"
        self.background_texture: Optional[pygame.Surface] = None
        self.wood_texture: Optional[pygame.Surface] = None
        self.character_art: dict[tuple[str, str], pygame.Surface] = {}
        self.victory_art: dict[str, Optional[pygame.Surface]] = {}
        self.activate_skin("anime")
        self.game = XiangqiGame()
        snapshots = __import__("glob").glob("checkpoints/ppo_snapshot_*.pt")
        pinned = max(snapshots, key=os.path.getmtime) if snapshots else None
        self.ai = PPOAI(checkpoint=pinned, depth=2)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xiangqi-ai")
        self.ai_future: Optional[Future] = None
        self.ai_position = ""
        self.mode = "play"
        self.opponent_mode = "ai"
        self.human_side = RED
        self.flipped = False
        self.selected: Optional[tuple[int, int]] = None
        self.legal_moves: list[Move] = []
        self.editor_piece: Optional[Piece] = Piece(RED, ROOK)
        self.buttons: list[UIButton] = []
        self.palette: list[tuple[pygame.Rect, Piece]] = []
        self.move_log: list[str] = []
        self.message = "红方先行 · 请落子"
        self.modal = ""
        self.text_input = ""
        self.move_animation: Optional[MoveAnimation] = None
        self.pending_board_click: Optional[tuple[tuple[int, int], int]] = None
        self.callout: Optional[TacticalCallout] = None
        self.last_callout_halfmove = -10
        self.commentary = "棋娘们已经就位，等待主人的第一道指令。"
        self.shake_until = 0
        self.flash_until = 0
        self.running = True
        try:
            pygame.scrap.init()
        except pygame.error:
            pass

    @staticmethod
    def _load_asset(prefix: str) -> Optional[pygame.Surface]:
        if not os.path.isdir(ASSET_DIR):
            return None
        match = next((name for name in os.listdir(ASSET_DIR) if name.startswith(prefix) and name.lower().endswith(".png")), None)
        if not match:
            return None
        try:
            return pygame.image.load(os.path.join(ASSET_DIR, match)).convert()
        except pygame.error:
            return None

    @staticmethod
    def _load_anime_asset(prefix: str) -> Optional[pygame.Surface]:
        if not os.path.isdir(ANIME_DIR):
            return None
        match = next((name for name in os.listdir(ANIME_DIR) if name.startswith(prefix) and name.lower().endswith(".png")), None)
        if not match:
            return None
        try:
            return pygame.image.load(os.path.join(ANIME_DIR, match)).convert()
        except pygame.error:
            return None

    @staticmethod
    def _load_nested_asset(folder: str, root: str = ANIME_DIR) -> Optional[pygame.Surface]:
        directory = os.path.join(root, folder)
        if not os.path.isdir(directory):
            return None
        match = next((name for name in os.listdir(directory) if name.lower().endswith(".png")), None)
        if not match:
            return None
        try:
            return pygame.image.load(os.path.join(directory, match)).convert()
        except pygame.error:
            return None

    def _load_character_art(self, root: str) -> dict[tuple[str, str], pygame.Surface]:
        art: dict[tuple[str, str], pygame.Surface] = {}
        for kind in KINDS:
            directory = os.path.join(root, kind)
            if not os.path.isdir(directory):
                continue
            match = next((name for name in os.listdir(directory) if name.lower().endswith(".png")), None)
            if not match:
                continue
            try:
                sheet = pygame.image.load(os.path.join(directory, match)).convert()
            except pygame.error:
                continue
            half = sheet.get_width() // 2
            crop_size = min(560, half, sheet.get_height())
            left_x = max(0, (half - crop_size) // 2)
            right_x = half + max(0, (half - crop_size) // 2)
            y = max(0, int(sheet.get_height() * 0.03))
            art[(RED, kind)] = sheet.subsurface((left_x, y, crop_size, crop_size)).copy()
            art[(BLACK, kind)] = sheet.subsurface((right_x, y, crop_size, crop_size)).copy()
        return art

    def activate_skin(self, skin: str) -> None:
        if skin not in self.skin_assets:
            return
        self.skin = skin
        bundle = self.skin_assets[skin]
        self.background_texture = bundle["background"]
        self.wood_texture = bundle["board"]
        self.character_art = bundle["characters"]
        self.victory_art = bundle["victory"]
        self.piece_cache.clear()
        self.portrait_cache.clear()
        title = "棋灵物语 · 棋娘对弈" if skin == "anime" else "峡谷棋阵 · 英雄对弈"
        pygame.display.set_caption(title)

    def character_name(self, piece: Piece) -> str:
        if self.skin == "rift":
            return RIFT_CHARACTER_NAMES[(piece.side, piece.kind)]
        return CHARACTER_NAMES[piece.kind]

    @staticmethod
    def tactic_key(title: str) -> str:
        if "绝杀" in title or title.replace(" ", "") == "绝杀":
            return "mate"
        if "将军" in title or "卧槽马" in title:
            return "check"
        if "捉" in title:
            return "fork"
        if title in ("当 头 炮", "过 宫 炮", "仙 人 指 路", "起 马 局", "屏 风 马", "飞 相 局", "卒 底 炮"):
            return "opening"
        return "capture"

    def character_quote(self, piece: Piece, title: str) -> str:
        event_key = self.tactic_key(title)
        event_quotes = EVENT_QUOTES[event_key]
        character_quotes = RIFT_QUOTES[(piece.side, piece.kind)] if self.skin == "rift" else ANIME_QUOTES[piece.kind]
        pool = character_quotes + event_quotes
        seed = self.game.halfmove * 13 + piece.kind.__hash__() + piece.side.__hash__() + sum(ord(char) for char in title)
        return pool[seed % len(pool)]

    def character_portrait(self, piece: Piece, size: int) -> Optional[pygame.Surface]:
        key = (piece.side, piece.kind, size)
        if key in self.portrait_cache:
            return self.portrait_cache[key]
        source = self.character_art.get((piece.side, piece.kind))
        if source is None:
            return None
        portrait = pygame.transform.smoothscale(source, (size, size)).convert_alpha()
        mask = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2 - 2)
        portrait.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.portrait_cache[key] = portrait
        return portrait

    def virtual_mouse(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Map OS-window mouse coordinates to the fixed 1440×900 canvas.

        On macOS Retina displays, SDL may expose a physical framebuffer that is
        twice the size of the logical window used by mouse events. Converting
        through both dimensions keeps hit-testing aligned with the rendered UI.
        """
        framebuffer_width, framebuffer_height = self.window.get_size()
        try:
            logical_width, logical_height = pygame.display.get_window_size()
        except pygame.error:
            logical_width, logical_height = framebuffer_width, framebuffer_height
        logical_width = logical_width or framebuffer_width
        logical_height = logical_height or framebuffer_height
        pixel_x = pos[0] * framebuffer_width / logical_width
        pixel_y = pos[1] * framebuffer_height / logical_height
        scale = min(framebuffer_width / VW, framebuffer_height / VH)
        offset_x = (framebuffer_width - VW * scale) / 2
        offset_y = (framebuffer_height - VH * scale) / 2
        return int(round((pixel_x - offset_x) / scale)), int(round((pixel_y - offset_y) / scale))

    def display_square(self, row: int, col: int) -> tuple[int, int]:
        if self.flipped:
            row, col = 9 - row, 8 - col
        return BOARD_X + col * CELL, BOARD_Y + row * CELL

    def board_square(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        col = round((pos[0] - BOARD_X) / CELL)
        row = round((pos[1] - BOARD_Y) / CELL)
        if not (0 <= row < 10 and 0 <= col < 9):
            return None
        x, y = BOARD_X + col * CELL, BOARD_Y + row * CELL
        # 棋子已改为方牌；使用方形命中区而非旧圆形半径，边角单击也能立即响应。
        tolerance = CELL * 0.495
        if abs(pos[0] - x) > tolerance or abs(pos[1] - y) > tolerance:
            return None
        return (9 - row, 8 - col) if self.flipped else (row, col)

    def draw_text(self, text: str, font: pygame.font.Font, color: tuple[int, int, int], pos: tuple[int, int], center: bool = False) -> None:
        surface = font.render(text, True, color)
        self.canvas.blit(surface, surface.get_rect(center=pos) if center else pos)

    def draw_glass(self, rect: pygame.Rect, color: tuple[int, int, int] = PANEL, alpha: int = 232, radius: int = 18) -> None:
        shadow = pygame.Surface((rect.w + 12, rect.h + 12), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (10, 4, 28, 95), shadow.get_rect(), border_radius=radius + 3)
        self.canvas.blit(shadow, (rect.x + 3, rect.y + 5))
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, (*color, alpha), layer.get_rect(), border_radius=radius)
        highlight = (240, 209, 133) if self.skin == "rift" else (255, 220, 255)
        outline = (86, 175, 195) if self.skin == "rift" else (213, 163, 255)
        pygame.draw.line(layer, (*highlight, 80), (radius, 1), (rect.w - radius, 1), 1)
        pygame.draw.rect(layer, (*outline, 92), layer.get_rect(), 1, border_radius=radius)
        pygame.draw.circle(layer, (*highlight, 22), (rect.w - 44, 30), 38)
        self.canvas.blit(layer, rect)

    def add_button(self, key: str, x: int, y: int, w: int, label: str, active: bool = False, accent: bool = False) -> None:
        button = UIButton(key, pygame.Rect(x, y, w, 42), label, active, accent)
        self.buttons.append(button)
        mouse = self.virtual_mouse(pygame.mouse.get_pos())
        hovered = button.rect.collidepoint(mouse) and not self.modal
        if self.skin == "rift":
            if accent:
                fill, border = ((54, 132, 158) if hovered else (30, 91, 111)), GOLD_LIGHT
            elif active:
                fill, border = ((40, 102, 116) if hovered else (26, 76, 89)), GOLD
            else:
                fill, border = ((28, 67, 75) if hovered else (15, 43, 51)), ((111, 192, 210) if hovered else (64, 116, 127))
            shadow_color, shine = (1, 10, 15), (226, 206, 147)
        else:
            if accent:
                fill, border = ((238, 89, 145) if hovered else (198, 58, 119)), (255, 183, 215)
            elif active:
                fill, border = ((92, 66, 139) if hovered else (70, 50, 112)), (211, 169, 255)
            else:
                fill, border = ((70, 52, 99) if hovered else (48, 36, 76)), ((205, 165, 237) if hovered else (105, 82, 139))
            shadow_color, shine = (16, 8, 35), (255, 231, 249)
        shadow = button.rect.move(0, 3)
        pygame.draw.rect(self.canvas, shadow_color, shadow, border_radius=13)
        pygame.draw.rect(self.canvas, fill, button.rect, border_radius=13)
        pygame.draw.rect(self.canvas, border, button.rect, 1, border_radius=13)
        pygame.draw.line(self.canvas, shine, (x + 14, y + 2), (x + w - 14, y + 2), 1)
        self.draw_text(label, self.fonts["button"], TEXT, button.rect.center, center=True)

    def draw_background(self) -> None:
        if self.background_texture:
            texture = pygame.transform.smoothscale(self.background_texture, (VW, VH))
            self.canvas.blit(texture, (0, 0))
            veil = pygame.Surface((VW, VH), pygame.SRCALPHA)
            veil.fill((8, 18, 28, 66) if self.skin == "rift" else (25, 10, 54, 58))
            self.canvas.blit(veil, (0, 0))
        else:
            self.canvas.fill(INK)
        vignette = pygame.Surface((VW, VH), pygame.SRCALPHA)
        for index in range(10):
            pygame.draw.rect(vignette, (3, 9, 16, 11) if self.skin == "rift" else (12, 3, 29, 10), (index * 6, index * 6, VW - index * 12, VH - index * 12), 7, border_radius=30)
        self.canvas.blit(vignette, (0, 0))
        now = pygame.time.get_ticks()
        particles = pygame.Surface((VW, VH), pygame.SRCALPHA)
        for index in range(26):
            x = int((index * 173 + now * (0.012 + index % 3 * 0.004)) % VW)
            y = int((index * 97 + now * (0.018 + index % 4 * 0.003)) % VH)
            if self.skin == "rift":
                color = (80, 196, 255, 58) if index % 2 else (242, 192, 92, 58)
                pygame.draw.polygon(particles, color, [(x, y - 5), (x + 4, y), (x, y + 5), (x - 4, y)])
            else:
                pygame.draw.ellipse(particles, (255, 181, 220, 65), (x, y, 8 + index % 5, 4 + index % 3))
        self.canvas.blit(particles, (0, 0))

    def _wood_panel(self, rect: pygame.Rect) -> None:
        if self.wood_texture:
            material = pygame.transform.smoothscale(self.wood_texture, rect.size)
            self.canvas.blit(material, rect)
        else:
            pygame.draw.rect(self.canvas, (47, 34, 79), rect)
        tint = pygame.Surface(rect.size, pygame.SRCALPHA)
        tint.fill((4, 24, 30, 38) if self.skin == "rift" else (35, 18, 70, 42))
        for index in range(14):
            x = (index * 83) % rect.w
            y = (index * 127) % rect.h
            if self.skin == "rift":
                pygame.draw.circle(tint, (69, 186, 225, 24), (x, y), 4 + index % 4, 1)
            else:
                pygame.draw.circle(tint, (255, 192, 228, 24), (x, y), 4 + index % 4)
        self.canvas.blit(tint, rect)

    def _draw_board_marker(self, center: tuple[int, int], edges: tuple[bool, bool, bool, bool] = (True, True, True, True)) -> None:
        x, y = center
        length, gap = 10, 5
        color = (75, 180, 207) if self.skin == "rift" else BOARD_INK
        left, right, up, down = edges
        if left:
            pygame.draw.lines(self.canvas, color, False, [(x - gap - length, y - gap), (x - gap, y - gap), (x - gap, y - gap - length)], 2)
            pygame.draw.lines(self.canvas, color, False, [(x - gap - length, y + gap), (x - gap, y + gap), (x - gap, y + gap + length)], 2)
        if right:
            pygame.draw.lines(self.canvas, color, False, [(x + gap + length, y - gap), (x + gap, y - gap), (x + gap, y - gap - length)], 2)
            pygame.draw.lines(self.canvas, color, False, [(x + gap + length, y + gap), (x + gap, y + gap), (x + gap, y + gap + length)], 2)

    def draw_board(self) -> None:
        frame = pygame.Rect(29, 25, 730, 832)
        if self.skin == "rift":
            shadow, frame_fill = (1, 8, 12), (20, 42, 48)
            outer, inner, rail = (196, 154, 72), (61, 155, 193), (39, 128, 167)
            board_outer, board_inner = (202, 161, 81), (79, 192, 225)
        else:
            shadow, frame_fill = (18, 7, 37), (72, 46, 101)
            outer, inner, rail = (245, 183, 224), (255, 225, 164), (203, 151, 229)
            board_outer, board_inner = (238, 182, 229), (255, 228, 170)
        pygame.draw.rect(self.canvas, shadow, frame.move(8, 10), border_radius=26)
        pygame.draw.rect(self.canvas, frame_fill, frame, border_radius=26)
        pygame.draw.rect(self.canvas, outer, frame, 3, border_radius=26)
        pygame.draw.rect(self.canvas, inner, frame.inflate(-12, -12), 2, border_radius=20)
        for x in (frame.x + 17, frame.right - 23):
            pygame.draw.line(self.canvas, rail, (x, frame.y + 34), (x, frame.bottom - 34), 2)
        board_rect = pygame.Rect(55, 50, 678, 782)
        self._wood_panel(board_rect)
        pygame.draw.rect(self.canvas, board_outer, board_rect, 2, border_radius=12)
        pygame.draw.rect(self.canvas, board_inner, board_rect.inflate(-8, -8), 1, border_radius=9)
        grid_shadow = (14, 58, 69) if self.skin == "rift" else (88, 56, 126)
        grid_color = (75, 180, 207) if self.skin == "rift" else BOARD_INK

        for row in range(10):
            y = BOARD_Y + row * CELL
            pygame.draw.line(self.canvas, grid_shadow, (BOARD_X, y + 2), (BOARD_X + GRID_W, y + 2), 4)
            pygame.draw.line(self.canvas, grid_color, (BOARD_X, y), (BOARD_X + GRID_W, y), 1)
        for col in range(9):
            x = BOARD_X + col * CELL
            segments = [(BOARD_Y, BOARD_Y + GRID_H)] if col in (0, 8) else [(BOARD_Y, BOARD_Y + 4 * CELL), (BOARD_Y + 5 * CELL, BOARD_Y + GRID_H)]
            for top, bottom in segments:
                pygame.draw.line(self.canvas, grid_shadow, (x + 2, top), (x + 2, bottom), 4)
                pygame.draw.line(self.canvas, grid_color, (x, top), (x, bottom), 1)
        for base_row in (0, 7):
            y1, y3 = BOARD_Y + base_row * CELL, BOARD_Y + (base_row + 2) * CELL
            pygame.draw.line(self.canvas, grid_color, (BOARD_X + 3 * CELL, y1), (BOARD_X + 5 * CELL, y3), 1)
            pygame.draw.line(self.canvas, grid_color, (BOARD_X + 5 * CELL, y1), (BOARD_X + 3 * CELL, y3), 1)

        river_y = BOARD_Y + 4.5 * CELL
        river_layer = pygame.Surface((GRID_W, CELL - 4), pygame.SRCALPHA)
        if self.skin == "rift":
            pygame.draw.rect(river_layer, (16, 79, 94, 88), river_layer.get_rect(), border_radius=12)
            for offset in range(12, GRID_W, 62):
                pygame.draw.circle(river_layer, (234, 191, 91, 48), (offset, 18 + offset % 25), 5, 1)
            labels = (("蓝方 · 河 道", (116, 213, 255)), ("峡 谷 · 红方", (255, 125, 116)))
        else:
            pygame.draw.rect(river_layer, (105, 63, 151, 70), river_layer.get_rect(), border_radius=12)
            for offset in range(12, GRID_W, 62):
                pygame.draw.ellipse(river_layer, (255, 188, 224, 42), (offset, 12 + offset % 17, 13, 7))
            labels = (("楚 河 · 绯 樱", (255, 195, 225)), ("苍 月 · 漢 界", (185, 221, 255)))
        pygame.draw.line(river_layer, (*labels[0][1], 100), (0, 2), (GRID_W, 2), 1)
        pygame.draw.line(river_layer, (*labels[1][1], 90), (0, CELL - 7), (GRID_W, CELL - 7), 1)
        self.canvas.blit(river_layer, (BOARD_X, int(river_y - CELL / 2 + 2)))
        self.draw_text(labels[0][0], self.fonts["river"], labels[0][1], (238, int(river_y)), center=True)
        self.draw_text(labels[1][0], self.fonts["river"], labels[1][1], (556, int(river_y)), center=True)

        for row, col in ((2, 1), (2, 7), (7, 1), (7, 7)):
            self._draw_board_marker(self.display_square(row, col))
        for row in (3, 6):
            for col in range(0, 9, 2):
                self._draw_board_marker(self.display_square(row, col), (col > 0, col < 8, True, True))

        files = "九八七六五四三二一" if not self.flipped else "一二三四五六七八九"
        for col, label in enumerate(files):
            self.draw_text(label, self.fonts["tiny"], CYAN_DIM, (BOARD_X + col * CELL, 816), center=True)

        now = pygame.time.get_ticks()
        pulse = (math.sin(now / 180) + 1) / 2
        if self.game.last_move:
            for square in (self.game.last_move.start, self.game.last_move.end):
                x, y = self.display_square(*square)
                pygame.draw.circle(self.canvas, CYAN, (x, y), 34, 2)
                pygame.draw.circle(self.canvas, CYAN_LIGHT, (x, y), 29, 1)
        if self.selected:
            x, y = self.display_square(*self.selected)
            aura = pygame.Surface((90, 90), pygame.SRCALPHA)
            pygame.draw.circle(aura, (*CYAN, 38), (45, 45), int(35 + pulse * 5))
            pygame.draw.circle(aura, (*CYAN_LIGHT, 230), (45, 45), int(34 + pulse * 2), 3)
            self.canvas.blit(aura, (x - 45, y - 45))
        for move in self.legal_moves:
            x, y = self.display_square(*move.end)
            if move.captured:
                pygame.draw.circle(self.canvas, RED_LIGHT, (x, y), int(32 + pulse * 2), 3)
                pygame.draw.circle(self.canvas, (255, 188, 206), (x, y), 25, 1)
            else:
                pygame.draw.circle(self.canvas, (2, 42, 52), (x + 2, y + 3), int(8 + pulse * 2))
                pygame.draw.circle(self.canvas, CYAN, (x, y), int(7 + pulse * 2))

        animation_end = self.move_animation.move.end if self.move_animation else None
        for row in range(10):
            for col in range(9):
                piece = self.game.board[row][col]
                if piece and (row, col) != animation_end:
                    center = self.display_square(row, col)
                    if self.selected == (row, col):
                        center = (center[0], center[1] - int(3 + pulse * 2))
                    self.draw_piece(piece, center, 31)
        self._draw_move_animation(now)
        self._draw_check_effect(now)

    def piece_surface(self, piece: Piece, radius: int) -> pygame.Surface:
        key = (piece.side, piece.kind, radius)
        if key in self.piece_cache:
            return self.piece_cache[key]
        size = radius * 2 + 28
        center = size // 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        team = RED_LIGHT if piece.side == RED else CYAN
        ink = (151, 37, 73) if piece.side == RED else (30, 67, 117)
        face = (255, 239, 225) if piece.side == RED else (229, 242, 251)
        face_inner = (248, 214, 205) if piece.side == RED else (198, 222, 239)
        half = radius + 3
        cut = max(7, radius // 3)

        def tile_points(cx: int, cy: int, extent: int, corner: int) -> list[tuple[int, int]]:
            return [
                (cx - extent + corner, cy - extent), (cx + extent - corner, cy - extent),
                (cx + extent, cy - extent + corner), (cx + extent, cy + extent - corner),
                (cx + extent - corner, cy + extent), (cx - extent + corner, cy + extent),
                (cx - extent, cy + extent - corner), (cx - extent, cy - extent + corner),
            ]

        for glow_extent, alpha in ((half + 5, 18), (half + 3, 34)):
            glow = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.polygon(glow, (*team, alpha), tile_points(center, center, glow_extent, cut + 2))
            surface.blit(glow, (0, 0))
        pygame.draw.polygon(surface, team, tile_points(center, center, half, cut))
        pygame.draw.polygon(surface, face, tile_points(center, center - 1, half - 3, max(5, cut - 2)))
        pygame.draw.polygon(surface, face_inner, tile_points(center, center, half - 7, max(4, cut - 4)), 2)

        portrait = self.character_portrait(piece, max(28, radius * 2 - 12))
        if portrait is not None:
            watermark = portrait.copy()
            watermark.set_alpha(34 if self.skin == "anime" else 42)
            surface.blit(watermark, watermark.get_rect(center=(center, center - 1)))

        font = self.fonts["piece"] if radius >= 30 else self.fonts["seal"]
        glyph_shadow = font.render(piece.name, True, (255, 255, 255))
        glyph = font.render(piece.name, True, ink)
        for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            ghost = glyph_shadow.copy()
            ghost.set_alpha(210)
            surface.blit(ghost, ghost.get_rect(center=(center + ox, center + oy - 1)))
        surface.blit(glyph, glyph.get_rect(center=(center, center - 2)))

        pygame.draw.line(surface, (255, 255, 255, 175), (center - half + cut + 2, center - half + 3), (center + half - cut - 2, center - half + 3), 2)
        pygame.draw.line(surface, (*ink, 145), (center - half + cut + 3, center + half - 4), (center + half - cut - 3, center + half - 4), 2)
        dot_color = GOLD if self.skin == "rift" else team
        pygame.draw.circle(surface, dot_color, (center + half - 8, center - half + 8), 3)
        self.piece_cache[key] = surface
        return surface

    def draw_piece(self, piece: Piece, center: tuple[int, int], radius: int, alpha: int = 255) -> None:
        surface = self.piece_surface(piece, radius)
        if alpha != 255:
            surface = surface.copy()
            surface.set_alpha(alpha)
        self.canvas.blit(surface, surface.get_rect(center=center))

    def _draw_move_animation(self, now: int) -> None:
        animation = self.move_animation
        if not animation:
            return
        raw = animation.progress(now)
        start = self.display_square(*animation.move.start)
        end = self.display_square(*animation.move.end)
        progress = ease_out_cubic(raw)
        x = start[0] + (end[0] - start[0]) * progress
        y = start[1] + (end[1] - start[1]) * progress - math.sin(raw * math.pi) * 17
        trail = pygame.Surface((VW, VH), pygame.SRCALPHA)
        energy = RED_LIGHT if animation.piece.side == RED else CYAN
        for index in range(1, 7):
            t = max(0.0, progress - index * 0.038)
            tx = start[0] + (end[0] - start[0]) * t
            ty = start[1] + (end[1] - start[1]) * t - math.sin(t * math.pi) * 15
            pygame.draw.circle(trail, (*energy, max(0, 58 - index * 8)), (int(tx), int(ty)), 26 - index * 2)
        self.canvas.blit(trail, (0, 0))
        if animation.move.captured and raw < 0.72:
            capture_alpha = int(255 * (1 - raw / 0.72))
            capture_scale = max(20, int(31 * (1 - raw * 0.22)))
            self.draw_piece(animation.move.captured, end, capture_scale, capture_alpha)
        if animation.move.captured and raw > 0.45:
            burst = min(1.0, (raw - 0.45) / 0.45)
            ring_radius = int(18 + burst * 48)
            fx = pygame.Surface((130, 130), pygame.SRCALPHA)
            pygame.draw.circle(fx, (*energy, int(210 * (1 - burst))), (65, 65), ring_radius, 3)
            for index in range(12):
                angle = index * math.tau / 12 + 0.25
                distance = 18 + burst * (28 + index % 3 * 8)
                px = 65 + math.cos(angle) * distance
                py = 65 + math.sin(angle) * distance
                pygame.draw.circle(fx, (*CYAN_LIGHT, int(230 * (1 - burst))), (int(px), int(py)), 3)
            self.canvas.blit(fx, (end[0] - 65, end[1] - 65))
        scale = 31 + int(math.sin(raw * math.pi) * 4)
        self.draw_piece(animation.piece, (int(x), int(y)), scale)
        if raw >= 1.0:
            self.move_animation = None

    def _draw_check_effect(self, now: int) -> None:
        if self.game.winner or not self.game.is_in_check(self.game.turn):
            return
        king = self.game.king_position(self.game.turn)
        if not king:
            return
        x, y = self.display_square(*king)
        pulse = (math.sin(now / 90) + 1) / 2
        fx = pygame.Surface((110, 110), pygame.SRCALPHA)
        pygame.draw.circle(fx, (*RED_LIGHT, int(70 + pulse * 55)), (55, 55), int(37 + pulse * 8), 5)
        pygame.draw.circle(fx, (*CYAN_LIGHT, int(130 - pulse * 50)), (55, 55), int(44 + pulse * 5), 2)
        self.canvas.blit(fx, (x - 55, y - 55))

    def classify_opening(self, move: Move, piece: Piece) -> Optional[tuple[str, str, str, tuple[int, int, int]]]:
        """Recognize only a side's opening move; later development stays quiet."""
        # push() 已完成后：第 1 半回合是红方首着，第 2 半回合是黑方首着。
        # 不对后续发展手反复贴开局标签，避免“过宫炮”等播报延迟触发。
        if self.game.halfmove not in (1, 2) or move.captured is not None:
            return None
        side_name = "红方" if piece.side == RED else "黑方"
        home_row = 9 if piece.side == RED else 0
        cannon_row = 7 if piece.side == RED else 2
        pawn_row = 6 if piece.side == RED else 3
        forward = -1 if piece.side == RED else 1
        developed_horse_row = 7 if piece.side == RED else 2

        if piece.kind == PAWN and move.start[0] == pawn_row and move.start[1] in (2, 6) and move.end == (pawn_row + forward, move.start[1]):
            return "仙 人 指 路", "挺三、七兵 · 试探全局", f"{side_name}轻挺边兵，为马开路并试探对方阵型，局面保留了充分弹性。", CYAN
        if piece.kind == CANNON and move.start[0] == cannon_row and move.start[1] in (1, 7) and move.end == (cannon_row, 4):
            return "当 头 炮", "炮镇中路 · 主动出击", f"{side_name}把炮架到中路，直接施压中卒与将门，是最具进攻性的经典开局。", CYAN
        if piece.kind == CANNON and move.start[0] == cannon_row and move.start[1] == 1 and move.end == (cannon_row, 5):
            return "过 宫 炮", "横越九宫 · 转向侧翼", f"{side_name}让边炮横越中宫，迅速把火力投向另一翼，攻守转换更具隐蔽性。", CYAN
        if piece.kind == CANNON and move.start[0] == cannon_row and move.start[1] == 7 and move.end == (cannon_row, 3):
            return "过 宫 炮", "横越九宫 · 转向侧翼", f"{side_name}让边炮横越中宫，迅速把火力投向另一翼，攻守转换更具隐蔽性。", CYAN
        if piece.kind == HORSE and move.start[0] == home_row and move.start[1] in (1, 7) and move.end[0] == developed_horse_row and move.end[1] in (2, 6):
            horses_ready = all(self.game.board[developed_horse_row][col] == Piece(piece.side, HORSE) for col in (2, 6))
            if horses_ready:
                return "屏 风 马", "双马成屏 · 稳守待发", f"{side_name}双马拱卫中兵，屏风马阵型完成，后续可稳守亦可从两翼反击。", CYAN
            return "起 马 局", "正马出阵 · 先固后攻", f"{side_name}先起正马，优先完成子力协调，为中路和侧翼的发展留出空间。", CYAN
        if piece.kind == ELEPHANT and move.start[0] == home_row and move.start[1] in (2, 6) and move.end == (developed_horse_row, 4):
            return "飞 相 局", "先固阵型 · 后发制人", f"{side_name}飞相巩固中路防线，属于重视阵型完整与反击弹性的经典布局。", CYAN
        if piece.kind == CANNON and piece.side == BLACK and move.start == (2, 1) and move.end == (2, 2):
            return "卒 底 炮", "借卒成架 · 针锋相对", "黑方以卒底炮回应，快速制造炮架并对红方边兵体系施加反击压力。", CYAN
        if piece.kind == CANNON and piece.side == BLACK and move.start == (2, 7) and move.end == (2, 6):
            return "卒 底 炮", "借卒成架 · 针锋相对", "黑方以卒底炮回应，快速制造炮架并对红方边兵体系施加反击压力。", CYAN
        return None

    def classify_tactic(self, move: Move, piece: Piece) -> tuple[str, str, str, tuple[int, int, int]]:
        side_name = "红方" if piece.side == RED else "黑方"
        capture_name = move.captured.name if move.captured else ""
        if self.game.winner:
            own_rooks = sum(item is not None and item.side == piece.side and item.kind == ROOK for row in self.game.board for item in row)
            own_cannons = sum(item is not None and item.side == piece.side and item.kind == CANNON for row in self.game.board for item in row)
            if piece.kind == HORSE and move.end[0] in (2, 7) and move.end[1] in (3, 5):
                title, subtitle = "卧槽马绝杀", "战马踏入宫前死角 · 封死将门"
            elif piece.kind == CANNON and own_cannons >= 2:
                title, subtitle = "重炮绝杀", "双炮叠阵 · 中轴火力完全贯穿"
            elif piece.kind == ROOK and own_rooks >= 2:
                title, subtitle = "双车绝杀", "双车错位封锁 · 无路可退"
            else:
                title, subtitle = "绝  杀", "退路尽封 · 命运终局"
            return title, subtitle, f"{side_name}完成致命一击，敌方将帅已无任何合法应对。", RED_LIGHT

        attacked = []
        for attack in self.game.pseudo_moves_from(*move.end):
            target = attack.captured
            if target and target.side != piece.side and target.kind in (KING, ROOK, CANNON, HORSE):
                attacked.append(target)
        if len(attacked) >= 2:
            targets = "、".join(target.name for target in attacked[:2])
            return "捉  双", f"单子双袭 · 同时锁定 {targets}", f"{piece.name}落位后同时攻击两个高价值目标，对手只能保存其一。", GOLD

        if self.game.is_in_check(self.game.turn):
            if piece.kind == HORSE and move.end[0] in (2, 7) and move.end[1] in (3, 5):
                return "卧 槽 马", "战马压宫 · 将门告急", "马跃卧槽点形成近身将军，并压缩了将帅在九宫内的活动空间。", RED_LIGHT
            if piece.kind == CANNON:
                return "中炮将军", "炮火贯穿中轴", "炮借架发力直击将帅，对方必须立刻处理这条攻击线路。", RED_LIGHT
            return "将  军", "KING UNDER ATTACK", f"{piece.name}形成直接将军，局面节奏被{side_name}强制接管。", RED_LIGHT

        opening = self.classify_opening(move, piece)
        if opening:
            return opening
        if move.captured and move.captured.kind in (ROOK, CANNON, HORSE):
            return "关键击破", f"{piece.name} × {capture_name}", f"{side_name}{piece.name}拿下高价值的{capture_name}，局面天平明显倾斜。", GOLD
        if move.captured:
            return "", "", f"{side_name}完成一次常规交换，棋局继续推进。", CYAN
        return "", "", f"{side_name}{piece.name}完成调动，正在重构棋盘上的攻防线路。", CYAN

    def trigger_tactical_callout(self, move: Move, piece: Piece) -> None:
        title, subtitle, commentary, color = self.classify_tactic(move, piece)
        if self.skin == "rift":
            hero = self.character_name(piece)
            if move.captured and title == "关键击破":
                victim = self.character_name(move.captured)
                subtitle = f"{hero} 击破 {victim}"
                commentary = f"{hero}拿下关键目标，{victim}退出战场。"
            elif title:
                commentary = f"{hero}触发『{title.replace(' ', '')}』：{commentary}"
        self.commentary = commentary
        if not title:
            self.callout = None
            return
        event_key = self.tactic_key(title)
        urgent = event_key in ("mate", "check", "fork")
        if not urgent and self.game.halfmove - getattr(self, "last_callout_halfmove", -10) < 3:
            self.callout = None
            return
        self.last_callout_halfmove = self.game.halfmove
        duration = 3600 if event_key == "mate" else 2800
        self.callout = TacticalCallout(
            title, subtitle, commentary, color, pygame.time.get_ticks(),
            actor=piece, victim=move.captured, duration=duration,
        )
        if event_key == "mate":
            self.flash_until = pygame.time.get_ticks() + 520
            self.shake_until = pygame.time.get_ticks() + 650

    def draw_tactical_callout(self) -> None:
        event = self.callout
        if not event:
            return
        now = pygame.time.get_ticks()
        progress = event.progress(now)
        if progress >= 1.0:
            self.callout = None
            return
        enter = min(1.0, progress / 0.16)
        exit_progress = max(0.0, (progress - 0.80) / 0.20)
        width, height = 690, 225
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        # 关键播报卡片必须完全实底，避免棋盘网格在任何退出帧透入内容区。
        panel_fill = (5, 25, 31, 255) if self.skin == "rift" else (31, 17, 61, 255)
        inner_line = (224, 190, 105, 100) if self.skin == "rift" else (255, 220, 247, 85)
        pygame.draw.rect(panel, panel_fill, panel.get_rect(), border_radius=28)
        pygame.draw.rect(panel, (*event.color, 225), panel.get_rect(), 3, border_radius=28)
        pygame.draw.rect(panel, inner_line, panel.get_rect().inflate(-12, -12), 1, border_radius=22)
        for index in range(18):
            x = (index * 47 + now // 25) % width
            y = 24 + (index * 31) % (height - 48)
            if self.skin == "rift":
                pygame.draw.polygon(panel, (104, 202, 238, 45), [(x, y - 5), (x + 4, y), (x, y + 5), (x - 4, y)])
            else:
                pygame.draw.ellipse(panel, (255, 184, 224, 40), (x, y, 10, 5))

        actor = self.character_portrait(event.actor, 176)
        if actor is not None:
            panel.blit(actor, (16, 24))
            pygame.draw.circle(panel, event.color, (104, 112), 87, 3)
        text_center = 430 if event.victim else 448
        event_label = "英雄时刻" if self.skin == "rift" else "必杀演出"
        label = self.fonts["tiny"].render(f"{self.character_name(event.actor)} · {event_label}", True, event.color)
        panel.blit(label, label.get_rect(center=(text_center, 31)))
        title = self.fonts["callout"].render(event.title, True, (255, 249, 255))
        shadow = self.fonts["callout"].render(event.title, True, (95, 32, 116))
        panel.blit(shadow, shadow.get_rect(center=(text_center + 4, 99 + 5)))
        panel.blit(title, title.get_rect(center=(text_center, 99)))
        subtitle = self.fonts["callout_small"].render(event.subtitle, True, GOLD_LIGHT)
        panel.blit(subtitle, subtitle.get_rect(center=(text_center, 154)))
        actor_line = self.fonts["small"].render(
            f"{self.character_name(event.actor)}：『{self.character_quote(event.actor, event.title)}』", True,
            (223, 197, 133) if self.skin == "rift" else (255, 207, 235),
        )
        panel.blit(actor_line, actor_line.get_rect(center=(text_center, 188)))

        if event.victim:
            victim = self.character_portrait(event.victim, 105)
            if victim is not None:
                victim.set_alpha(int(210 * max(0.2, 1 - progress)))
                panel.blit(victim, (565, 87))
                pygame.draw.circle(panel, (161, 138, 181), (617, 139), 51, 2)
            cross = int(180 * min(1.0, progress * 3))
            pygame.draw.line(panel, (255, 102, 153, cross), (580, 103), (656, 178), 5)
            pygame.draw.line(panel, (255, 102, 153, cross), (656, 103), (580, 178), 5)

        # 退出时保持卡片不透明并缩小上移；不使用整体透明度，防止棋盘线透到播报内容上。
        exit_scale = 1.0 - 0.18 * ease_out_cubic(exit_progress)
        scale = (0.82 + ease_out_back(enter) * 0.18) * exit_scale
        panel = pygame.transform.smoothscale(panel, (int(width * scale), int(height * scale)))
        exit_y = int(18 * ease_out_cubic(exit_progress))
        self.canvas.blit(panel, panel.get_rect(center=(394, 421 - exit_y)))
        if now < self.flash_until:
            flash = pygame.Surface((VW, VH), pygame.SRCALPHA)
            flash.fill((*event.color, int(70 * (self.flash_until - now) / 520)))
            self.canvas.blit(flash, (0, 0))

    def draw_victory_splash(self) -> None:
        if not self.game.winner:
            return
        art = self.victory_art.get(self.game.winner)
        if art is None:
            return
        overlay = pygame.Surface((VW, VH), pygame.SRCALPHA)
        overlay.fill((3, 14, 20, 155) if self.skin == "rift" else (20, 7, 42, 145))
        self.canvas.blit(overlay, (0, 0))
        card = pygame.Rect(170, 82, 1100, 720)
        image = pygame.transform.smoothscale(art, card.size)
        self.canvas.blit(image, card)
        shade = pygame.Surface(card.size, pygame.SRCALPHA)
        for y in range(card.h):
            alpha = int(205 * max(0, (y - card.h * 0.48) / (card.h * 0.52)))
            pygame.draw.line(shade, (28, 8, 48, alpha), (0, y), (card.w, y))
        self.canvas.blit(shade, card)
        color = RED_LIGHT if self.game.winner == RED else CYAN
        pygame.draw.rect(self.canvas, color, card, 4, border_radius=22)
        self.draw_text("VICTORY", self.fonts["callout_small"], GOLD_LIGHT, (VW // 2, 630), center=True)
        if self.skin == "rift":
            victory_title = "蓝 方 胜 利" if self.game.winner == RED else "红 方 胜 利"
            winner = "蓝方英雄联盟" if self.game.winner == RED else "红方英雄联盟"
        else:
            victory_title = "绯 樱 · 胜 局" if self.game.winner == RED else "苍 月 · 胜 局"
            winner = "绯樱阵营" if self.game.winner == RED else "苍月阵营"
        self.draw_text(victory_title, self.fonts["callout"], TEXT, (VW // 2, 690), center=True)
        self.draw_text(f"{winner}获得胜利 · {self.game.reason}", self.fonts["body"], color, (VW // 2, 748), center=True)
        self.draw_text("按 R 再次出阵", self.fonts["small"], TEXT, (VW // 2, 778), center=True)

    def draw_header(self) -> None:
        if self.skin == "rift":
            title, subtitle, crest_text = "峡谷棋阵", "RUNETERRA XIANGQI · 英雄对弈", "峡"
            accent, crest_fill = GOLD, (21, 67, 76)
        else:
            title, subtitle, crest_text = "棋灵物语", "KIRIN MONOGATARI · 棋娘对弈", "姬"
            accent, crest_fill = RED_LIGHT, (76, 43, 112)
        self.draw_text(title, self.fonts["hero"], TEXT, (805, 34))
        self.draw_text(subtitle, self.fonts["small"], accent, (808, 89))
        crest = pygame.Rect(1295, 35, 64, 60)
        pygame.draw.rect(self.canvas, crest_fill, crest, border_radius=18)
        pygame.draw.rect(self.canvas, GOLD_LIGHT if self.skin == "rift" else (242, 178, 231), crest, 2, border_radius=18)
        pygame.draw.circle(self.canvas, (220, 238, 220) if self.skin == "rift" else (255, 213, 237), crest.center, 22)
        self.draw_text(crest_text, self.fonts["title"], (30, 92, 104) if self.skin == "rift" else RED_C, crest.center, center=True)
        pygame.draw.line(self.canvas, (116, 122, 93) if self.skin == "rift" else (152, 112, 181), (805, 119), (1385, 119), 1)

    def draw_sidebar(self) -> None:
        self.buttons.clear()
        self.palette.clear()
        self.draw_header()
        side_name = "红方" if self.game.turn == RED else "黑方"
        status_color = RED_LIGHT if self.game.turn == RED else CYAN_LIGHT
        hero = pygame.Rect(805, 137, 580, 103)
        hero_color = (15, 43, 49) if self.skin == "rift" else (48, 31, 76)
        self.draw_glass(hero, hero_color, 240, 18)
        focus_piece = self.game.piece_at(self.selected) if self.selected else None
        focus_piece = focus_piece or Piece(self.game.turn, KING)
        portrait = self.character_portrait(focus_piece, 78)
        if portrait is not None:
            self.canvas.blit(portrait, portrait.get_rect(center=(852, 188)))
        pygame.draw.circle(self.canvas, RED_LIGHT if focus_piece.side == RED else CYAN, (852, 188), 39, 3)
        prefix = "当前选择" if self.selected else ("当前英雄" if self.skin == "rift" else "当前出阵")
        current_label = f"{prefix} · {self.character_name(focus_piece)}"
        self.draw_text(current_label, self.fonts["tiny"], MUTED, (899, 157))
        headline = f"{side_name}行棋"
        if self.game.winner:
            headline = ("红方" if self.game.winner == RED else "黑方") + "胜"
        elif self.mode == "editor":
            headline = "自定义棋局"
        self.draw_text(headline, self.fonts["title"], status_color, (898, 178))
        if self.mode == "editor":
            badge = "布阵中"
        elif self.skin == "rift":
            badge = "峡谷挑战" if self.opponent_mode == "ai" else "英雄对弈"
        else:
            badge = "棋娘挑战" if self.opponent_mode == "ai" else "好友对弈"
        badge_rect = pygame.Rect(1245, 170, 115, 34)
        badge_fill = (20, 74, 82) if self.skin == "rift" else (91, 58, 126)
        badge_border = GOLD if self.skin == "rift" else (231, 181, 250)
        pygame.draw.rect(self.canvas, badge_fill, badge_rect, border_radius=17)
        pygame.draw.rect(self.canvas, badge_border, badge_rect, 1, border_radius=17)
        self.draw_text(badge, self.fonts["small"], TEXT, badge_rect.center, center=True)

        y = 260
        self.add_button("new", 805, y, 132, "新对局", accent=True)
        self.add_button("undo", 949, y, 132, "悔棋")
        self.add_button("flip", 1093, y, 132, "翻转棋盘", active=self.flipped)
        self.add_button("editor", 1237, y, 148, "局面编辑", active=self.mode == "editor")
        y += 56
        self.add_button("mode", 805, y, 142, "模式 · " + ("人机" if self.opponent_mode == "ai" else "双人"), active=self.opponent_mode == "ai")
        self.add_button("side", 959, y, 128, "执子 · " + ("红" if self.human_side == RED else "黑"))
        self.add_button("depth", 1099, y, 132, f"棋力 · {'萌新' if self.ai.depth == 1 else '进阶' if self.ai.depth == 2 else '高手'}")
        skin_label = "皮肤 · 峡谷" if self.skin == "rift" else "皮肤 · 棋娘"
        self.add_button("skin", 1243, y, 142, skin_label, active=self.skin == "rift")

        if self.mode == "editor":
            self.draw_editor_panel(805, 382)
        else:
            self.draw_play_panel(805, 382)
        self.draw_status_bar()

    def draw_play_panel(self, x: int, y: int) -> None:
        players = pygame.Rect(x, y, 580, 78)
        self.draw_glass(players, (14, 43, 49) if self.skin == "rift" else (50, 32, 79), 245, 16)
        for index, side in enumerate((RED, BLACK)):
            px = x + 20 + index * 286
            active = self.game.turn == side
            team_color = RED_LIGHT if side == RED else CYAN
            portrait = self.character_portrait(Piece(side, KING), 64)
            if portrait is not None:
                self.canvas.blit(portrait, portrait.get_rect(center=(px + 31, y + 39)))
            pygame.draw.circle(self.canvas, team_color if active else (124, 99, 151), (px + 31, y + 39), 31, 2)
            if self.skin == "rift":
                label = RIFT_CHARACTER_NAMES[(side, KING)]
            else:
                label = "绯樱主将" if side == RED else "苍月主将"
            if side != self.human_side and self.opponent_mode == "ai":
                label += " · AI"
            self.draw_text(label, self.fonts["button"], TEXT, (px + 72, y + 18))
            self.draw_text("正在出阵" if active else "等待回合", self.fonts["tiny"], team_color if active else MUTED, (px + 72, y + 45))

        commentary_rect = pygame.Rect(x, y + 92, 580, 94)
        self.draw_glass(commentary_rect, (12, 38, 45) if self.skin == "rift" else (48, 30, 76), 246, 16)
        commentary_title = "峡谷战术台" if self.skin == "rift" else "棋娘解说"
        self.draw_text(commentary_title, self.fonts["tiny"], GOLD if self.skin == "rift" else RED_LIGHT, (x + 18, y + 106))
        pygame.draw.line(self.canvas, CYAN_DIM, (x + 172, y + 115), (x + 558, y + 115), 1)
        commentary_lines = [self.commentary[index:index + 31] for index in range(0, min(len(self.commentary), 62), 31)]
        for index, line in enumerate(commentary_lines[:2]):
            self.draw_text(line, self.fonts["small"], TEXT, (x + 18, y + 132 + index * 25))

        log_rect = pygame.Rect(x, y + 200, 580, 150)
        self.draw_glass(log_rect, (9, 31, 38) if self.skin == "rift" else (35, 24, 61), 252, 16)
        self.draw_text("对局棋谱", self.fonts["button"], TEXT, (x + 18, y + 214))
        self.draw_text(f"第 {self.game.halfmove // 2 + 1:02d} 回合", self.fonts["tiny"], RED_LIGHT, (x + 468, y + 217))
        pygame.draw.line(self.canvas, (124, 87, 151), (x + 18, y + 244), (x + 560, y + 244), 1)
        if not self.move_log:
            self.draw_text("等待行动数据…", self.fonts["small"], MUTED, (x + 20, y + 270))
        else:
            recent = self.move_log[-8:]
            for index in range(0, len(recent), 2):
                number = max(1, (len(self.move_log) - len(recent) + index) // 2 + 1)
                row_y = y + 253 + index // 2 * 24
                row_rect = pygame.Rect(x + 14, row_y - 2, 550, 22)
                pygame.draw.rect(self.canvas, (66, 43, 91) if index // 2 % 2 == 0 else (48, 32, 73), row_rect, border_radius=6)
                self.draw_text(f"{number:02d}", self.fonts["tiny"], CYAN, (x + 22, row_y))
                self.draw_text(recent[index], self.fonts["small"], (239, 250, 252), (x + 72, row_y - 4))
                if index + 1 < len(recent):
                    self.draw_text(recent[index + 1], self.fonts["small"], (239, 250, 252), (x + 298, row_y - 4))

    def draw_editor_panel(self, x: int, y: int) -> None:
        panel = pygame.Rect(x, y, 580, 365)
        self.draw_glass(panel, (24, 32, 33), 225, 15)
        self.draw_text("棋子工具箱", self.fonts["button"], TEXT, (x + 20, y + 17))
        self.draw_text("选择棋子放置 · 右键擦除", self.fonts["tiny"], MUTED, (x + 396, y + 20))
        for side_index, side in enumerate((RED, BLACK)):
            py = y + 94 + side_index * 82
            self.draw_text("红" if side == RED else "黑", self.fonts["button"], RED_LIGHT if side == RED else TEXT, (x + 20, py - 12))
            for index, kind in enumerate(KINDS):
                rect = pygame.Rect(x + 70 + index * 68, py - 30, 56, 56)
                piece = Piece(side, kind)
                self.palette.append((rect, piece))
                if piece == self.editor_piece:
                    pygame.draw.rect(self.canvas, JADE, rect.inflate(6, 6), 3, border_radius=14)
                self.draw_piece(piece, rect.center, 24)
        by = y + 245
        self.add_button("standard", x + 20, by, 126, "标准布局")
        self.add_button("clear", x + 158, by, 100, "清空")
        self.add_button("turn", x + 270, by, 134, "先手 · " + ("红" if self.game.turn == RED else "黑"))
        self.add_button("start", x + 416, by, 144, "开始对局", accent=True)
        by += 55
        self.add_button("import", x + 20, by, 126, "导入 FEN")
        self.add_button("export", x + 158, by, 126, "复制 FEN")
        self.draw_text("支持常用中国象棋 FEN", self.fonts["tiny"], MUTED, (x + 312, by + 13))

    def draw_status_bar(self) -> None:
        rect = pygame.Rect(805, 766, 580, 88)
        self.draw_glass(rect, (24, 32, 33), 235, 14)
        color = RED_LIGHT if "将军" in self.message or "获胜" in self.message else GOLD
        pygame.draw.circle(self.canvas, color, (832, 810), 6)
        message = self.message
        if "思考" in message:
            message = "AI 正在思考" + "." * (1 + pygame.time.get_ticks() // 350 % 3)
        self.draw_text(message[:38], self.fonts["body"], TEXT, (851, 790))
        self.draw_text("F 翻转   U 悔棋   R 新局   S 换肤", self.fonts["tiny"], MUTED, (851, 824))

    def draw_modal(self) -> None:
        shade = pygame.Surface((VW, VH), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 188))
        self.canvas.blit(shade, (0, 0))
        rect = pygame.Rect(250, 310, 940, 230)
        self.draw_glass(rect, (19, 27, 29), 248, 20)
        pygame.draw.rect(self.canvas, GOLD, rect, 2, border_radius=20)
        self.draw_text("导入自定义棋局", self.fonts["title"], TEXT, (282, 339))
        self.draw_text("粘贴 10 行棋盘布局及先手标记（w 红 / b 黑）", self.fonts["small"], MUTED, (282, 383))
        field = pygame.Rect(282, 423, 876, 50)
        pygame.draw.rect(self.canvas, (8, 14, 16), field, border_radius=9)
        pygame.draw.rect(self.canvas, JADE, field, 2, border_radius=9)
        self.draw_text(self.text_input[-82:] + "|", self.fonts["small"], TEXT, (297, 438))
        self.draw_text("Enter 导入    Esc 取消    ⌘/Ctrl + V 粘贴", self.fonts["small"], GOLD, (282, 493))

    def draw(self) -> None:
        self.draw_background()
        shake_x = 0
        if pygame.time.get_ticks() < self.shake_until:
            shake_x = int(math.sin(pygame.time.get_ticks() * 0.12) * 3)
        if shake_x:
            board_layer = self.canvas.copy()
            self.draw_board()
            board_crop = self.canvas.subsurface((0, 0, 775, VH)).copy()
            self.canvas.blit(board_layer, (0, 0))
            self.canvas.blit(board_crop, (shake_x, 0))
        else:
            self.draw_board()
        self.draw_sidebar()
        self.draw_tactical_callout()
        if self.game.winner and self.callout is None:
            self.draw_victory_splash()
        if self.modal:
            self.draw_modal()
        width, height = self.window.get_size()
        scale = min(width / VW, height / VH)
        target = pygame.transform.smoothscale(self.canvas, (int(VW * scale), int(VH * scale)))
        self.window.fill((3, 7, 8))
        self.window.blit(target, ((width - target.get_width()) // 2, (height - target.get_height()) // 2))
        pygame.display.flip()

    def new_game(self) -> None:
        self.cancel_ai()
        self.game.reset()
        self.mode = "play"
        self.selected = None
        self.legal_moves = []
        self.move_log.clear()
        self.move_animation = None
        self.pending_board_click = None
        self.callout = None
        self.last_callout_halfmove = -10
        if self.skin == "rift":
            self.commentary = "欢迎来到峡谷棋阵。嘉文四世率领蓝方英雄先手出击。"
        else:
            self.commentary = "新一幕棋灵物语开始啦！绯樱阵营率先出阵。"
        self.flipped = self.human_side == BLACK
        self.message = "新对局开始 · 红方先行"

    def cancel_ai(self) -> None:
        self.ai_future = None
        self.ai_position = ""

    def play_move(self, move: Move) -> None:
        piece = self.game.piece_at(move.start)
        if piece is None:
            return
        self.game.push(move)
        text = self.game.move_text(move) or piece.name
        self.move_log.append(text)
        self.trigger_tactical_callout(move, piece)
        self.move_animation = MoveAnimation(move, piece, pygame.time.get_ticks())
        self.selected = None
        self.legal_moves = []
        if move.captured:
            self.shake_until = pygame.time.get_ticks() + 240
        if self.game.winner:
            winner = "红方" if self.game.winner == RED else "黑方"
            self.message = f"{winner}获胜 · {self.game.reason}"
        elif self.game.is_in_check(self.game.turn):
            self.message = "将军！请谨慎应对"
        else:
            self.message = ("红方" if self.game.turn == RED else "黑方") + "行棋"

    def handle_board_click(self, square: tuple[int, int], button: int) -> None:
        if self.move_animation:
            self.pending_board_click = (square, button)
            self.message = "已记录本次点击，落子动画结束后立即执行。"
            return
        if self.mode == "editor":
            if button == 3:
                self.game.set_piece(*square, None)
            elif button == 1 and self.editor_piece:
                self.game.set_piece(*square, self.editor_piece)
            return
        if button != 1 or self.game.winner:
            return
        if self.opponent_mode == "ai" and self.game.turn != self.human_side:
            return
        if self.selected:
            chosen = next((move for move in self.legal_moves if move.end == square), None)
            if chosen:
                self.play_move(chosen)
                return
        piece = self.game.piece_at(square)
        if piece and piece.side == self.game.turn:
            self.selected = square
            self.legal_moves = self.game.legal_moves_from(square)
            self.message = f"已选择 {piece.name} · {len(self.legal_moves)} 个合法着法"
        else:
            self.selected = None
            self.legal_moves = []

    def handle_button(self, key: str) -> None:
        if key == "new":
            self.new_game()
        elif key == "undo":
            self.cancel_ai()
            self.move_animation = None
            self.pending_board_click = None
            undone = self.game.undo()
            if self.opponent_mode == "ai" and undone and self.game.turn != self.human_side:
                self.game.undo()
                if self.move_log:
                    self.move_log.pop()
            if undone and self.move_log:
                self.move_log.pop()
            self.selected, self.legal_moves = None, []
            self.message = "已撤销上一回合" if undone else "没有可撤销的着法"
        elif key == "flip":
            self.flipped = not self.flipped
        elif key == "editor":
            self.cancel_ai()
            self.move_animation = None
            self.pending_board_click = None
            self.mode = "play" if self.mode == "editor" else "editor"
            self.selected, self.legal_moves = None, []
            self.message = "局面编辑已开启" if self.mode == "editor" else "返回对局"
        elif key == "mode":
            self.opponent_mode = "local" if self.opponent_mode == "ai" else "ai"
            self.message = "已切换为双人对弈" if self.opponent_mode == "local" else "已切换为人机对弈"
        elif key == "side":
            self.human_side = BLACK if self.human_side == RED else RED
            self.new_game()
        elif key == "depth":
            self.ai.depth = 1 if self.ai.depth >= 3 else self.ai.depth + 1
            self.message = f"AI 棋力已设为 {'入门' if self.ai.depth == 1 else '进阶' if self.ai.depth == 2 else '高手'}"
        elif key == "skin":
            next_skin = "rift" if self.skin == "anime" else "anime"
            self.activate_skin(next_skin)
            self.callout = None
            if next_skin == "rift":
                self.commentary = "英雄已降临棋盘。召唤师峡谷阵容接管本局视觉。"
                self.message = "已切换：峡谷英雄皮肤"
            else:
                self.commentary = "棋娘们重新登场，樱花舞台已经准备完毕。"
                self.message = "已切换：日系棋娘皮肤"
        elif key == "standard":
            turn = self.game.turn
            self.game.reset()
            self.game.turn = turn
            self.move_log.clear()
            self.message = "已恢复标准布局"
        elif key == "clear":
            self.game.clear()
            self.message = "棋盘已清空"
        elif key == "turn":
            self.game.turn = BLACK if self.game.turn == RED else RED
            self.message = ("红方" if self.game.turn == RED else "黑方") + "先行"
        elif key == "start":
            errors = self.game.validate_position()
            if errors:
                self.message = errors[0]
            else:
                self.mode = "play"
                self.game.history.clear()
                self.game.winner = None
                self.game.reason = ""
                self.game.halfmove = 0
                self.move_log.clear()
                self.message = "自定义棋局开始"
        elif key == "import":
            self.modal = "fen"
            self.text_input = ""
            pygame.key.start_text_input()
        elif key == "export":
            try:
                pygame.scrap.put(pygame.SCRAP_TEXT, self.game.to_fen().encode("utf-8") + b"\x00")
                self.message = "FEN 已复制到剪贴板"
            except pygame.error:
                self.message = self.game.to_fen()

    def handle_modal_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.modal = ""
                pygame.key.stop_text_input()
            elif event.key == pygame.K_RETURN:
                try:
                    self.game = XiangqiGame.from_fen(self.text_input)
                    self.mode = "editor"
                    self.modal = ""
                    self.move_log.clear()
                    self.message = "FEN 导入成功，可继续编辑或开始对局"
                    pygame.key.stop_text_input()
                except ValueError as exc:
                    self.message = f"FEN 无效：{exc}"
            elif event.key == pygame.K_BACKSPACE:
                self.text_input = self.text_input[:-1]
            elif event.key == pygame.K_v and (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)):
                try:
                    raw = pygame.scrap.get(pygame.SCRAP_TEXT)
                    if raw:
                        self.text_input += raw.rstrip(b"\x00").decode("utf-8")
                except (pygame.error, UnicodeDecodeError):
                    pass
        elif event.type == pygame.TEXTINPUT:
            self.text_input += event.text

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.modal:
            self.handle_modal_event(event)
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_f:
                self.flipped = not self.flipped
            elif event.key == pygame.K_r:
                self.new_game()
            elif event.key == pygame.K_u:
                self.handle_button("undo")
            elif event.key == pygame.K_s:
                self.handle_button("skin")
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = self.virtual_mouse(event.pos)
            for button in self.buttons:
                if button.rect.collidepoint(pos):
                    self.handle_button(button.key)
                    return
            if self.mode == "editor":
                for rect, piece in self.palette:
                    if rect.collidepoint(pos):
                        self.editor_piece = piece
                        self.message = f"画笔：{'红' if piece.side == RED else '黑'}方{piece.name}"
                        return
            square = self.board_square(pos)
            if square:
                self.handle_board_click(square, event.button)

    def process_pending_click(self) -> None:
        if self.move_animation and self.move_animation.progress(pygame.time.get_ticks()) >= 1.0:
            self.move_animation = None
        if self.move_animation is not None or self.pending_board_click is None:
            return
        square, button = self.pending_board_click
        self.pending_board_click = None
        self.handle_board_click(square, button)

    def update_ai(self) -> None:
        if self.move_animation or self.mode != "play" or self.opponent_mode != "ai" or self.game.winner or self.game.turn == self.human_side:
            return
        if self.ai_future is None:
            clone = self.game.clone()
            self.ai_position = self.game.to_fen()
            self.ai_future = self.executor.submit(self.ai.choose_move, clone)
            self.message = "AI 正在思考"
            return
        if self.ai_future.done():
            future, self.ai_future = self.ai_future, None
            try:
                move = future.result()
                if self.ai.load_error:
                    self.message = f"PPO 模型：{self.ai.load_error}"
                elif move and self.ai_position == self.game.to_fen():
                    self.play_move(move)
            except Exception as exc:
                self.message = f"AI 计算失败：{exc}"

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.process_pending_click()
            self.update_ai()
            self.draw()
            self.clock.tick(60)
        self.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()


def main() -> None:
    XiangqiApp().run()


if __name__ == "__main__":
    main()
