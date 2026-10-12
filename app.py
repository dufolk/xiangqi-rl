"""2.5D wooden-board Pygame interface for Chinese chess, where every move plays out as a little scene."""

from __future__ import annotations

import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "0")

import pygame

from ai_player import PPOAI
from cinema.demos import catalogue
from cinema.detect import SITUATIONS, StoryEvent, analyze_move, material_balance, situation
from cinema.fonts import load_font
from cinema.fx import seg
from cinema.show import MoveShow, draw_ribbon
from cinema.view3d import VIEW_W, BoardView, Frame, Sprite
from game_engine import KING, KINDS, RED, BLACK, ROOK, HORSE, CANNON, Move, Piece, XiangqiGame

VW, VH = 1440, 900
SIDE_X, SIDE_W = 880, 530

TEXT = (255, 242, 218)
MUTED = (214, 184, 140)
GOLD = (232, 186, 100)
GOLD_LIGHT = (255, 226, 160)
LACQUER = (52, 28, 18)
LACQUER_2 = (74, 40, 24)
RED_T = (232, 86, 64)
BLACK_T = (150, 196, 200)
JADE = (130, 214, 170)
LEGAL = (255, 232, 140)
CAPTURE = (255, 86, 60)

STORY_MODES = ("full", "highlights", "off")
STORY_LABELS = {"full": "全程", "highlights": "精彩", "off": "关闭"}
MODE_LABELS = {"ai": "人机", "local": "双人", "watch": "观战"}
CATEGORY_LABELS = {"opening": "开局", "capture": "吃子", "tone": "局势", "tactic": "战术", "mate": "杀法", "stalemate": "困毙"}
GALLERY_LEAD_MS = 700
GALLERY_LINGER_MS = 1400


@dataclass
class UIButton:
    key: str
    rect: pygame.Rect
    label: str
    active: bool = False
    accent: bool = False


def side_name(side: str) -> str:
    return "红方" if side == RED else "黑方"


class XiangqiApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("木韵棋院 · 2.5D 中国象棋")
        self.window = pygame.display.set_mode((VW, VH), pygame.RESIZABLE)
        self.canvas = pygame.Surface((VW, VH)).convert()
        self.clock = pygame.time.Clock()
        self.fonts = {
            "tiny": load_font(14), "small": load_font(17), "body": load_font(20),
            "button": load_font(17, True), "title": load_font(30, True), "hero": load_font(42, True),
            "callout": load_font(26, True),
        }
        self.view = BoardView()
        self.panel_cache: dict[tuple, pygame.Surface] = {}
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
        self.selected_at = 0
        self.legal_moves: list[Move] = []
        self.editor_piece: Optional[Piece] = Piece(RED, ROOK)
        self.buttons: list[UIButton] = []
        self.palette: list[tuple[pygame.Rect, Piece]] = []
        self.move_log: list[str] = []
        self.message = "红方先行 · 请落子"
        self.modal = ""
        self.text_input = ""
        self.pending_board_click: Optional[tuple[tuple[int, int], int]] = None
        self.shows: list[MoveShow] = []
        self.story_mode = "full"
        self.threats: list[tuple[int, int]] = []
        self.threats_until = 0
        self.ai_ready_at = 0
        self.ai_error_shown = ""
        self.mood = 0.0
        self.last_tick = pygame.time.get_ticks()
        self.game_over_at = 0
        self.gallery_index: Optional[int] = None
        self.gallery_saved: Optional[tuple] = None
        self.gallery_due = 0
        self.gallery_played = False
        self.gallery_done_at = 0
        self.commentary = "木棋已摆好，棋子们正打着瞌睡，等你落下第一步。"
        self.running = True
        try:
            pygame.scrap.init()
        except pygame.error:
            pass

    # --- geometry ----------------------------------------------------------------
    def virtual_mouse(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Map OS-window mouse coordinates to the fixed 1440×900 canvas (handles HiDPI framebuffers)."""
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

    def board_square(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        if pos[0] >= VIEW_W:
            return None
        return self.view.pick(pos)

    # --- widgets -----------------------------------------------------------------
    def draw_text(self, text: str, font: pygame.font.Font, color, pos, center: bool = False) -> None:
        surface = font.render(text, True, color)
        self.canvas.blit(surface, surface.get_rect(center=pos) if center else pos)

    def draw_panel(self, rect: pygame.Rect, tint=LACQUER, alpha: int = 228) -> None:
        key = (rect.size, tint, alpha)
        panel = self.panel_cache.get(key)
        if panel is None:
            panel = pygame.Surface((rect.w + 10, rect.h + 12), pygame.SRCALPHA)
            pygame.draw.rect(panel, (12, 6, 2, 110), pygame.Rect(4, 7, rect.w, rect.h), border_radius=16)
            body = pygame.Rect(0, 0, rect.w, rect.h)
            pygame.draw.rect(panel, (*tint, alpha), body, border_radius=16)
            for k in range(0, rect.h, 6):
                pygame.draw.line(panel, (255, 220, 170, 6 + (k * 7) % 9), (14, k), (rect.w - 14, k + 2), 1)
            pygame.draw.rect(panel, (*GOLD, 200), body, 2, border_radius=16)
            pygame.draw.rect(panel, (255, 236, 190, 40), body.inflate(-8, -8), 1, border_radius=12)
            self.panel_cache[key] = panel
        self.canvas.blit(panel, rect.topleft)

    def add_button(self, key: str, x: int, y: int, w: int, label: str, active: bool = False, accent: bool = False) -> None:
        button = UIButton(key, pygame.Rect(x, y, w, 42), label, active, accent)
        self.buttons.append(button)
        hovered = button.rect.collidepoint(self.virtual_mouse(pygame.mouse.get_pos())) and not self.modal
        if accent:
            fill, border = ((204, 70, 48) if hovered else (176, 52, 36)), GOLD_LIGHT
        elif active:
            fill, border = ((120, 76, 40) if hovered else (98, 60, 30)), GOLD
        else:
            fill, border = ((92, 54, 30) if hovered else (68, 38, 22)), ((226, 182, 110) if hovered else (150, 108, 62))
        pygame.draw.rect(self.canvas, (20, 10, 4), button.rect.move(0, 4), border_radius=12)
        pygame.draw.rect(self.canvas, fill, button.rect, border_radius=12)
        pygame.draw.rect(self.canvas, border, button.rect, 2, border_radius=12)
        pygame.draw.line(self.canvas, (255, 236, 200), (x + 14, y + 3), (x + w - 14, y + 3), 1)
        self.draw_text(label, self.fonts["button"], TEXT, button.rect.center, center=True)

    # --- board scene -------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return bool(self.shows)

    def current_show(self) -> Optional[MoveShow]:
        return self.shows[0] if self.shows else None

    def shown_board(self):
        show = self.current_show()
        return show.after if show is not None else self.game.board

    def ground_marks(self, now: int):
        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            pulse = 0.5 + 0.5 * math.sin(now / 170.0)
            if self.busy:
                return
            last = self.game.last_move
            if last is not None and self.gallery_index is None:
                view.ground_ring(last.start, 0.36, GOLD, 2, 150)
                view.ground_ring(last.end, 0.5, GOLD, 3, 190)
            if self.selected:
                view.ground_ring(self.selected, 0.56 + 0.04 * pulse, (255, 240, 170), 4, 230)
            for move in self.legal_moves:
                if move.captured:
                    view.ground_ring(move.end, 0.5 + 0.05 * pulse, CAPTURE, 4, 230)
                else:
                    view.ground_ring(move.end, 0.13 + 0.03 * pulse, LEGAL, 0, 210)
                    view.ground_ring(move.end, 0.2, (90, 50, 20), 2, 120)
            if now < self.threats_until:
                fade = (self.threats_until - now) / 2400
                for sq in self.threats:
                    view.ground_ring(sq, 0.48 + 0.08 * pulse, CAPTURE, 3, 220 * fade)
            if self.game.winner is None and self.game.is_in_check(self.game.turn):
                king = self.game.king_position(self.game.turn)
                if king:
                    view.ground_ring(king, 0.6 + 0.08 * pulse, CAPTURE, 5, 240)
                    view.ground_ring(king, 0.78 + 0.1 * pulse, (255, 200, 120), 2, 160)
        return draw

    def ambient_frame(self, now: int) -> Frame:
        """Idle life on the board when no move is playing: chosen piece awake, scared king, KO after a mate."""
        f = Frame(now=now)
        board = self.game.board
        if self.selected and board[self.selected[0]][self.selected[1]] is not None:
            piece = board[self.selected[0]][self.selected[1]]
            awake = min(1.0, (now - self.selected_at) / 260)
            f.add(Sprite(piece, *self.selected, z=0.06 + 0.06 * math.sin(now / 160.0), awake=awake, expr="fierce",
                         walk=now / 160.0 if awake >= 1 else 0.0), hide=self.selected)
        if self.game.winner is not None:
            t = now - self.game_over_at
            loser = BLACK if self.game.winner == RED else RED
            lk, wk = self.game.king_position(loser), self.game.king_position(self.game.winner)
            if lk:
                f.add(Sprite(Piece(loser, KING), *lk, awake=min(1.0, t / 300), expr="ko", tilt=-1.3,
                             props=frozenset({"no_hat"})), hide=lk)
            if wk:
                f.add(Sprite(Piece(self.game.winner, KING), *wk, awake=min(1.0, t / 300), expr="cheer",
                             z=abs(math.sin(now / 220.0)) * 0.3, swing=0.5 + 0.5 * math.sin(now / 300.0)), hide=wk)
        elif self.game.is_in_check(self.game.turn) and self.mode == "play":
            king = self.game.king_position(self.game.turn)
            if king and king != self.selected:
                f.add(Sprite(Piece(self.game.turn, KING), *king, awake=1.0, expr="scared",
                             shake=math.sin(now / 30.0) * 2.0), hide=king)
        return f

    def target_mood(self) -> float:
        return max(-1.0, min(1.0, material_balance(self.shown_board()) / 12.0))

    def draw_board_scene(self, now: int) -> None:
        self.view.flipped = self.flipped
        show = self.current_show()
        frame = show.frame(now) if show is not None and show.started_at is not None else self.ambient_frame(now)
        self.view.compose(self.canvas, self.shown_board(), frame, now, self.mood, (self.mood, -self.mood),
                          ground=self.ground_marks(now))
        if self.game.winner and not self.busy and self.gallery_index is None:
            self.draw_victory(now)

    def draw_victory(self, now: int) -> None:
        t = now - self.game_over_at
        u = 0.13 + 0.6 * seg(t, 0, 600)
        title = f"{side_name(self.game.winner)}胜"
        color = (190, 40, 30) if self.game.winner == RED else (40, 46, 56)
        draw_ribbon(self.canvas, title, f"{self.game.reason} · 按 R 再来一局", (VIEW_W / 2, 812), u, color)

    # --- sidebar -----------------------------------------------------------------
    def draw_header(self) -> None:
        self.draw_text("木韵棋院", self.fonts["hero"], TEXT, (SIDE_X + 4, 22))
        self.draw_text("2.5D 木棋剧场 · 每一步都有戏", self.fonts["small"], GOLD_LIGHT, (SIDE_X + 8, 76))
        seal = pygame.Rect(SIDE_X + SIDE_W - 70, 26, 62, 62)
        pygame.draw.rect(self.canvas, (190, 40, 30), seal, border_radius=8)
        pygame.draw.rect(self.canvas, (255, 226, 200), seal.inflate(-8, -8), 2, border_radius=6)
        self.draw_text("棋", self.fonts["title"], (255, 236, 220), seal.center, center=True)

    def draw_status_card(self, y: int) -> None:
        card = pygame.Rect(SIDE_X, y, SIDE_W, 104)
        self.draw_panel(card)
        if self.gallery_index is not None:
            headline = "演出图鉴"
        elif self.game.winner:
            headline = f"{side_name(self.game.winner)}胜"
        elif self.mode == "editor":
            headline = "自定义棋局"
        else:
            headline = f"{side_name(self.game.turn)}行棋"
        color = RED_T if (self.game.winner or self.game.turn) == RED else BLACK_T
        self.draw_text(headline, self.fonts["title"], color, (SIDE_X + 20, y + 14))
        if self.mode == "editor":
            badge = "布阵中"
        else:
            badge = {"ai": "人机对弈", "local": "双人对弈", "watch": "观棋剧场"}[self.opponent_mode]
        badge_rect = pygame.Rect(SIDE_X + SIDE_W - 132, y + 16, 112, 32)
        pygame.draw.rect(self.canvas, LACQUER_2, badge_rect, border_radius=16)
        pygame.draw.rect(self.canvas, GOLD, badge_rect, 1, border_radius=16)
        self.draw_text(badge, self.fonts["small"], TEXT, badge_rect.center, center=True)
        balance = material_balance(self.shown_board())
        label = SITUATIONS[situation(balance)]
        bar = pygame.Rect(SIDE_X + 20, y + 66, SIDE_W - 140, 18)
        pygame.draw.rect(self.canvas, (24, 26, 30), bar, border_radius=9)
        split = bar.x + int(bar.w * (0.5 + 0.5 * self.mood))
        pygame.draw.rect(self.canvas, (196, 52, 40), (bar.x, bar.y, max(0, split - bar.x), bar.h), border_radius=9)
        pygame.draw.line(self.canvas, GOLD_LIGHT, (split, bar.y - 3), (split, bar.bottom + 3), 3)
        pygame.draw.rect(self.canvas, GOLD, bar, 1, border_radius=9)
        self.draw_text("局势", self.fonts["tiny"], MUTED, (SIDE_X + 20, y + 47))
        self.draw_text(label, self.fonts["button"], RED_T if balance > 2.5 else BLACK_T if balance < -2.5 else TEXT,
                       (bar.right + 60, bar.centery), center=True)

    def draw_sidebar(self) -> None:
        self.buttons.clear()
        self.palette.clear()
        self.draw_header()
        self.draw_status_card(112)
        y, x = 232, SIDE_X
        self.add_button("new", x, y, 124, "新对局", accent=True)
        self.add_button("undo", x + 135, y, 124, "悔棋")
        self.add_button("flip", x + 270, y, 124, "翻转棋盘", active=self.flipped)
        self.add_button("editor", x + 405, y, 125, "局面编辑", active=self.mode == "editor")
        y += 54
        self.add_button("mode", x, y, 124, "模式 · " + MODE_LABELS[self.opponent_mode], active=self.opponent_mode != "local")
        self.add_button("side", x + 135, y, 124, "执子 · " + ("红" if self.human_side == RED else "黑"))
        self.add_button("depth", x + 270, y, 124, f"棋力 · {'萌新' if self.ai.depth == 1 else '进阶' if self.ai.depth == 2 else '高手'}")
        self.add_button("story", x + 405, y, 125, "演出 · " + STORY_LABELS[self.story_mode], active=self.story_mode != "off")
        if self.mode == "editor":
            self.draw_editor_panel(x, y + 62)
        else:
            y += 54
            self.add_button("gallery", x, y, SIDE_W, "演出图鉴 · G（开局 / 吃子 / 局势 / 战术 / 杀法）",
                            active=self.gallery_index is not None)
            self.draw_play_panel(x, y + 60)
        self.draw_status_bar()

    def draw_play_panel(self, x: int, y: int) -> None:
        players = pygame.Rect(x, y, SIDE_W, 70)
        self.draw_panel(players)
        for index, side in enumerate((RED, BLACK)):
            px = x + 18 + index * 262
            active = self.game.turn == side and not self.game.winner
            face = self.view.puck(Piece(side, KING), 56, 1.0)
            self.canvas.blit(face, face.get_rect(center=(px + 26, y + 35)))
            label = ("红帅" if side == RED else "黑将")
            if self.opponent_mode == "watch" or (side != self.human_side and self.opponent_mode == "ai"):
                label += " · AI"
            self.draw_text(label, self.fonts["button"], TEXT, (px + 62, y + 14))
            self.draw_text("正在思考" if active else "等待回合", self.fonts["tiny"], (RED_T if side == RED else BLACK_T) if active else MUTED, (px + 62, y + 40))

        commentary = pygame.Rect(x, y + 82, SIDE_W, 100)
        self.draw_panel(commentary)
        self.draw_text("解说", self.fonts["tiny"], GOLD_LIGHT, (x + 18, y + 94))
        pygame.draw.line(self.canvas, (140, 96, 54), (x + 60, y + 103), (x + SIDE_W - 20, y + 103), 1)
        width = 27
        lines = [self.commentary[i:i + width] for i in range(0, min(len(self.commentary), width * 3), width)]
        for index, line in enumerate(lines[:3]):
            self.draw_text(line, self.fonts["small"], TEXT, (x + 18, y + 114 + index * 22))

        log = pygame.Rect(x, y + 194, SIDE_W, 168)
        self.draw_panel(log)
        self.draw_text("对局棋谱", self.fonts["button"], TEXT, (x + 18, y + 206))
        self.draw_text(f"第 {self.game.halfmove // 2 + 1:02d} 回合", self.fonts["tiny"], GOLD_LIGHT, (x + SIDE_W - 110, y + 209))
        pygame.draw.line(self.canvas, (140, 96, 54), (x + 18, y + 234), (x + SIDE_W - 18, y + 234), 1)
        if not self.move_log:
            self.draw_text("尚无着法……", self.fonts["small"], MUTED, (x + 20, y + 250))
            return
        first = max(0, len(self.move_log) - 10)
        first -= first % 2
        recent = self.move_log[first:]
        for index in range(0, len(recent), 2):
            number = (first + index) // 2 + 1
            row_y = y + 244 + index // 2 * 23
            if index // 2 % 2 == 0:
                pygame.draw.rect(self.canvas, (84, 48, 26), (x + 14, row_y - 2, SIDE_W - 28, 22), border_radius=6)
            self.draw_text(f"{number:02d}", self.fonts["tiny"], GOLD, (x + 22, row_y + 1))
            self.draw_text(recent[index], self.fonts["small"], (255, 214, 200), (x + 70, row_y - 2))
            if index + 1 < len(recent):
                self.draw_text(recent[index + 1], self.fonts["small"], (210, 230, 232), (x + 290, row_y - 2))

    def draw_editor_panel(self, x: int, y: int) -> None:
        panel = pygame.Rect(x, y, SIDE_W, 330)
        self.draw_panel(panel)
        self.draw_text("棋子工具箱", self.fonts["button"], TEXT, (x + 20, y + 16))
        self.draw_text("左键放置 · 右键擦除", self.fonts["tiny"], MUTED, (x + SIDE_W - 150, y + 20))
        for side_index, side in enumerate((RED, BLACK)):
            py = y + 86 + side_index * 76
            self.draw_text("红" if side == RED else "黑", self.fonts["button"], RED_T if side == RED else BLACK_T, (x + 20, py - 12))
            for index, kind in enumerate(KINDS):
                rect = pygame.Rect(x + 62 + index * 64, py - 30, 56, 56)
                piece = Piece(side, kind)
                self.palette.append((rect, piece))
                if piece == self.editor_piece:
                    pygame.draw.rect(self.canvas, JADE, rect.inflate(6, 6), 3, border_radius=14)
                face = self.view.puck(piece, 54, 1.0)
                self.canvas.blit(face, face.get_rect(center=rect.center))
        by = y + 218
        self.add_button("standard", x + 20, by, 120, "标准布局")
        self.add_button("clear", x + 150, by, 100, "清空")
        self.add_button("turn", x + 260, by, 120, "先手 · " + ("红" if self.game.turn == RED else "黑"))
        self.add_button("start", x + 390, by, 120, "开始对局", accent=True)
        by += 54
        self.add_button("import", x + 20, by, 120, "导入 FEN")
        self.add_button("export", x + 150, by, 120, "复制 FEN")

    def draw_status_bar(self) -> None:
        rect = pygame.Rect(SIDE_X, 790, SIDE_W, 82)
        self.draw_panel(rect)
        color = CAPTURE if "将军" in self.message or "胜" in self.message else GOLD
        pygame.draw.circle(self.canvas, color, (SIDE_X + 26, 816), 6)
        message = self.message
        if "思考" in message:
            message = "AI 正在思考" + "." * (1 + pygame.time.get_ticks() // 350 % 3)
        self.draw_text(message[:30], self.fonts["body"], TEXT, (SIDE_X + 44, 804))
        self.draw_text("F 翻转  U 悔棋  R 新局  空格/点击 跳过演出  G 图鉴", self.fonts["tiny"], MUTED, (SIDE_X + 44, 840))

    def draw_gallery_hud(self) -> None:
        demo = catalogue()[self.gallery_index]
        hud = f"演出图鉴 {self.gallery_index + 1:02d}/{len(catalogue())} · {CATEGORY_LABELS[demo.category]} · {demo.title}"
        label = self.fonts["button"].render(hud, True, (255, 236, 200))
        hint = self.fonts["tiny"].render("← → 切换   空格 重播   G / Esc 退出", True, MUTED)
        plate = pygame.Rect(SIDE_X, 726, SIDE_W, 56)
        pygame.draw.rect(self.canvas, (36, 20, 12), plate, border_radius=14)
        pygame.draw.rect(self.canvas, GOLD, plate, 2, border_radius=14)
        self.canvas.blit(label, label.get_rect(center=(plate.centerx, plate.y + 19)))
        self.canvas.blit(hint, hint.get_rect(center=(plate.centerx, plate.y + 42)))

    def draw_modal(self) -> None:
        shade = pygame.Surface((VW, VH), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 188))
        self.canvas.blit(shade, (0, 0))
        rect = pygame.Rect(250, 310, 940, 230)
        self.draw_panel(rect, alpha=250)
        self.draw_text("导入自定义棋局", self.fonts["title"], TEXT, (282, 334))
        self.draw_text("粘贴 FEN（w 红先 / b 黑先）", self.fonts["small"], MUTED, (282, 383))
        field = pygame.Rect(282, 423, 876, 50)
        pygame.draw.rect(self.canvas, (24, 12, 6), field, border_radius=9)
        pygame.draw.rect(self.canvas, JADE, field, 2, border_radius=9)
        self.draw_text(self.text_input[-82:] + "|", self.fonts["small"], TEXT, (297, 438))
        self.draw_text("Enter 导入    Esc 取消    ⌘/Ctrl + V 粘贴", self.fonts["small"], GOLD, (282, 493))

    def draw(self) -> None:
        now = pygame.time.get_ticks()
        dt = max(0, now - self.last_tick)
        self.last_tick = now
        self.mood += (self.target_mood() - self.mood) * min(1.0, dt / 700.0)
        self.draw_board_scene(now)
        self.draw_sidebar()
        if self.gallery_index is not None:
            self.draw_gallery_hud()
        if self.modal:
            self.draw_modal()
        width, height = self.window.get_size()
        scale = min(width / VW, height / VH)
        if (width, height) == (VW, VH):
            self.window.blit(self.canvas, (0, 0))
        else:
            target = pygame.transform.smoothscale(self.canvas, (int(VW * scale), int(VH * scale)))
            self.window.fill((10, 6, 4))
            self.window.blit(target, ((width - target.get_width()) // 2, (height - target.get_height()) // 2))
        pygame.display.flip()

    # --- game flow ---------------------------------------------------------------
    def new_game(self) -> None:
        self.cancel_ai()
        self.game.reset()
        self.mode = "play"
        self.selected = None
        self.legal_moves = []
        self.move_log.clear()
        self.clear_story()
        self.commentary = "新的一局开始了！棋子们揉揉眼睛，红方先行。"
        self.flipped = self.human_side == BLACK
        self.message = "新对局开始 · 红方先行"

    def cancel_ai(self) -> None:
        self.ai_future = None
        self.ai_position = ""

    def play_move(self, move: Move) -> None:
        piece = self.game.piece_at(move.start)
        if piece is None:
            return
        before = [row[:] for row in self.game.board]
        played = self.game.push(move)
        text = self.game.move_text(played) or piece.name
        self.move_log.append(text)
        self.selected = None
        self.legal_moves = []
        events = analyze_move(self.game, played, piece)
        after = [row[:] for row in self.game.board]
        self.shows.append(MoveShow(played, piece, before, after, events, self.story_mode, self.game.winner))
        self.threats = [
            attack.end for attack in self.game.pseudo_moves_from(*move.end)
            if attack.captured and attack.captured.side != piece.side and attack.captured.kind in (ROOK, HORSE, CANNON, KING)
        ]
        self.threats_until = 0
        self.update_commentary(piece, played, events)
        if self.game.winner:
            self.message = f"{side_name(self.game.winner)}获胜 · {self.game.reason}"
        elif self.game.is_in_check(self.game.turn):
            self.message = "将军！请谨慎应对"
        else:
            self.message = side_name(self.game.turn) + "行棋"

    def update_commentary(self, piece: Piece, move: Move, events: list[StoryEvent]) -> None:
        name = side_name(piece.side) + piece.name
        if not events:
            self.commentary = f"{name}伸了个懒腰，挪到了新位置，悄悄调整着攻防线路。"
            return
        order = ("mate", "stalemate", "tactic", "sacrifice", "capture", "opening", "check")
        event = min(events, key=lambda e: order.index(e.kind))
        if event.kind in ("mate", "stalemate"):
            self.commentary = f"『{event.title.replace(' ', '')}』！{event.subtitle}。{name}一锤定音。"
        elif event.kind == "tactic":
            self.commentary = f"妙手『{event.title}』—— {event.subtitle}。"
        elif event.kind == "capture":
            tag = f"【{event.tag}】" if event.tag else ""
            self.commentary = f"{tag}『{event.title}』{name}：{event.attacker_line}"
        elif event.kind == "opening":
            self.commentary = f"{side_name(piece.side)}布下『{event.title}』—— {event.subtitle.split(' · ', 1)[-1]}"
        else:
            self.commentary = f"『{event.title.replace(' ', '')}』{event.subtitle}，{name}步步紧逼。"

    def update_story(self) -> None:
        now = pygame.time.get_ticks()
        while self.shows:
            show = self.shows[0]
            if show.started_at is None:
                show.started_at = now
            if not show.done(now):
                break
            self.shows.pop(0)
            self.threats_until = now + 2400
            if self.game.winner and not self.shows:
                self.game_over_at = now
        self.update_gallery(now)

    def skip_show(self) -> bool:
        show = self.current_show()
        if show is not None and show.started_at is not None:
            show.skip(pygame.time.get_ticks())
            return True
        return False

    def clear_story(self) -> None:
        self.shows.clear()
        self.pending_board_click = None
        self.threats_until = 0

    # --- gallery -----------------------------------------------------------------
    def open_gallery(self, index: int) -> None:
        items = catalogue()
        if self.gallery_saved is None:
            self.gallery_saved = (self.game, list(self.move_log), self.flipped, self.mode, self.story_mode, self.commentary, self.message)
        self.cancel_ai()
        self.clear_story()
        self.gallery_index = index % len(items)
        demo = items[self.gallery_index]
        self.game = demo.build()
        self.move_log = []
        self.mode = "play"
        self.flipped = False
        self.selected, self.legal_moves = None, []
        self.story_mode = "full"
        self.gallery_due = pygame.time.get_ticks() + GALLERY_LEAD_MS
        self.gallery_played = False
        self.gallery_done_at = 0
        self.commentary = f"演出图鉴 · {CATEGORY_LABELS[demo.category]}『{demo.title}』即将上演。"
        self.message = "演出图鉴播放中"

    def close_gallery(self) -> None:
        if self.gallery_saved is not None:
            self.game, self.move_log, self.flipped, self.mode, self.story_mode, self.commentary, self.message = self.gallery_saved
        self.gallery_saved = None
        self.gallery_index = None
        self.clear_story()

    def update_gallery(self, now: int) -> None:
        if self.gallery_index is None:
            return
        if not self.gallery_played:
            if now >= self.gallery_due:
                self.gallery_played = True
                self.play_move(Move(*catalogue()[self.gallery_index].move))
            return
        if self.shows:
            return
        if not self.gallery_done_at:
            self.gallery_done_at = now
        elif now - self.gallery_done_at > GALLERY_LINGER_MS:
            self.open_gallery(self.gallery_index + 1)

    def handle_gallery_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_g):
                self.close_gallery()
            elif event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                self.open_gallery((self.gallery_index or 0) + 1)
            elif event.key in (pygame.K_LEFT, pygame.K_UP):
                self.open_gallery((self.gallery_index or 0) - 1)
            elif event.key == pygame.K_SPACE:
                self.open_gallery(self.gallery_index or 0)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = self.virtual_mouse(event.pos)
            gallery_button = next((b for b in self.buttons if b.key == "gallery"), None)
            if gallery_button and gallery_button.rect.collidepoint(pos):
                self.close_gallery()
            elif pos[0] < VIEW_W:
                self.open_gallery((self.gallery_index or 0) + (1 if event.button != 3 else -1))

    # --- input -------------------------------------------------------------------
    def handle_board_click(self, square: tuple[int, int], button: int) -> None:
        if self.busy:
            self.pending_board_click = (square, button)
            self.message = "已记录本次点击，演出结束后立即执行。"
            return
        if self.mode == "editor":
            if button == 3:
                self.game.set_piece(*square, None)
            elif button == 1 and self.editor_piece:
                self.game.set_piece(*square, self.editor_piece)
            return
        if button != 1 or self.game.winner or self.ai_controls_turn():
            return
        if self.selected:
            chosen = next((move for move in self.legal_moves if move.end == square), None)
            if chosen:
                self.play_move(chosen)
                return
        piece = self.game.piece_at(square)
        if piece and piece.side == self.game.turn:
            if self.selected != square:
                self.selected_at = pygame.time.get_ticks()
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
            self.clear_story()
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
            self.clear_story()
            self.mode = "play" if self.mode == "editor" else "editor"
            self.selected, self.legal_moves = None, []
            self.message = "局面编辑已开启" if self.mode == "editor" else "返回对局"
        elif key == "mode":
            order = ("ai", "local", "watch")
            self.opponent_mode = order[(order.index(self.opponent_mode) + 1) % len(order)]
            self.cancel_ai()
            self.message = {"ai": "已切换为人机对弈", "local": "已切换为双人对弈", "watch": "观战模式：AI 对 AI，尽情欣赏剧情"}[self.opponent_mode]
        elif key == "story":
            self.story_mode = STORY_MODES[(STORY_MODES.index(self.story_mode) + 1) % len(STORY_MODES)]
            self.message = {"full": "演出：每一步都有完整剧情", "highlights": "演出：只放大精彩时刻",
                            "off": "演出：只保留走子与吃子动作"}[self.story_mode]
        elif key == "gallery":
            if self.gallery_index is None:
                self.open_gallery(0)
            else:
                self.close_gallery()
        elif key == "side":
            self.human_side = BLACK if self.human_side == RED else RED
            self.new_game()
        elif key == "depth":
            self.ai.depth = 1 if self.ai.depth >= 3 else self.ai.depth + 1
            self.message = f"AI 棋力已设为 {'入门' if self.ai.depth == 1 else '进阶' if self.ai.depth == 2 else '高手'}"
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
            self.message = side_name(self.game.turn) + "先行"
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
        if self.gallery_index is not None:
            self.handle_gallery_event(event)
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.skip_show()
            elif event.key == pygame.K_g:
                self.open_gallery(0)
            elif event.key == pygame.K_f:
                self.flipped = not self.flipped
            elif event.key == pygame.K_r:
                self.new_game()
            elif event.key == pygame.K_u:
                self.handle_button("undo")
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = self.virtual_mouse(event.pos)
            for button in self.buttons:
                if button.rect.collidepoint(pos):
                    self.handle_button(button.key)
                    return
            if self.skip_show():
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
        if self.busy or self.pending_board_click is None:
            return
        square, button = self.pending_board_click
        self.pending_board_click = None
        self.handle_board_click(square, button)

    def ai_controls_turn(self) -> bool:
        if self.opponent_mode == "watch":
            return True
        return self.opponent_mode == "ai" and self.game.turn != self.human_side

    def update_ai(self) -> None:
        if self.mode != "play" or self.game.winner or not self.ai_controls_turn() or self.gallery_index is not None:
            return
        now = pygame.time.get_ticks()
        if self.busy:
            self.ai_ready_at = now + (650 if self.opponent_mode == "watch" else 120)
            return
        if self.ai_future is None:
            if now < self.ai_ready_at:
                return
            clone = self.game.clone()
            self.ai_position = self.game.to_fen()
            self.ai_future = self.executor.submit(self.ai.choose_move, clone)
            self.message = "AI 正在思考"
            return
        if self.ai_future.done():
            future, self.ai_future = self.ai_future, None
            try:
                move = future.result()
                if move and self.ai_position == self.game.to_fen():
                    self.play_move(move)
                if self.ai.load_error and self.ai.load_error != self.ai_error_shown:
                    self.ai_error_shown = self.ai.load_error
                    self.message = f"PPO 模型：{self.ai.load_error}"
            except Exception as exc:
                self.message = f"AI 计算失败：{exc}"

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update_story()
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
