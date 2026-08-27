"""UI-free Chinese chess (Xiangqi) rules engine.

The engine uses canonical coordinates: black starts at row 0, red at row 9.
A position is a 10 x 9 board containing :class:`Piece` objects or ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

ROWS, COLS = 10, 9
RED, BLACK = "red", "black"
SIDES = (RED, BLACK)
KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN = "K", "A", "E", "H", "R", "C", "P"
KINDS = (KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN)
PIECE_NAMES = {
    (RED, KING): "帅", (RED, ADVISOR): "仕", (RED, ELEPHANT): "相",
    (RED, HORSE): "马", (RED, ROOK): "车", (RED, CANNON): "炮", (RED, PAWN): "兵",
    (BLACK, KING): "将", (BLACK, ADVISOR): "士", (BLACK, ELEPHANT): "象",
    (BLACK, HORSE): "马", (BLACK, ROOK): "车", (BLACK, CANNON): "砲", (BLACK, PAWN): "卒",
}
FEN_TO_PIECE = {
    "K": (RED, KING), "A": (RED, ADVISOR), "B": (RED, ELEPHANT),
    "N": (RED, HORSE), "R": (RED, ROOK), "C": (RED, CANNON), "P": (RED, PAWN),
    "k": (BLACK, KING), "a": (BLACK, ADVISOR), "b": (BLACK, ELEPHANT),
    "n": (BLACK, HORSE), "r": (BLACK, ROOK), "c": (BLACK, CANNON), "p": (BLACK, PAWN),
}
PIECE_TO_FEN = {value: key for key, value in FEN_TO_PIECE.items()}


@dataclass(frozen=True)
class Piece:
    side: str
    kind: str

    @property
    def name(self) -> str:
        return PIECE_NAMES[(self.side, self.kind)]


@dataclass(frozen=True)
class Move:
    start: tuple[int, int]
    end: tuple[int, int]
    captured: Optional[Piece] = None

    @property
    def action(self) -> int:
        return square_index(*self.start) * 90 + square_index(*self.end)


@dataclass
class _Snapshot:
    board: list[list[Optional[Piece]]]
    turn: str
    winner: Optional[str]
    reason: str
    halfmove: int
    last_move: Optional[Move]


def opponent(side: str) -> str:
    return BLACK if side == RED else RED


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def square_index(row: int, col: int) -> int:
    if not in_bounds(row, col):
        raise ValueError("square out of board")
    return row * COLS + col


def action_to_squares(action: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if not 0 <= action < 8100:
        raise ValueError("action must be in [0, 8100)")
    start, end = divmod(action, 90)
    return divmod(start, COLS), divmod(end, COLS)


def standard_board() -> list[list[Optional[Piece]]]:
    board: list[list[Optional[Piece]]] = [[None for _ in range(COLS)] for _ in range(ROWS)]
    back = (ROOK, HORSE, ELEPHANT, ADVISOR, KING, ADVISOR, ELEPHANT, HORSE, ROOK)
    for col, kind in enumerate(back):
        board[0][col] = Piece(BLACK, kind)
        board[9][col] = Piece(RED, kind)
    for col in (1, 7):
        board[2][col] = Piece(BLACK, CANNON)
        board[7][col] = Piece(RED, CANNON)
    for col in range(0, COLS, 2):
        board[3][col] = Piece(BLACK, PAWN)
        board[6][col] = Piece(RED, PAWN)
    return board


class XiangqiGame:
    """Complete mutable game state with legal move validation and undo."""

    def __init__(self, board: Optional[list[list[Optional[Piece]]]] = None, turn: str = RED):
        self.board = self._copy_board(board if board is not None else standard_board())
        self.turn = turn
        self.winner: Optional[str] = None
        self.reason = ""
        self.halfmove = 0
        self.last_move: Optional[Move] = None
        self.history: list[_Snapshot] = []
        self._validate_shape()

    @staticmethod
    def _copy_board(board: list[list[Optional[Piece]]]) -> list[list[Optional[Piece]]]:
        return [row[:] for row in board]

    def _validate_shape(self) -> None:
        if len(self.board) != ROWS or any(len(row) != COLS for row in self.board):
            raise ValueError("board must be 10 x 9")
        if self.turn not in SIDES:
            raise ValueError("turn must be red or black")

    def reset(self) -> None:
        self.board = standard_board()
        self.turn = RED
        self.winner = None
        self.reason = ""
        self.halfmove = 0
        self.last_move = None
        self.history.clear()

    def clear(self) -> None:
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.winner = None
        self.reason = ""
        self.halfmove = 0
        self.last_move = None
        self.history.clear()

    def clone(self, with_history: bool = False) -> "XiangqiGame":
        game = XiangqiGame(self.board, self.turn)
        game.winner = self.winner
        game.reason = self.reason
        game.halfmove = self.halfmove
        game.last_move = self.last_move
        if with_history:
            game.history = list(self.history)
        return game

    def piece_at(self, square: tuple[int, int]) -> Optional[Piece]:
        return self.board[square[0]][square[1]]

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        if not in_bounds(row, col):
            raise ValueError("square out of board")
        self.board[row][col] = piece
        self.winner = None
        self.reason = ""
        self.history.clear()
        self.last_move = None

    def king_position(self, side: str) -> Optional[tuple[int, int]]:
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece == Piece(side, KING):
                    return row, col
        return None

    def validate_position(self) -> list[str]:
        errors: list[str] = []
        for side, label in ((RED, "红方"), (BLACK, "黑方")):
            kings = sum(piece == Piece(side, KING) for row in self.board for piece in row)
            if kings != 1:
                errors.append(f"{label}必须恰有一个{PIECE_NAMES[(side, KING)]}")
        if not errors and self.generals_facing():
            errors.append("帅与将不能直接照面")
        return errors

    def generals_facing(self) -> bool:
        red = self.king_position(RED)
        black = self.king_position(BLACK)
        if red is None or black is None or red[1] != black[1]:
            return False
        col = red[1]
        lo, hi = sorted((red[0], black[0]))
        return all(self.board[row][col] is None for row in range(lo + 1, hi))

    def _palace_contains(self, side: str, row: int, col: int) -> bool:
        return 3 <= col <= 5 and ((7 <= row <= 9) if side == RED else (0 <= row <= 2))

    def _own_half(self, side: str, row: int) -> bool:
        return row >= 5 if side == RED else row <= 4

    def _ray_moves(self, row: int, col: int, piece: Piece, cannon: bool) -> Iterator[Move]:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            crossed = False
            nr, nc = row + dr, col + dc
            while in_bounds(nr, nc):
                target = self.board[nr][nc]
                if not cannon:
                    if target is None:
                        yield Move((row, col), (nr, nc))
                    else:
                        if target.side != piece.side:
                            yield Move((row, col), (nr, nc), target)
                        break
                elif not crossed:
                    if target is None:
                        yield Move((row, col), (nr, nc))
                    else:
                        crossed = True
                elif target is not None:
                    if target.side != piece.side:
                        yield Move((row, col), (nr, nc), target)
                    break
                nr, nc = nr + dr, nc + dc

    def pseudo_moves_from(self, row: int, col: int) -> Iterator[Move]:
        piece = self.board[row][col]
        if piece is None:
            return
        if piece.kind == ROOK:
            yield from self._ray_moves(row, col, piece, False)
            return
        if piece.kind == CANNON:
            yield from self._ray_moves(row, col, piece, True)
            return

        destinations: list[tuple[int, int]] = []
        if piece.kind == KING:
            destinations = [(row + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            destinations = [(r, c) for r, c in destinations if self._palace_contains(piece.side, r, c)]
            enemy_king = self.king_position(opponent(piece.side))
            if enemy_king and enemy_king[1] == col:
                lo, hi = sorted((row, enemy_king[0]))
                if all(self.board[r][col] is None for r in range(lo + 1, hi)):
                    destinations.append(enemy_king)
        elif piece.kind == ADVISOR:
            destinations = [(row + dr, col + dc) for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
            destinations = [(r, c) for r, c in destinations if self._palace_contains(piece.side, r, c)]
        elif piece.kind == ELEPHANT:
            for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc) and self._own_half(piece.side, nr) and self.board[row + dr // 2][col + dc // 2] is None:
                    destinations.append((nr, nc))
        elif piece.kind == HORSE:
            for dr, dc, lr, lc in ((2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
                                        (1, 2, 0, 1), (-1, 2, 0, 1), (1, -2, 0, -1), (-1, -2, 0, -1)):
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc) and self.board[row + lr][col + lc] is None:
                    destinations.append((nr, nc))
        elif piece.kind == PAWN:
            forward = -1 if piece.side == RED else 1
            destinations.append((row + forward, col))
            crossed = row <= 4 if piece.side == RED else row >= 5
            if crossed:
                destinations.extend(((row, col - 1), (row, col + 1)))

        for nr, nc in destinations:
            if not in_bounds(nr, nc):
                continue
            target = self.board[nr][nc]
            if target is None or target.side != piece.side:
                yield Move((row, col), (nr, nc), target)

    def _attacks_square(self, side: str, target: tuple[int, int]) -> bool:
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece is not None and piece.side == side:
                    if any(move.end == target for move in self.pseudo_moves_from(row, col)):
                        return True
        return False

    def is_in_check(self, side: str) -> bool:
        king = self.king_position(side)
        return king is None or self._attacks_square(opponent(side), king)

    def _is_legal(self, move: Move, side: str) -> bool:
        moving = self.board[move.start[0]][move.start[1]]
        captured = self.board[move.end[0]][move.end[1]]
        self.board[move.end[0]][move.end[1]] = moving
        self.board[move.start[0]][move.start[1]] = None
        legal = not self.is_in_check(side)
        self.board[move.start[0]][move.start[1]] = moving
        self.board[move.end[0]][move.end[1]] = captured
        return legal

    def legal_moves(self, side: Optional[str] = None) -> list[Move]:
        side = side or self.turn
        if self.king_position(side) is None:
            return []
        moves: list[Move] = []
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece is not None and piece.side == side:
                    moves.extend(move for move in self.pseudo_moves_from(row, col) if self._is_legal(move, side))
        return moves

    def legal_moves_from(self, square: tuple[int, int]) -> list[Move]:
        piece = self.piece_at(square)
        if piece is None or piece.side != self.turn or self.winner is not None:
            return []
        return [move for move in self.pseudo_moves_from(*square) if self._is_legal(move, self.turn)]

    def push(self, move: Move, validate: bool = True) -> Move:
        piece = self.piece_at(move.start)
        if piece is None:
            raise ValueError("no piece at move start")
        if validate:
            legal = next((candidate for candidate in self.legal_moves_from(move.start) if candidate.end == move.end), None)
            if legal is None:
                raise ValueError("illegal move")
            move = legal
        self.history.append(_Snapshot(self._copy_board(self.board), self.turn, self.winner, self.reason, self.halfmove, self.last_move))
        captured = self.piece_at(move.end)
        played = Move(move.start, move.end, captured)
        self.board[move.end[0]][move.end[1]] = piece
        self.board[move.start[0]][move.start[1]] = None
        self.last_move = played
        self.halfmove += 1
        self.turn = opponent(self.turn)
        if captured is not None and captured.kind == KING:
            self.winner = piece.side
            self.reason = "将帅被擒"
        elif not self.legal_moves(self.turn):
            self.winner = piece.side
            self.reason = "将死" if self.is_in_check(self.turn) else "困毙"
        return played

    def undo(self) -> bool:
        if not self.history:
            return False
        state = self.history.pop()
        self.board = state.board
        self.turn = state.turn
        self.winner = state.winner
        self.reason = state.reason
        self.halfmove = state.halfmove
        self.last_move = state.last_move
        return True

    def to_fen(self) -> str:
        rows: list[str] = []
        for board_row in self.board:
            empty = 0
            text = ""
            for piece in board_row:
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        text += str(empty)
                        empty = 0
                    text += PIECE_TO_FEN[(piece.side, piece.kind)]
            if empty:
                text += str(empty)
            rows.append(text)
        return "/".join(rows) + (" w" if self.turn == RED else " b")

    @classmethod
    def from_fen(cls, fen: str) -> "XiangqiGame":
        parts = fen.strip().split()
        if not parts:
            raise ValueError("empty FEN")
        ranks = parts[0].split("/")
        if len(ranks) != ROWS:
            raise ValueError("FEN must contain 10 ranks")
        board: list[list[Optional[Piece]]] = []
        for rank in ranks:
            row: list[Optional[Piece]] = []
            for char in rank:
                if char.isdigit():
                    row.extend([None] * int(char))
                elif char in FEN_TO_PIECE:
                    row.append(Piece(*FEN_TO_PIECE[char]))
                else:
                    raise ValueError(f"invalid FEN piece: {char}")
            if len(row) != COLS:
                raise ValueError("each FEN rank must contain 9 files")
            board.append(row)
        turn = RED if len(parts) < 2 or parts[1].lower() in ("w", "r", "red") else BLACK
        return cls(board, turn)

    def move_text(self, move: Move) -> str:
        piece = self.piece_at(move.start) or self.piece_at(move.end)
        if piece is None:
            return ""
        files = "九八七六五四三二一" if piece.side == RED else "一二三四五六七八九"
        start_file = files[move.start[1]]
        end_file = files[move.end[1]]
        delta = abs(move.end[0] - move.start[0])
        if move.start[0] == move.end[0]:
            verb, suffix = "平", end_file
        else:
            forward = move.end[0] < move.start[0] if piece.side == RED else move.end[0] > move.start[0]
            verb = "进" if forward else "退"
            suffix = end_file if piece.kind in (HORSE, ELEPHANT, ADVISOR) else "一二三四五六七八九"[delta - 1]
        return f"{piece.name}{start_file}{verb}{suffix}"


def iter_pieces(board: Iterable[Iterable[Optional[Piece]]]) -> Iterator[tuple[int, int, Piece]]:
    for row, rank in enumerate(board):
        for col, piece in enumerate(rank):
            if piece is not None:
                yield row, col, piece
