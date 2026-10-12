"""Recognise the story beats of a move: openings, captures, sacrifices, checks and mates.

Everything here is UI-free so it can be unit tested. Geometry is expressed relative to
the defending side: ``depth`` 0 is the defender's back rank, 4 the defender's river bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from game_engine import (
    ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK,
    Move, Piece, XiangqiGame, iter_pieces, opponent,
)

KIND_LABEL = {
    (RED, KING): "帅", (RED, ADVISOR): "仕", (RED, ELEPHANT): "相", (RED, HORSE): "马",
    (RED, ROOK): "车", (RED, CANNON): "炮", (RED, PAWN): "兵",
    (BLACK, KING): "将", (BLACK, ADVISOR): "士", (BLACK, ELEPHANT): "象", (BLACK, HORSE): "马",
    (BLACK, ROOK): "车", (BLACK, CANNON): "炮", (BLACK, PAWN): "卒",
}


Square = tuple[int, int]


@dataclass(frozen=True)
class StoryEvent:
    """One dramatic beat produced by a move.

    ``lines`` are (from, to) squares the board show should draw beams/arrows along,
    ``targets`` are squares under attack, ``cast`` are attacker pieces that take part.
    """

    kind: str  # opening | capture | sacrifice | tactic | check | mate | stalemate
    key: str
    title: str
    subtitle: str
    actor: Piece
    victim: Optional[Piece] = None
    attacker_line: str = ""
    victim_line: str = ""
    lines: tuple[tuple[Square, Square], ...] = ()
    targets: tuple[Square, ...] = ()
    cast: tuple[Square, ...] = ()
    tone: str = "even"  # dominant | even | desperate, from the actor's point of view
    tag: str = ""


OPENINGS: dict[str, tuple[str, str]] = {
    "center_cannon": ("当头炮", "炮镇中路 · 刚猛直爽，直取中卒"),
    "cross_palace_cannon": ("过宫炮", "横越九宫 · 双炮一侧，攻守兼备"),
    "advisor_cannon": ("仕角炮", "炮守仕角 · 稳健持重"),
    "pawn_base_cannon": ("兵底炮", "炮藏兵后 · 冷门奇袭"),
    "retreat_cannon": ("敛炮局", "退炮藏锋 · 以静制动"),
    "edge_cannon": ("边炮局", "炮走边陲 · 为车马让路"),
    "river_cannon": ("巡河炮", "炮巡河岸 · 左右牵制"),
    "hook_cannon": ("金钩炮", "金钩垂钓 · 静待鱼来"),
    "cross_river_cannon": ("过河炮", "炮过楚河 · 先发制人"),
    "immortal_pawn": ("仙人指路", "投石问路 · 保留万千变化"),
    "edge_pawn": ("边兵局", "边路挺兵 · 剑走偏锋"),
    "center_pawn": ("中兵局", "中兵冲锋 · 直面中路"),
    "flying_elephant": ("飞相局", "飞相镇中 · 后发制人"),
    "edge_elephant": ("边相局", "相走边路 · 为拐脚马让路"),
    "horse_opening": ("起马局", "骏马出厩 · 灵活多变"),
    "edge_horse": ("边马局", "马屯边陲 · 迂回包抄"),
    "screen_horses": ("屏风马", "双马成屏 · 拱卫中路"),
    "same_cannons": ("顺炮", "同侧架炮 · 针锋相对"),
    "crossed_cannons": ("列炮", "异侧架炮 · 两军对垒"),
    "advisor_opening": ("补仕局", "补仕固宫 · 以守待攻"),
    "rook_opening": ("出车局", "战车先行 · 直线开道"),
    "king_opening": ("御驾亲征", "主帅出宫巡视 · 胆气过人"),
}

CAPTURES: dict[str, tuple[str, str, str]] = {
    "RR": ("双车对撞", "这条线，只容得下一辆车！", "刹……刹不住了！"),
    "RH": ("铁轮断蹄", "跑得再快，也快不过直线！", "我的腿还没热身呢！"),
    "RC": ("直捣炮台", "装填太慢了！", "等、等我点个火——"),
    "RE": ("碾碎象阵", "大块头也挡不住车轮。", "我的象牙！我的象牙！"),
    "RA": ("破宫斩仕", "九宫的门，我替你拆了。", "成何体统！成何体统！"),
    "RP": ("车轮碾卒", "小兵也敢挡路？", "早知道就不过河了……"),
    "RK": ("铁骑擒王", "王座，换人坐了。", "护驾——"),
    "HR": ("蹄踏战车", "拐个弯你就看不见我了吧？", "我只会走直线啊！"),
    "HH": ("双骑缠斗", "同是千里驹，这局我先到！", "嘶——同类相残！"),
    "HC": ("踏破炮营", "炮身太重，躲不开吧？", "近身了！近身我打不了！"),
    "HE": ("惊象踏阵", "象眼塞不住我的蹄子！", "别踩我鼻子！"),
    "HA": ("跃马擒仕", "宫门之内，我也来去自如。", "护驾！快护驾！"),
    "HP": ("骏马践卒", "一蹄子的事。", "我、我只是路过！"),
    "HK": ("单骑擒王", "这一跃，直入中军。", "怎么是马？！"),
    "CR": ("隔山打车", "隔着一座山，照样打中你。", "躲在后面的最阴险！"),
    "CH": ("炮轰骏马", "跳得再高，也在弹道里。", "炮弹怎么会拐弯？！"),
    "CC": ("炮火对轰", "比比谁的引信短！", "同行何必相轰——"),
    "CE": ("炮打飞象", "大目标，好瞄准。", "我只是站得比较显眼……"),
    "CA": ("炮破宫墙", "宫墙再厚，也挡不住炮火。", "斯文扫地！"),
    "CP": ("一炮清卒", "小试牛刀。", "用炮打我？太看得起我了！"),
    "CK": ("一炮定乾坤", "终点，就在炮口前。", "这一炮……躲不掉了。"),
    "ER": ("巨象掀车", "我的地盘，车也得翻！", "轮子……轮子朝天了！"),
    "EH": ("象踏惊马", "田字一迈，马就趴下。", "居然被慢吞吞的家伙抓住了……"),
    "EC": ("象鼻缴炮", "这根炮管，没收了。", "我的炮管被卷走了！"),
    "EE": ("双象角力", "比比谁的牙更硬！", "呜——象牙断了！"),
    "EA": ("飞象擒仕", "田字之内，皆是我土。", "这大块头怎么进来的？"),
    "EP": ("象足踏兵", "过河的小兵，来我家做客？", "这地好软……不，是象脚！"),
    "EK": ("巨象擒王", "大地在为我震动。", "地震了？！"),
    "AR": ("羽扇斩车", "谈笑间，战车灰飞烟灭。", "被书生给拦下了？！"),
    "AH": ("宫门擒马", "宫里不许骑马。", "这扇子比刀还快！"),
    "AC": ("扇灭火绳", "引信？一扇就灭。", "我的火……灭了……"),
    "AE": ("仕象相争", "文官也有三分力气。", "被扇子扇倒了……"),
    "AA": ("双仕论道", "论辩，你输了。", "竟无言以对……"),
    "AP": ("护宫逐卒", "九宫重地，闲人免进。", "我就看一眼……"),
    "AK": ("仕擒主帅", "臣，斗胆了。", "你、你造反？！"),
    "KR": ("御驾斩车", "敢闯本帅的宫殿？", "老大亲自动手了！"),
    "KH": ("王剑落马", "本帅也懂骑术。", "被国王踢下马了……"),
    "KC": ("王者缴械", "近身的炮，不过是铁管。", "离太近了！失算！"),
    "KE": ("王威镇象", "跪下。", "遵、遵命……"),
    "KA": ("王斥谋士", "退下吧。", "臣告退……"),
    "KP": ("亲手擒卒", "勇气可嘉，可惜到此为止。", "能被大王亲手抓住，值了！"),
    "KK": ("王见王", "这里只能有一位王。", "……"),
    "PR": ("小卒逆袭", "别小看过河的兵！", "被一个小卒……丢人啊！"),
    "PH": ("长矛挑马", "一步一步，终于够到你了。", "我竟然被兵绊倒了！"),
    "PC": ("卒拔炮台", "近身之后，炮就是废铁！", "离我远点！"),
    "PE": ("兵扳巨象", "大象怕老鼠，也怕小兵！", "痒……痒死了……"),
    "PA": ("兵入九宫", "九宫，我来了！", "无礼！无礼之极！"),
    "PP": ("兵卒相搏", "狭路相逢勇者胜！", "大家都是小兵，何苦呢……"),
    "PK": ("小卒擒王", "小卒也能改写历史！", "居然……是你？"),
}

MATES: dict[str, tuple[str, str]] = {
    "double_rook": ("双车错", "长短双车交错将军 · 王无处可逃"),
    "wocao_horse": ("卧槽马", "马卧相前槽口 · 将门洞开"),
    "corner_horse": ("挂角马", "马挂士角 · 斜刺宫心"),
    "fishing_horse": ("钓鱼马", "马钓三七路 · 静待王入彀"),
    "horse_cannon": ("马后炮", "以马为架 · 炮从后发"),
    "double_cannon": ("重炮", "双炮叠阵 · 中轴贯穿"),
    "heaven_earth": ("天地炮", "一炮镇天 · 一炮沉地"),
    "iron_bolt": ("铁门栓", "中炮镇宫 · 车兵栓门"),
    "heart_strike": ("大胆穿心", "弃车剜心 · 一刀入宫"),
    "facing_kings": ("白脸将", "王不见王 · 对面笑杀"),
    "sea_moon": ("海底捞月", "沉底绕后 · 捞月擒王"),
    "two_ghosts": ("二鬼拍门", "双兵叩门 · 贴身逼杀"),
    "generic": ("绝杀", "退路尽封 · 无棋可应"),
}

STALEMATE = ("困毙", "将未被将，却已无路可走")

TACTICS: dict[str, tuple[str, str]] = {
    "fork": ("捉双", "一子双捉 · 顾此失彼"),
    "discovered_attack": ("闪击", "移步让线 · 身后暗箭齐发"),
    "discovered_check": ("闪将", "让开一线 · 背后将军"),
    "double_check": ("双将", "两路同将 · 只能动王"),
    "check_win": ("将军抽{x}", "先将一军 · 回手再抽{x}"),
}

VALUE = {KING: 0.0, ROOK: 9.0, CANNON: 4.5, HORSE: 4.0, ELEPHANT: 2.0, ADVISOR: 2.0, PAWN: 1.0}

SITUATIONS = {
    "red_big": "红方大优", "red_up": "红方小优", "even": "相持",
    "black_up": "黑方小优", "black_big": "黑方大优",
}

TONE_TAGS = {"dominant": "乘胜追击", "desperate": "绝地反击", "comeback": "扳回一城", "breakthrough": "打破均势"}


def _side_name(side: str) -> str:
    return "红方" if side == RED else "黑方"


def depth_of(row: int, defender: str) -> int:
    """Rows counted from the defender's own back rank (0) towards the enemy (9)."""
    return row if defender == BLACK else 9 - row


def _first_move_opening(game: XiangqiGame, move: Move, piece: Piece) -> Optional[str]:
    side = piece.side
    home = 9 if side == RED else 0
    fwd = -1 if side == RED else 1
    cannon_row = home + 2 * fwd
    pawn_row = home + 3 * fwd
    (sr, sc), (er, ec) = move.start, move.end

    if piece.kind == CANNON and sr == cannon_row and sc in (1, 7):
        if er == sr:
            if ec == 4:
                return "center_cannon"
            if (sc, ec) in ((1, 5), (7, 3)):
                return "cross_palace_cannon"
            if (sc, ec) in ((1, 3), (7, 5)):
                return "advisor_cannon"
            if (sc, ec) in ((1, 2), (7, 6)):
                return "pawn_base_cannon"
            if ec in (0, 8):
                return "edge_cannon"
        elif ec == sc:
            steps = (er - sr) * fwd
            if steps == -1:
                return "retreat_cannon"
            if steps == 1:
                return "hook_cannon"
            if steps == 2:
                return "river_cannon"
            if steps >= 3:
                return "cross_river_cannon"
    if piece.kind == PAWN and sr == pawn_row and ec == sc and er == sr + fwd:
        if sc in (2, 6):
            return "immortal_pawn"
        if sc in (0, 8):
            return "edge_pawn"
        if sc == 4:
            return "center_pawn"
    if piece.kind == ELEPHANT and sr == home:
        if move.end == (home + 2 * fwd, 4):
            return "flying_elephant"
        if er == home + 2 * fwd and ec in (0, 8):
            return "edge_elephant"
    if piece.kind == HORSE and sr == home and er == home + 2 * fwd:
        if ec in (2, 6):
            return "horse_opening"
        if ec in (0, 8):
            return "edge_horse"
    if piece.kind == ADVISOR:
        return "advisor_opening"
    if piece.kind == ROOK:
        return "rook_opening"
    if piece.kind == KING:
        return "king_opening"
    return None


def detect_opening(game: XiangqiGame, move: Move, piece: Piece) -> Optional[str]:
    """Classify a side's first move (plus the 屏风马 / 顺炮 / 列炮 replies).

    ``game`` is the position *after* ``move`` was pushed.
    """
    if move.captured is not None:
        return None
    count = sum(1 for _ in iter_pieces(game.board))
    side = piece.side
    home = 9 if side == RED else 0
    fwd = -1 if side == RED else 1
    if piece.kind == HORSE and game.halfmove <= 6 and move.end[0] == home + 2 * fwd and move.end[1] in (2, 6):
        partner = (move.end[0], 8 - move.end[1])
        if game.piece_at(partner) == Piece(side, HORSE) and game.halfmove > 2 and count >= 30:
            return "screen_horses"
    if game.halfmove not in (1, 2) or count != 32:
        return None
    key = _first_move_opening(game, move, piece)
    if key == "center_cannon" and game.halfmove == 2 and len(game.history) >= 2:
        red_cannon_from = 1 if game.history[1].board[7][1] is None and game.history[0].board[7][1] is not None else 7
        red_first_was_center = game.history[1].board[7][4] == Piece(RED, CANNON)
        if red_first_was_center:
            return "same_cannons" if red_cannon_from == move.start[1] else "crossed_cannons"
    return key


def _attackers(game: XiangqiGame, side: str, target: tuple[int, int]) -> list[tuple[tuple[int, int], Piece]]:
    found = []
    for row, col, piece in iter_pieces(game.board):
        if piece.side == side and any(m.end == target for m in game.pseudo_moves_from(row, col)):
            found.append(((row, col), piece))
    return found


def _cannon_screen(game: XiangqiGame, cannon: tuple[int, int], target: tuple[int, int]) -> Optional[tuple[tuple[int, int], Piece]]:
    (r1, c1), (r2, c2) = cannon, target
    if r1 != r2 and c1 != c2:
        return None
    dr = (r2 > r1) - (r2 < r1)
    dc = (c2 > c1) - (c2 < c1)
    r, c = r1 + dr, c1 + dc
    while (r, c) != (r2, c2):
        piece = game.board[r][c]
        if piece is not None:
            return (r, c), piece
        r, c = r + dr, c + dc
    return None


def _facing_blocks_escape(game: XiangqiGame, defender: str) -> bool:
    """True when some king escape square is unsafe only because the two kings would face each other."""
    attacker = opponent(defender)
    king = game.king_position(defender)
    enemy_king = game.king_position(attacker)
    if king is None or enemy_king is None:
        return False
    kr, kc = king
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = kr + dr, kc + dc
        if not (3 <= c <= 5 and 0 <= depth_of(r, defender) <= 2):
            continue
        occupant = game.board[r][c]
        if occupant is not None and occupant.side == defender:
            continue
        if c != enemy_king[1]:
            continue
        saved = game.board[r][c]
        game.board[kr][kc] = None
        game.board[r][c] = Piece(defender, KING)
        lo, hi = sorted((r, enemy_king[0]))
        facing = all(game.board[x][c] is None for x in range(lo + 1, hi))
        others = [
            (row, col) for row, col, p in iter_pieces(game.board)
            if p.side == attacker and p.kind != KING
            and any(m.end == (r, c) for m in game.pseudo_moves_from(row, col))
        ]
        game.board[r][c] = saved
        game.board[kr][kc] = Piece(defender, KING)
        if facing and not others:
            return True
    return False


def classify_attack_pattern(game: XiangqiGame, move: Move, piece: Piece) -> str:
    """Name the attacking pattern against the side to move (whose king is in check or mated)."""
    attacker = piece.side
    defender = opponent(attacker)
    king = game.king_position(defender)
    if king is None:
        return "generic"
    if piece.kind == ROOK and move.captured == Piece(defender, ADVISOR) and depth_of(move.end[0], defender) == 1 and move.end[1] == 4:
        return "heart_strike"
    checkers = _attackers(game, attacker, king)
    own = [((r, c), p) for r, c, p in iter_pieces(game.board) if p.side == attacker]

    def d(pos: tuple[int, int]) -> int:
        return depth_of(pos[0], defender)

    pawns_near = [pos for pos, p in own if p.kind == PAWN and d(pos) <= 2 and 2 <= pos[1] <= 6]
    if len(pawns_near) >= 2 and (piece.kind == PAWN or any(p.kind == PAWN for _, p in checkers)):
        return "two_ghosts"
    for pos, p in checkers:
        if p.kind == CANNON:
            screen = _cannon_screen(game, pos, king)
            if screen and screen[1] == Piece(attacker, CANNON):
                return "double_cannon"
            if screen and screen[1] == Piece(attacker, HORSE):
                return "horse_cannon"
    cannons = [pos for pos, p in own if p.kind == CANNON]
    checker_squares = {pos for pos, _ in checkers}
    for top in cannons:
        for bottom in cannons:
            if top != bottom and top[1] == 4 and d(bottom) == 0 and (top in checker_squares or bottom in checker_squares):
                return "heaven_earth"
    if any(pos[1] == 4 for pos in cannons) and d(king) == 0:
        if any(p.kind in (ROOK, PAWN) and pos[1] in (3, 5) for pos, p in checkers):
            return "iron_bolt"
    rooks = [pos for pos, p in own if p.kind == ROOK]
    if len(rooks) >= 2 and any(p.kind == ROOK for _, p in checkers):
        a, b = rooks[0], rooks[1]
        if abs(a[0] - b[0]) == 1 or abs(a[1] - b[1]) == 1:
            return "double_rook"
    for pos, p in checkers:
        if p.kind == HORSE:
            depth, col = d(pos), pos[1]
            if col in (3, 5) and depth in (0, 2):
                return "corner_horse"
            if depth == 1 and col in (1, 2, 6, 7):
                return "wocao_horse"
            if depth == 2 and col in (1, 2, 6, 7):
                return "fishing_horse"
    if piece.kind == KING or _facing_blocks_escape(game, defender):
        return "facing_kings"
    for pos, p in checkers:
        if p.kind in (ROOK, CANNON) and d(pos) == 0:
            return "sea_moon"
    return "generic"


def pattern_cast(game: XiangqiGame, key: str, attacker: str) -> tuple[Square, ...]:
    """Attacker pieces that star in a mating / checking pattern, checkers first."""
    defender = opponent(attacker)
    king = game.king_position(defender)
    if king is None:
        return ()
    cast = [pos for pos, _ in _attackers(game, attacker, king)]
    own = [((r, c), p) for r, c, p in iter_pieces(game.board) if p.side == attacker]
    extra: list[Square] = []
    if key == "double_rook":
        extra = [pos for pos, p in own if p.kind == ROOK]
    elif key == "two_ghosts":
        extra = [pos for pos, p in own if p.kind == PAWN and depth_of(pos[0], defender) <= 2 and 2 <= pos[1] <= 6]
    elif key in ("double_cannon", "horse_cannon"):
        for pos in list(cast):
            if game.piece_at(pos).kind == CANNON:
                screen = _cannon_screen(game, pos, king)
                if screen and screen[1].side == attacker:
                    extra.append(screen[0])
    elif key in ("heaven_earth", "iron_bolt"):
        extra = [pos for pos, p in own if p.kind == CANNON]
    elif key == "facing_kings":
        extra = [pos for pos, p in own if p.kind == KING]
    for pos in extra:
        if pos not in cast:
            cast.append(pos)
    return tuple(cast)


# --- material & situation --------------------------------------------------------

def piece_value(piece: Piece, row: int) -> float:
    value = VALUE[piece.kind]
    if piece.kind == PAWN and (row <= 4 if piece.side == RED else row >= 5):
        value += 1.0
    return value


def material_balance(board) -> float:
    """Material from red's point of view, in pawn units."""
    total = 0.0
    for row, _, piece in iter_pieces(board):
        total += piece_value(piece, row) * (1 if piece.side == RED else -1)
    return total


def situation(balance: float) -> str:
    if balance >= 6:
        return "red_big"
    if balance >= 2.5:
        return "red_up"
    if balance <= -6:
        return "black_big"
    if balance <= -2.5:
        return "black_up"
    return "even"


def lead_of(balance: float, side: str) -> float:
    return balance if side == RED else -balance


def tone_for(balance: float, side: str) -> str:
    lead = lead_of(balance, side)
    if lead >= 2.5:
        return "dominant"
    if lead <= -2.5:
        return "desperate"
    return "even"


def capture_tag(before: float, after: float, side: str) -> str:
    tone = tone_for(before, side)
    if tone == "desperate":
        return TONE_TAGS["comeback"] if lead_of(after, side) > -2.5 else TONE_TAGS["desperate"]
    if tone == "dominant":
        return TONE_TAGS["dominant"]
    return TONE_TAGS["breakthrough"] if lead_of(after, side) >= 2.5 else ""


# --- tactics -------------------------------------------------------------------

def _hits(game: XiangqiGame, square: Square) -> list[tuple[Square, Piece]]:
    return [(m.end, m.captured) for m in game.pseudo_moves_from(*square) if m.captured is not None]


def _protected(game: XiangqiGame, square: Square) -> bool:
    piece = game.piece_at(square)
    if piece is None:
        return False
    game.board[square[0]][square[1]] = Piece(opponent(piece.side), PAWN)
    guarded = game._attacks_square(piece.side, square)
    game.board[square[0]][square[1]] = piece
    return guarded


def _real_threat(game: XiangqiGame, attacker_square: Square, target: Square) -> bool:
    """A target worth shouting about: a major piece that is undefended or worth more than its attacker."""
    attacker = game.piece_at(attacker_square)
    victim = game.piece_at(target)
    if attacker is None or victim is None or VALUE[victim.kind] < 4:
        return False
    return VALUE[victim.kind] > VALUE[attacker.kind] or not _protected(game, target)


def _between(a: Square, b: Square, square: Square) -> bool:
    (r1, c1), (r2, c2), (r, c) = a, b, square
    if r1 == r2 == r:
        return min(c1, c2) < c < max(c1, c2)
    if c1 == c2 == c:
        return min(r1, r2) < r < max(r1, r2)
    return False


def _horse_leg(horse: Square, target: Square) -> Square:
    dr, dc = target[0] - horse[0], target[1] - horse[1]
    if abs(dr) == 2:
        return horse[0] + dr // 2, horse[1]
    return horse[0], horse[1] + dc // 2


def _unmasked(before: XiangqiGame, after: XiangqiGame, move: Move, side: str) -> list[tuple[Square, Square]]:
    """(attacker, target) pairs opened up because ``move`` stepped out of the line."""
    found = []
    for row, col, piece in iter_pieces(after.board):
        here = (row, col)
        if piece.side != side or here == move.end or piece.kind not in (ROOK, CANNON, HORSE):
            continue
        old = {sq for sq, _ in _hits(before, here)}
        for target, _ in _hits(after, here):
            if target in old:
                continue
            if piece.kind == HORSE:
                opened = _horse_leg(here, target) == move.start
            else:
                opened = _between(here, target, move.start)
            if opened:
                found.append((here, target))
    return found


def _check_win(game: XiangqiGame, attacker: str) -> Optional[tuple[Square, Square, Piece]]:
    """After a check, the material the attacker is guaranteed to pick up whatever the defence."""
    defender = opponent(attacker)
    replies = game.legal_moves(defender)
    if not replies:
        return None
    worst: Optional[tuple[float, tuple[Square, Square, Piece]]] = None
    board = game.board
    for reply in replies:
        moving, taken = board[reply.start[0]][reply.start[1]], board[reply.end[0]][reply.end[1]]
        board[reply.end[0]][reply.end[1]], board[reply.start[0]][reply.start[1]] = moving, None
        best: Optional[tuple[float, tuple[Square, Square, Piece]]] = None
        for row, col, piece in list(iter_pieces(board)):
            if piece.side != attacker:
                continue
            for m in game.pseudo_moves_from(row, col):
                if m.captured is None or VALUE[m.captured.kind] < 4 or not game._is_legal(m, attacker):
                    continue
                gain = VALUE[m.captured.kind]
                board[m.end[0]][m.end[1]], board[row][col] = piece, None
                if game._attacks_square(defender, m.end):
                    gain -= VALUE[piece.kind]
                board[row][col], board[m.end[0]][m.end[1]] = piece, m.captured
                if best is None or gain > best[0]:
                    best = (gain, ((row, col), m.end, m.captured))
        board[reply.start[0]][reply.start[1]], board[reply.end[0]][reply.end[1]] = moving, taken
        if best is None or best[0] < 3.5:
            return None
        if worst is None or best[0] < worst[0]:
            worst = best
    return worst[1] if worst else None


def detect_tactic(before: XiangqiGame, after: XiangqiGame, move: Move, piece: Piece) -> Optional[StoryEvent]:
    """捉双 / 闪击 / 闪将 / 双将 / 将军抽X for a move that did not end the game."""
    side = piece.side
    defender = opponent(side)
    king = after.king_position(defender)
    if king is None or after.winner is not None:
        return None
    if after.is_in_check(defender):
        checkers = [pos for pos, _ in _attackers(after, side, king)]
        lines = tuple((pos, king) for pos in checkers)
        won = _check_win(after, side)
        if won is not None:
            capturer, target, victim = won
            title, subtitle = TACTICS["check_win"]
            name = KIND_LABEL[(victim.side, victim.kind)]
            return StoryEvent(
                "tactic", "check_win", title.format(x=name), subtitle.format(x=name), piece, victim,
                lines=lines + ((capturer, target),), targets=(king, target), cast=tuple(dict.fromkeys(checkers + [capturer])),
            )
        if len(checkers) >= 2:
            title, subtitle = TACTICS["double_check"]
            return StoryEvent("tactic", "double_check", title, subtitle, piece, lines=lines, targets=(king,), cast=tuple(checkers))
        if move.end not in checkers:
            title, subtitle = TACTICS["discovered_check"]
            return StoryEvent("tactic", "discovered_check", title, subtitle, piece, lines=lines, targets=(king,),
                              cast=tuple(checkers + [move.end]))
        return None
    forks = [sq for sq, victim in _hits(after, move.end) if victim.kind != KING and _real_threat(after, move.end, sq)]
    if len(forks) >= 2:
        title, subtitle = TACTICS["fork"]
        names = "与".join(KIND_LABEL[(after.piece_at(sq).side, after.piece_at(sq).kind)] for sq in forks[:2])
        return StoryEvent("tactic", "fork", title, f"{subtitle} · {names}难以兼顾", piece,
                          lines=tuple((move.end, sq) for sq in forks), targets=tuple(forks), cast=(move.end,))
    opened = [(src, dst) for src, dst in _unmasked(before, after, move, side) if _real_threat(after, src, dst)]
    if opened:
        title, subtitle = TACTICS["discovered_attack"]
        return StoryEvent("tactic", "discovered_attack", title, subtitle, piece, lines=tuple(opened),
                          targets=tuple(dst for _, dst in opened), cast=tuple(dict.fromkeys([src for src, _ in opened] + [move.end])))
    return None


def capture_event(piece: Piece, victim: Piece, tone: str = "even", tag: str = "") -> StoryEvent:
    key = piece.kind + victim.kind
    title, attacker_line, victim_line = CAPTURES[key]
    subtitle = f"{KIND_LABEL[(piece.side, piece.kind)]} 吃 {KIND_LABEL[(victim.side, victim.kind)]}"
    return StoryEvent("capture", key, title, subtitle, piece, victim, attacker_line, victim_line, tone=tone, tag=tag)


def opening_event(key: str, piece: Piece) -> StoryEvent:
    title, subtitle = OPENINGS[key]
    if piece.side == BLACK:
        title = {"兵底炮": "卒底炮"}.get(title, title)
    return StoryEvent("opening", key, title, f"{_side_name(piece.side)} · {subtitle}", piece)


def mate_event(key: str, piece: Piece, finale: bool, game: Optional[XiangqiGame] = None) -> StoryEvent:
    title, subtitle = MATES[key]
    kind = "mate" if finale else "check"
    if not finale:
        title = "将  军" if key == "generic" else f"{title} · 将"
    lines: tuple = ()
    targets: tuple = ()
    cast: tuple = ()
    if game is not None:
        king = game.king_position(opponent(piece.side))
        if king is not None:
            cast = pattern_cast(game, key, piece.side)
            checkers = [pos for pos, _ in _attackers(game, piece.side, king)]
            lines = tuple((pos, king) for pos in checkers)
            targets = (king,)
    return StoryEvent(kind, key, title, subtitle, piece, lines=lines, targets=targets, cast=cast)


def _before(game: XiangqiGame) -> Optional[XiangqiGame]:
    if not game.history:
        return None
    snap = game.history[-1]
    previous = XiangqiGame(snap.board, snap.turn)
    previous.halfmove = snap.halfmove
    return previous


def analyze_move(game: XiangqiGame, move: Move, piece: Piece) -> list[StoryEvent]:
    """Story beats for ``move`` (already pushed onto ``game``), in playback order."""
    events: list[StoryEvent] = []
    before = _before(game)
    opening = detect_opening(game, move, piece)
    if opening:
        events.append(opening_event(opening, piece))
    if move.captured is not None:
        balance_before = material_balance(before.board) if before else material_balance(game.board)
        balance_after = material_balance(game.board)
        events.append(capture_event(
            piece, move.captured, tone_for(balance_before, piece.side),
            capture_tag(balance_before, balance_after, piece.side),
        ))
    defender = opponent(piece.side)
    if game.winner == piece.side and game.reason == "困毙":
        title, subtitle = STALEMATE
        king = game.king_position(defender)
        events.append(StoryEvent("stalemate", "stalemate", title, subtitle, piece, targets=(king,) if king else ()))
        return events
    if before is not None:
        tactic = detect_tactic(before, game, move, piece)
        if tactic is not None:
            events.append(tactic)
    in_check = game.winner == piece.side or game.is_in_check(defender)
    if in_check:
        key = classify_attack_pattern(game, move, piece)
        events.append(mate_event(key, piece, finale=game.winner == piece.side, game=game))
    elif (
        piece.kind == ROOK and move.captured == Piece(defender, ADVISOR)
        and depth_of(move.end[0], defender) == 1 and move.end[1] == 4
    ):
        title, subtitle = MATES["heart_strike"]
        events.append(StoryEvent("sacrifice", "heart_strike", title, "车入宫心 · 剜去护卫", piece))
    return events
