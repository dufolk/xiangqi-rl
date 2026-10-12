"""CJK font lookup shared by the GUI and the cinema renderer."""

from __future__ import annotations

import os
import sys

import pygame

_CANDIDATES = {
    "darwin": ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"],
    "win32": [
        os.path.join(os.environ.get("SystemRoot", "C:/Windows"), "Fonts", "msyhbd.ttc"),
        os.path.join(os.environ.get("SystemRoot", "C:/Windows"), "Fonts", "msyh.ttc"),
    ],
}.get(sys.platform, []) + [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]

_cache: dict[tuple[int, bool], pygame.font.Font] = {}


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key in _cache:
        return _cache[key]
    font = None
    for path in _CANDIDATES:
        if os.path.exists(path):
            font = pygame.font.Font(path, size)
            break
    if font is None:
        font = pygame.font.SysFont("Arial Unicode MS,PingFang SC,Microsoft YaHei", size)
    font.set_bold(bold)
    _cache[key] = font
    return font


def font(size: float, bold: bool = True) -> pygame.font.Font:
    """Size-bucketed font so per-frame scaling doesn't create hundreds of font objects."""
    bucket = max(8, int(round(size / 2.0)) * 2)
    return load_font(bucket, bold)
