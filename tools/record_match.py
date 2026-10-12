"""Record an AI-vs-AI match (or the on-board story gallery) to an MP4 using a virtual clock.

Examples:
    python tools/record_match.py --out match.mp4 --seconds 90
    python tools/record_match.py --gallery --out gallery.mp4 --seconds 120
    python tools/record_match.py --fen "<fen>" --story highlights --out endgame.mp4

Requires ffmpeg on PATH. Runs headless (SDL dummy driver); frames are rendered at a fixed
time step so the video plays at real speed regardless of how fast this machine is.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

_clock = {"ms": 0}
pygame.time.get_ticks = lambda: _clock["ms"]  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="match.mp4")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--story", choices=("full", "highlights", "off"), default="full")
    parser.add_argument("--fen", default="", help="start from a custom position instead of the opening")
    parser.add_argument("--gallery", action="store_true", help="record the story gallery instead of a match")
    parser.add_argument("--gallery-start", type=int, default=0)
    parser.add_argument("--gallery-keys", default="", help="comma separated category/key demos to play in order")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--frames-dir", default="", help="also dump every Nth frame as PNG into this directory")
    parser.add_argument("--frames-every", type=int, default=15)
    args = parser.parse_args()

    from app import XiangqiApp
    from game_engine import XiangqiGame

    app = XiangqiApp()
    app.story_mode = args.story
    app.ai.depth = args.depth
    playlist: list[int] = []
    if args.gallery_keys:
        from cinema.demos import catalogue

        index = {(d.category, d.key): i for i, d in enumerate(catalogue())}
        playlist = [index[tuple(item.split("/"))] for item in args.gallery_keys.split(",")]
        args.gallery = True
    if args.gallery:
        app.open_gallery(playlist[0] if playlist else args.gallery_start)
    else:
        app.opponent_mode = "watch"
        app.new_game()
        if args.fen:
            app.game = XiangqiGame.from_fen(args.fen)
    height = int(args.width * 900 / 1440) // 2 * 2
    width = args.width // 2 * 2
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast", args.out],
        stdin=subprocess.PIPE,
    )
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
    step = 1000.0 / args.fps
    frames = int(args.seconds * args.fps)
    started = time.time()
    for index in range(frames):
        _clock["ms"] = int(index * step) + 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        if playlist:
            shown = app.gallery_index
            app.update_story()
            if app.gallery_index != shown:
                playlist.pop(0)
                if not playlist:
                    break
                app.open_gallery(playlist[0])
        else:
            app.update_story()
        app.update_ai()
        if app.ai_future is not None:
            app.ai_future.result()
        app.draw()
        frame = app.canvas if width == 1440 else pygame.transform.smoothscale(app.canvas, (width, height))
        ffmpeg.stdin.write(pygame.image.tostring(frame, "RGB"))
        if args.frames_dir and index % args.frames_every == 0:
            pygame.image.save(app.canvas, os.path.join(args.frames_dir, f"frame_{index:05d}.png"))
        if app.game.winner and not app.busy and not args.gallery and index > 0:
            linger = getattr(app, "_linger", None)
            if linger is None:
                app._linger = index
            elif index - linger > args.fps * 4:
                break
    ffmpeg.stdin.close()
    ffmpeg.wait()
    elapsed = time.time() - started
    print(f"wrote {args.out}: {index + 1} frames in {elapsed:.1f}s ({(index + 1) / max(elapsed, 1e-6):.1f} fps render)")
    app.executor.shutdown(wait=False, cancel_futures=True)
    pygame.quit()


if __name__ == "__main__":
    main()
