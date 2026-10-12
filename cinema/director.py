"""Turns story events into a playlist of cutscenes."""

from __future__ import annotations

from typing import Optional

from cinema.detect import StoryEvent
from cinema.stage import Cutscene


def scene_for_event(event: StoryEvent) -> Optional[Cutscene]:
    if event.kind == "capture":
        from cinema.scenes_capture import CaptureCutscene

        return CaptureCutscene(event)
    if event.kind == "opening":
        from cinema.scenes_opening import opening_scene

        return opening_scene(event)
    if event.kind in ("mate", "check", "sacrifice", "stalemate"):
        from cinema.scenes_mate import mate_scene

        return mate_scene(event)
    return None
