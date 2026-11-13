# -*- coding: utf-8 -*-
"""
core/map.py
===========
全局/活跃子地图容器类 (Map)
严格 1:1 对齐 ORB-SLAM3 C++ Map.h / Map.cc。
维护所有活跃 KeyFrame、MapPoint 以及当前局部参考地图点 (reference_mappoints)。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .keyframe import KeyFrame
    from .mappoint import MapPoint


class Map:
    _next_id: int = 0

    __slots__ = (
        "id", "keyframes", "mappoints",
        "reference_mappoints", "max_kf_id",
        "big_change_idx"
    )

    def __init__(self):
        self.id = Map._next_id
        Map._next_id += 1

        self.keyframes: set[KeyFrame] = set()
        self.mappoints: set[MapPoint] = set()
        self.reference_mappoints: list[MapPoint] = []
        self.max_kf_id: int = 0
        self.big_change_idx: int = 0

    def add_keyframe(self, kf: KeyFrame) -> None:
        self.keyframes.add(kf)
        if kf.id > self.max_kf_id:
            self.max_kf_id = kf.id

    def add_mappoint(self, mp: MapPoint) -> None:
        self.mappoints.add(mp)

    def erase_mappoint(self, mp: MapPoint) -> None:
        self.mappoints.discard(mp)

    def erase_keyframe(self, kf: KeyFrame) -> None:
        self.keyframes.discard(kf)

    def set_reference_mappoints(self, mps: list[MapPoint]) -> None:
        self.reference_mappoints = list(mps)

    def inform_new_big_change(self) -> None:
        self.big_change_idx += 1

    def get_all_keyframes(self) -> list[KeyFrame]:
        return sorted(
            [kf for kf in self.keyframes if not kf.is_bad],
            key=lambda k: k.id
        )

    def get_all_mappoints(self) -> list[MapPoint]:
        return sorted(
            [mp for mp in self.mappoints if not mp.is_bad],
            key=lambda m: m.id
        )

    def num_keyframes(self) -> int:
        return len(self.keyframes)

    def num_mappoints(self) -> int:
        return len(self.mappoints)

    def clear(self) -> None:
        self.keyframes.clear()
        self.mappoints.clear()
        self.reference_mappoints.clear()
        self.max_kf_id = 0
        self.big_change_idx = 0
