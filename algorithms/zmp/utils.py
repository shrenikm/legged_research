from __future__ import annotations

from enum import Enum, auto


class FootstepType(Enum):
    LEFT = auto()
    RIGHT = auto()

    def invert(self) -> FootstepType:
        if self == FootstepType.LEFT:
            return FootstepType.RIGHT
        elif self == FootstepType.RIGHT:
            return FootstepType.LEFT
        else:
            raise NotImplementedError


class WalkPhase(Enum):
    SWING = auto()
    STANCE = auto()

    def invert(self) -> WalkPhase:
        if self == WalkPhase.SWING:
            return WalkPhase.STANCE
        elif self == WalkPhase.STANCE:
            return WalkPhase.SWING
        else:
            raise NotImplementedError
