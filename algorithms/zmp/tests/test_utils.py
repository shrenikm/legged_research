from algorithms.zmp.utils import FootstepType, WalkPhase
from common.testing_utils import execute_pytest_file


def test_footstep_type() -> None:

    assert FootstepType.LEFT.invert() == FootstepType.RIGHT
    assert FootstepType.RIGHT.invert() == FootstepType.LEFT


def test_walk_phase() -> None:

    assert WalkPhase.SWING.invert() == WalkPhase.STANCE
    assert WalkPhase.STANCE.invert() == WalkPhase.SWING


if __name__ == "__main__":
    execute_pytest_file()
