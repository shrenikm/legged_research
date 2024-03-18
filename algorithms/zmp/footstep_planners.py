from enum import Enum, auto

import attr
from pydrake.trajectories import PiecewisePolynomial

from common.custom_types import PolygonArray, XYPath


class FootstepType(Enum):
    LEFT = auto()
    RIGHT = auto()


@attr.frozen
class NaiveFootstepPlanner:
    """
    Naive footstep planner that tries to track the given XY trajectory with
    footsteps on either side of the trajectory.
    As the name suggests, it utilizes a naive heuristic way of determining
    the footsteps without taking into account the robot kinematics or dynamics.
    """

    com_z_h: float
    left_foot_polygon: PolygonArray
    right_foot_polygon: PolygonArray

    def plan_cop_trajectory(
        self,
        xy_path: XYPath,
        stride_length_m: float,
        swing_phase_time_s: float,
        stance_phase_time_s: float,
        first_footstep: FootstepType = FootstepType.RIGHT,
    ) -> PiecewisePolynomial:
        assert xy_path.shape[1] == 2

        # Assuming that initially, both feet are on either side of the first
        # point on the xy path. COP here is  ~ assumed to be equal to this value.
        breaks = [0.0]
        samples = [xy_path[0]]

        is_first_step = True
        next_swing_phase_footstep = first_footstep

        return PiecewisePolynomial.FirstOrderHold(
            breaks=breaks,
            samples=samples,
        )
