import attr
from pydrake.trajectories import PiecewisePolynomial

from common.custom_types import XYPoint


@attr.frozen
class NaiveCOMPlanner:

    com_height_m: float

    def plan_zmp_trajectory(
        self,
        cop_trajectory: PiecewisePolynomial,
        initial_zmp_xy: XYPoint,
    ) -> None:
        ...
