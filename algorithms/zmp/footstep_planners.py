from __future__ import annotations

from enum import Enum, auto

import attr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pydrake.math import RotationMatrix
from pydrake.trajectories import PiecewisePolynomial

from common.custom_types import PolygonArray, XYPath
from numeric.geometry.path_utils import (
    compute_oriented_xy_path,
    compute_xytheta_side_poses,
    normalize_angles,
    segment_path_index,
    segment_path_indices,
)


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


def _plot_cop_trajectory_on_ax(
    ax: Axes,
    cop_trajectory: PiecewisePolynomial,
    xy_path: XYPath,
    left_foot_polygon: PolygonArray,
    right_foot_polygon: PolygonArray,
) -> None:
    ax.set_xlabel("x (m)")
    ax.set_xlabel("y (m)")
    ax.set_xlim(
        np.min(xy_path[:, 0]).item() - 0.5,
        np.max(xy_path[:, 0]).item() + 0.5,
    )
    ax.set_ylim(
        np.min(xy_path[:, 1]).item() - 0.5,
        np.max(xy_path[:, 1]).item() + 0.5,
    )
    ax.set_aspect(1.0)

    # Plot xy_path
    ax.plot(xy_path[:, 0], xy_path[:, 1], linestyle="--", color="green")

    cop_poses = np.empty((0, 3), dtype=np.float64)

    for t in cop_trajectory.get_segment_times():
        cop_poses = np.vstack((cop_poses, cop_trajectory.value(t).reshape(3)))
        footstep_pose = cop_trajectory.value(t)
        rot_mat = RotationMatrix.MakeXRotation(
            theta=footstep_pose[2].item(),
        ).matrix()[1:, 1:]

        lf = rot_mat @ left_foot_polygon.T
        lf += footstep_pose[:2]
        rf = rot_mat @ right_foot_polygon.T
        rf += footstep_pose[:2]

        ax.add_patch(
            patches.Polygon(xy=lf.T, fill=False, color="brown"),
        )
        ax.add_patch(
            patches.Polygon(xy=lf.T, fill=False, color="brown"),
        )

    # Plot COP poses.
    ax.plot(cop_poses[:, 0], cop_poses[:, 1], color="cornflowerblue")


@attr.frozen
class NaiveFootstepPlanner:
    """
    Naive footstep planner that tries to track the given XY trajectory with
    footsteps on either side of the trajectory.
    As the name suggests, it utilizes a naive heuristic way of determining
    the footsteps without taking into account the robot kinematics or dynamics.
    """

    com_height_m: float
    distance_between_feet: float
    max_orientation_delta: float
    left_foot_polygon: PolygonArray
    right_foot_polygon: PolygonArray

    def plan_cop_trajectory(
        self,
        xy_path: XYPath,
        stride_length_m: float,
        swing_phase_time_s: float,
        stance_phase_time_s: float,
        first_footstep: FootstepType = FootstepType.RIGHT,
        debug: bool = False,
    ) -> PiecewisePolynomial:
        assert xy_path.shape[1] == 2

        xytheta_path = compute_oriented_xy_path(
            xy_path=xy_path,
        )

        # Assuming that initially, both feet are on either side of the first
        # point on the xy path. COP here is  ~ assumed to be equal to this value.
        breaks = [0.0]
        samples = np.copy(xytheta_path[0]).reshape(3, 1)

        # First we shift the COP to the non first step leg.
        breaks.append(stance_phase_time_s)
        left_xytheta_pose, right_xytheta_pose = compute_xytheta_side_poses(
            xytheta_pose=xytheta_path[0],
            half_distance_m=0.5 * self.distance_between_feet,
        )
        if first_footstep == FootstepType.LEFT:
            samples = np.hstack((samples, right_xytheta_pose.reshape(3, 1)))
        elif first_footstep == FootstepType.RIGHT:
            samples = np.hstack((samples, left_xytheta_pose.reshape(3, 1)))
        else:
            raise NotImplementedError

        # Taking the first half step.
        first_footstep_index = segment_path_index(
            xy_path=xy_path,
            segment_length=0.5 * stride_length_m,
            start_index=0,
        )

        left_xytheta_pose, right_xytheta_pose = compute_xytheta_side_poses(
            xytheta_pose=xytheta_path[first_footstep_index],
            half_distance_m=0.5 * self.distance_between_feet,
        )
        breaks.append(breaks[-1] + stance_phase_time_s)
        if first_footstep == FootstepType.LEFT:
            samples = np.hstack((samples, left_xytheta_pose.reshape(3, 1)))
        elif first_footstep == FootstepType.RIGHT:
            samples = np.hstack((samples, right_xytheta_pose.reshape(3, 1)))
        else:
            raise NotImplementedError

        footstep_indices = segment_path_indices(
            xy_path=xy_path,
            segment_length=stride_length_m,
            start_index=first_footstep_index,
        )

        next_footstep = first_footstep.invert()
        for footstep_index in footstep_indices:

            (
                next_left_xytheta_pose,
                next_right_xytheta_pose,
            ) = compute_xytheta_side_poses(
                xytheta_pose=xytheta_path[footstep_index],
                half_distance_m=0.5 * self.distance_between_feet,
            )

            # Clamping theta to be within limits so that the feet orientation
            # doesn't change too much.
            next_theta = xytheta_path[footstep_index, 2]
            if (
                normalize_angles(next_theta - samples[2, -1])
                > self.max_orientation_delta
            ):
                next_theta = samples[2, -1] + np.clip(
                    next_theta - samples[2, -1],
                    -self.max_orientation_delta,
                    self.max_orientation_delta,
                )

            breaks.append(breaks[-1] + swing_phase_time_s + stance_phase_time_s)
            if next_footstep == FootstepType.LEFT:
                pose = np.copy(next_left_xytheta_pose)
                pose[2] = next_theta
                samples = np.hstack((samples, pose.reshape(3, 1)))
            elif next_footstep == FootstepType.RIGHT:
                pose = np.copy(next_right_xytheta_pose)
                pose[2] = next_theta
                samples = np.hstack((samples, pose.reshape(3, 1)))
            else:
                raise NotImplementedError

            next_footstep = next_footstep.invert()

        cop_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=breaks,
            samples=samples,
        )

        if debug:
            fig = plt.figure("Naive Footstep Trajectory")
            ax = fig.gca()
            _plot_cop_trajectory_on_ax(
                ax=ax,
                cop_trajectory=cop_trajectory,
                xy_path=xy_path,
                left_foot_polygon=self.left_foot_polygon,
                right_foot_polygon=self.right_foot_polygon,
            )
            plt.show()

        return cop_trajectory
