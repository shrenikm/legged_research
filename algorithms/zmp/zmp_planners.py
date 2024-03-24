from __future__ import annotations

from enum import Enum, auto
from functools import partial
from typing import Optional, Tuple

import attr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pydrake.math import RotationMatrix
from pydrake.trajectories import PiecewisePolynomial
from scipy.linalg import solve_discrete_are

from common.attr_utils import AttrsValidators
from common.constants import ACC_DUE_TO_GRAVITY
from common.custom_types import (
    GainsVector,
    NpArrayNNf64,
    NpVectorNf64,
    PolygonArray,
    XYPath,
    XYThetaPose,
    XYZPoint,
    XYZThetaPose,
)
from numeric.geometry.path_utils import (
    compute_oriented_xy_path,
    compute_xytheta_side_poses,
    normalize_angles,
    segment_path_index,
    segment_path_indices,
)


def _plot_zmp_trajectory_on_ax(
    ax: Axes,
    left_foot_trajectory: PiecewisePolynomial,
    right_foot_trajectory: PiecewisePolynomial,
    zmp_trajectory: PiecewisePolynomial,
    xy_path: XYPath,
    left_foot_polygon: PolygonArray,
    right_foot_polygon: PolygonArray,
    initialize_axes: bool = True,
) -> None:
    if initialize_axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
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

    zmp_poses = np.empty((0, 2), dtype=np.float64)

    for t in zmp_trajectory.get_segment_times():
        zmp_poses = np.vstack((zmp_poses, zmp_trajectory.value(t).reshape(2)))

        left_foot_pose = left_foot_trajectory.value(t)
        right_foot_pose = right_foot_trajectory.value(t)
        theta = left_foot_pose[3, 0]

        rot_mat = RotationMatrix.MakeXRotation(
            theta=theta,
        ).matrix()[1:, 1:]

        lf = rot_mat @ left_foot_polygon.T
        lf += left_foot_pose[:2]
        rf = rot_mat @ right_foot_polygon.T
        rf += right_foot_pose[:2]

        ax.add_patch(
            patches.Polygon(xy=lf.T, fill=False, color="rosybrown"),
        )
        ax.add_patch(
            patches.Polygon(xy=rf.T, fill=False, color="brown"),
        )

    # Plot ZMP/COP poses.
    ax.plot(
        zmp_poses[:, 0],
        zmp_poses[:, 1],
        color="cornflowerblue",
        label="ZMP trajectory",
    )
    ax.legend(loc="upper right")


def _plot_foot_trajectory_on_ax(
    ax1: Axes,
    ax2: Axes,
    left_foot_trajectory: PiecewisePolynomial,
    right_foot_trajectory: PiecewisePolynomial,
    initialize_axes: bool = True,
) -> None:
    """
    Plots the x/y and height trajectories of the left and right feet
    on ax1 and ax2 respectively.
    """
    if initialize_axes:
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_aspect(1.0)

        ax2.set_xlabel("t (x)")
        ax2.set_ylabel("z (m)")

    sample_time = 0.1
    sample_times = [
        i * sample_time
        for i in range(int(left_foot_trajectory.end_time() / sample_time))
    ]
    lft = left_foot_trajectory.vector_values(sample_times)
    rft = right_foot_trajectory.vector_values(sample_times)

    ax1.plot(lft[0, :], lft[1, :], label="left foot", color="coral")
    ax1.plot(rft[0, :], rft[1, :], label="right foot", color="royalblue")

    ax2.plot(sample_times, lft[2, :], label="left z", color="coral")
    ax2.plot(sample_times, rft[2, :], label="right z", color="royalblue")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")


def _plot_com_trajectory_on_ax(
    ax: Axes,
    ax1: Axes,
    ax2: Axes,
    zmp_output_trajectory: PiecewisePolynomial,
    com_trajectory: PiecewisePolynomial,
    initialize_axes: bool = True,
) -> None:
    """
    Plots the com trajectory on the axis ax.
    Plots the x/y trajectories in ax2/ax3.
    """
    if initialize_axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect(1.0)

    com_poses = np.empty((0, 3), dtype=np.float64)
    zmp_output_poses = np.empty((0, 2), dtype=np.float64)

    t = 0.0
    sample_time = 0.1
    sample_times = [
        i * sample_time for i in range(int(com_trajectory.end_time() / sample_time))
    ]
    for t in sample_times:
        zmp_output_poses = np.vstack(
            (
                zmp_output_poses,
                zmp_output_trajectory.value(t).reshape(2),
            )
        )
        com_poses = np.vstack((com_poses, com_trajectory.value(t).reshape(3)))

    # Plot ZMP output poses
    ax.plot(
        zmp_output_poses[:, 0],
        zmp_output_poses[:, 1],
        color="olive",
        linestyle="dotted",
        label="ZMP output trajectory",
    )

    # Plot COM poses (Just x and y).
    ax.plot(
        com_poses[:, 0],
        com_poses[:, 1],
        color="lightcoral",
        label="COM trajectory",
    )

    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("x (m)")
    ax1.plot(
        sample_times,
        com_poses[:, 0],
        color="mediumslateblue",
        label="COM x coordinates",
    )
    ax1.plot(
        sample_times,
        zmp_output_poses[:, 0],
        color="olive",
        linestyle="dotted",
        label="ZMP x coordinates",
    )
    ax1.legend(loc="upper right")

    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("y (m)")
    ax2.plot(
        sample_times,
        com_poses[:, 1],
        color="mediumslateblue",
        label="COM y coordinates",
    )
    ax2.plot(
        sample_times,
        zmp_output_poses[:, 1],
        color="olive",
        linestyle="dotted",
        label="ZMP y coordinates",
    )
    ax2.legend(loc="upper right")

    ax.legend(loc="upper right")


def _xytheta_pose_to_xyztheta_pose(
    xytheta_pose: XYThetaPose,
    z: float,
) -> XYZThetaPose:
    xyztheta_pose = np.zeros(4, dtype=np.float64)
    xyztheta_pose[:2] = xytheta_pose[:2]
    xyztheta_pose[2] = z
    xyztheta_pose[3] = xytheta_pose[2]

    return xyztheta_pose


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


@attr.frozen
class ZMPPlannerResult:
    left_foot_trajectory: PiecewisePolynomial
    right_foot_trajectory: PiecewisePolynomial
    zmp_trajectory: PiecewisePolynomial
    zmp_output_trajectory: PiecewisePolynomial
    com_trajectory: PiecewisePolynomial


@attr.frozen
class NaiveZMPPlanner:
    """
    Naive ZMP planner that tries to track the given XY trajectory with
    footsteps on either side of the trajectory.
    As the name suggests, it utilizes a naive heuristic way of determining
    the footsteps without taking into account the robot kinematics or dynamics.
    It then computes a COP/ZMP trajectory from the footstep poses.
    """

    stride_length_m: float = attr.ib(
        validator=AttrsValidators.positive_validator(),
    )
    foot_lift_height_m: float = attr.ib(
        validator=AttrsValidators.positive_validator(),
    )
    default_foot_height_m: float
    swing_phase_time_s: float = attr.ib(
        validator=AttrsValidators.positive_validator(),
    )
    stance_phase_time_s: float = attr.ib(
        validator=AttrsValidators.positive_validator(),
    )
    distance_between_feet: float = attr.ib(
        validator=AttrsValidators.positive_validator(),
    )
    max_orientation_delta: float = attr.ib(
        validator=AttrsValidators.positive_validator(),
    )
    left_foot_polygon: PolygonArray
    right_foot_polygon: PolygonArray

    preview_time_s: float = attr.ib(
        validator=AttrsValidators.positive_validator(),
    )
    dt: float = attr.ib(
        default=0.001,
        validator=AttrsValidators.positive_validator(),
    )
    g: float = attr.ib(init=False, default=ACC_DUE_TO_GRAVITY)

    def plan_zmp_trajectory(
        self,
        xy_path: XYPath,
        first_footstep: FootstepType = FootstepType.RIGHT,
        debug: bool = False,
    ) -> Tuple[PiecewisePolynomial, PiecewisePolynomial, PiecewisePolynomial]:
        assert xy_path.shape[1] == 2

        f_xyzt = partial(_xytheta_pose_to_xyztheta_pose, z=self.default_foot_height_m)

        xytheta_path = compute_oriented_xy_path(
            xy_path=xy_path,
        )

        # Initial foot poses.
        left_xytheta_pose, right_xytheta_pose = compute_xytheta_side_poses(
            xytheta_pose=xytheta_path[0],
            half_distance_m=0.5 * self.distance_between_feet,
        )
        left_xyztheta_pose = f_xyzt(left_xytheta_pose)
        right_xyztheta_pose = f_xyzt(right_xytheta_pose)

        # Assuming that initially, both feet are on either side of the first
        # point on the xy path. COP here is  ~ assumed to be equal to this value.
        left_foot_breaks, right_foot_breaks, zmp_breaks = [0.0], [0.0], [0.0]
        left_foot_samples = np.copy(left_xyztheta_pose).reshape(4, 1)
        right_foot_samples = np.copy(right_xyztheta_pose).reshape(4, 1)
        zmp_samples = np.copy(xy_path[0]).reshape(2, 1)

        # First we shift the COP to the non first step leg. Feet are still placed
        # on the same positions on the ground.
        left_foot_breaks.append(self.stance_phase_time_s)
        right_foot_breaks.append(self.stance_phase_time_s)
        zmp_breaks.append(self.stance_phase_time_s)

        left_foot_samples = np.hstack(
            (left_foot_samples, left_foot_samples[:, -1].reshape(4, 1))
        )
        right_foot_samples = np.hstack(
            (right_foot_samples, right_foot_samples[:, -1].reshape(4, 1))
        )

        if first_footstep == FootstepType.LEFT:
            zmp_samples = np.hstack((zmp_samples, right_xytheta_pose[:2].reshape(2, 1)))
        elif first_footstep == FootstepType.RIGHT:
            zmp_samples = np.hstack((zmp_samples, left_xytheta_pose[:2].reshape(2, 1)))
        else:
            raise NotImplementedError

        # Taking the first half step.
        first_footstep_index = segment_path_index(
            xy_path=xy_path,
            segment_length=0.5 * self.stride_length_m,
            start_index=0,
        )

        left_xytheta_pose, right_xytheta_pose = compute_xytheta_side_poses(
            xytheta_pose=xytheta_path[first_footstep_index],
            half_distance_m=0.5 * self.distance_between_feet,
        )
        left_xyztheta_pose = f_xyzt(left_xytheta_pose)
        right_xyztheta_pose = f_xyzt(right_xytheta_pose)

        # Mid swing feet positions.
        left_foot_breaks.append(left_foot_breaks[-1] + 0.5 * self.swing_phase_time_s)
        right_foot_breaks.append(right_foot_breaks[-1] + 0.5 * self.swing_phase_time_s)
        if first_footstep == FootstepType.LEFT:
            # Lifted up pose at the mid point.
            _foot_pose = 0.5 * (left_xyztheta_pose + left_foot_samples[:, -1])
            _foot_pose[2] = self.foot_lift_height_m
            left_foot_samples = np.hstack((left_foot_samples, _foot_pose.reshape(4, 1)))
            right_foot_samples = np.hstack(
                (right_foot_samples, right_foot_samples[:, -1].reshape(4, 1))
            )
        elif first_footstep == FootstepType.RIGHT:
            # Lifted up pose at the mid point.
            _foot_pose = 0.5 * (right_xyztheta_pose + right_foot_samples[:, -1])
            _foot_pose[2] = self.foot_lift_height_m
            right_foot_samples = np.hstack(
                (right_foot_samples, _foot_pose.reshape(4, 1))
            )
            left_foot_samples = np.hstack(
                (left_foot_samples, left_foot_samples[:, -1].reshape(4, 1))
            )
        else:
            raise NotImplementedError

        left_foot_breaks.append(left_foot_breaks[-1] + 0.5 * self.swing_phase_time_s)
        right_foot_breaks.append(right_foot_breaks[-1] + 0.5 * self.swing_phase_time_s)
        zmp_breaks.append(zmp_breaks[-1] + self.swing_phase_time_s)

        if first_footstep == FootstepType.LEFT:
            left_foot_samples = np.hstack(
                (left_foot_samples, left_xyztheta_pose.reshape(4, 1))
            )
            right_foot_samples = np.hstack(
                (right_foot_samples, right_foot_samples[:, -1].reshape(4, 1))
            )
        elif first_footstep == FootstepType.RIGHT:
            right_foot_samples = np.hstack(
                (right_foot_samples, right_xyztheta_pose.reshape(4, 1))
            )
            left_foot_samples = np.hstack(
                (left_foot_samples, left_foot_samples[:, -1].reshape(4, 1))
            )
        else:
            raise NotImplementedError
        zmp_samples = np.hstack((zmp_samples, zmp_samples[:, -1].reshape(2, 1)))

        left_foot_breaks.append(left_foot_breaks[-1] + self.stance_phase_time_s)
        right_foot_breaks.append(right_foot_breaks[-1] + self.stance_phase_time_s)
        zmp_breaks.append(zmp_breaks[-1] + self.stance_phase_time_s)

        left_foot_samples = np.hstack(
            (left_foot_samples, left_foot_samples[:, -1].reshape(4, 1))
        )
        right_foot_samples = np.hstack(
            (right_foot_samples, right_foot_samples[:, -1].reshape(4, 1))
        )
        if first_footstep == FootstepType.LEFT:
            zmp_samples = np.hstack((zmp_samples, left_xytheta_pose[:2].reshape(2, 1)))
        elif first_footstep == FootstepType.RIGHT:
            zmp_samples = np.hstack((zmp_samples, right_xytheta_pose[:2].reshape(2, 1)))
        else:
            raise NotImplementedError

        footstep_indices = segment_path_indices(
            xy_path=xy_path,
            segment_length=self.stride_length_m,
            start_index=first_footstep_index,
        )

        next_footstep = first_footstep.invert()
        for footstep_index in footstep_indices:

            (left_xytheta_pose, right_xytheta_pose,) = compute_xytheta_side_poses(
                xytheta_pose=xytheta_path[footstep_index],
                half_distance_m=0.5 * self.distance_between_feet,
            )
            left_xyztheta_pose = f_xyzt(left_xytheta_pose)
            right_xyztheta_pose = f_xyzt(right_xytheta_pose)

            # Clamping theta to be within limits so that the feet orientation
            # doesn't change too much.
            # For the previous theta, we can use either left or right foot samples.
            theta = xytheta_path[footstep_index, 2]
            prev_theta = left_foot_samples[3, -1]
            if normalize_angles(theta - prev_theta) > self.max_orientation_delta:
                theta = prev_theta + np.clip(
                    theta - prev_theta,
                    -self.max_orientation_delta,
                    self.max_orientation_delta,
                )
            left_xyztheta_pose[3] = theta
            right_xyztheta_pose[3] = theta

            # Mid swing feet positions.
            left_foot_breaks.append(
                left_foot_breaks[-1] + 0.5 * self.swing_phase_time_s
            )
            right_foot_breaks.append(
                right_foot_breaks[-1] + 0.5 * self.swing_phase_time_s
            )
            if next_footstep == FootstepType.LEFT:
                # Lifted up pose at the mid point.
                _foot_pose = 0.5 * (left_xyztheta_pose + left_foot_samples[:, -1])
                _foot_pose[2] = self.foot_lift_height_m
                left_foot_samples = np.hstack(
                    (left_foot_samples, _foot_pose.reshape(4, 1))
                )
                right_foot_samples = np.hstack(
                    (right_foot_samples, right_foot_samples[:, -1].reshape(4, 1))
                )
            elif next_footstep == FootstepType.RIGHT:
                # Lifted up pose at the mid point.
                _foot_pose = 0.5 * (right_xyztheta_pose + right_foot_samples[:, -1])
                _foot_pose[2] = self.foot_lift_height_m
                right_foot_samples = np.hstack(
                    (right_foot_samples, _foot_pose.reshape(4, 1))
                )
                left_foot_samples = np.hstack(
                    (left_foot_samples, left_foot_samples[:, -1].reshape(4, 1))
                )
            else:
                raise NotImplementedError

            left_foot_breaks.append(
                left_foot_breaks[-1] + 0.5 * self.swing_phase_time_s
            )
            right_foot_breaks.append(
                right_foot_breaks[-1] + 0.5 * self.swing_phase_time_s
            )
            zmp_breaks.append(zmp_breaks[-1] + self.swing_phase_time_s)

            if next_footstep == FootstepType.LEFT:
                left_foot_samples = np.hstack(
                    (left_foot_samples, left_xyztheta_pose.reshape(4, 1))
                )
                right_foot_samples = np.hstack(
                    (right_foot_samples, right_foot_samples[:, -1].reshape(4, 1))
                )
            elif next_footstep == FootstepType.RIGHT:
                right_foot_samples = np.hstack(
                    (right_foot_samples, right_xyztheta_pose.reshape(4, 1))
                )
                left_foot_samples = np.hstack(
                    (left_foot_samples, left_foot_samples[:, -1].reshape(4, 1))
                )
            else:
                raise NotImplementedError
            zmp_samples = np.hstack((zmp_samples, zmp_samples[:, -1].reshape(2, 1)))

            left_foot_breaks.append(left_foot_breaks[-1] + self.stance_phase_time_s)
            right_foot_breaks.append(right_foot_breaks[-1] + self.stance_phase_time_s)
            zmp_breaks.append(zmp_breaks[-1] + self.stance_phase_time_s)

            left_foot_samples = np.hstack(
                (left_foot_samples, left_foot_samples[:, -1].reshape(4, 1))
            )
            right_foot_samples = np.hstack(
                (right_foot_samples, right_foot_samples[:, -1].reshape(4, 1))
            )
            if next_footstep == FootstepType.LEFT:
                zmp_samples = np.hstack(
                    (zmp_samples, left_xyztheta_pose[:2].reshape(2, 1))
                )
            elif next_footstep == FootstepType.RIGHT:
                zmp_samples = np.hstack(
                    (zmp_samples, right_xyztheta_pose[:2].reshape(2, 1))
                )
            else:
                raise NotImplementedError

            next_footstep = next_footstep.invert()

        left_foot_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=left_foot_breaks,
            samples=left_foot_samples,
        )
        right_foot_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=right_foot_breaks,
            samples=right_foot_samples,
        )
        zmp_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=zmp_breaks,
            samples=zmp_samples,
        )

        if debug:
            fig = plt.figure("Naive Footstep Trajectory")
            ax = fig.gca()

            _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            _plot_zmp_trajectory_on_ax(
                ax=ax,
                left_foot_trajectory=left_foot_trajectory,
                right_foot_trajectory=right_foot_trajectory,
                zmp_trajectory=zmp_trajectory,
                xy_path=xy_path,
                left_foot_polygon=self.left_foot_polygon,
                right_foot_polygon=self.right_foot_polygon,
                initialize_axes=True,
            )
            _plot_foot_trajectory_on_ax(
                ax1=ax1,
                ax2=ax2,
                left_foot_trajectory=left_foot_trajectory,
                right_foot_trajectory=right_foot_trajectory,
                initialize_axes=True,
            )
            plt.show()

        return left_foot_trajectory, right_foot_trajectory, zmp_trajectory

    def plan_com_trajectory(
        self,
        initial_com: XYZPoint,
        zmp_trajectory: PiecewisePolynomial,
        debug: bool = False,
    ) -> Tuple[PiecewisePolynomial, PiecewisePolynomial]:

        assert initial_com.size == 3
        assert zmp_trajectory.start_time() == 0.0

        def _compute_gains(
            A: NpArrayNNf64,
            B: NpVectorNf64,
            C: NpVectorNf64,
            Qe: float,
            Qx: NpArrayNNf64,
        ) -> Tuple[float, GainsVector, GainsVector]:

            F_bar = np.vstack((C @ A, A))
            I_bar = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4, 1)
            A_bar = np.hstack((I_bar, F_bar))
            B_bar = np.vstack((C @ B, B))
            Q_bar = np.block([[Qe, np.zeros((1, 3))], [np.zeros((3, 1)), Qx]])

            P_bar = solve_discrete_are(a=A_bar, b=B_bar, q=Q_bar, r=R)
            K_bar = np.linalg.inv(R + B_bar.T @ P_bar @ B_bar) @ (
                B_bar.T @ P_bar @ A_bar
            )
            Gi = K_bar[0, 0]
            Gx = K_bar[0, 1:]

            Ac_bar = A_bar - B_bar @ K_bar
            X_bar = -Ac_bar.T @ P_bar @ I_bar
            Gd = np.zeros(num_preview_points, dtype=np.float64)
            Gd[0] = -Gi
            for i in range(1, num_preview_points):
                Gd[i] = (
                    np.linalg.inv(R + B_bar.T @ P_bar @ B_bar) @ B_bar.T @ X_bar
                ).item()
                X_bar = Ac_bar.T @ X_bar

            return Gi, Gx, Gd

        com_z_m = initial_com[2]
        num_preview_points = int(self.preview_time_s / self.dt)
        num_zmp_trajectory_points = int(zmp_trajectory.end_time() / self.dt)
        # For the COM trajectory, we need 'num_preview_points' in the future, so we can't
        # compute it all the way to the end by this method.
        num_com_trajectory_points = num_zmp_trajectory_points - num_preview_points

        A = np.array(
            [
                [1.0, self.dt, self.dt**2 / 2.0],
                [0.0, 1.0, self.dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        B = np.array(
            [self.dt**3 / 6.0, self.dt**2 / 2.0, self.dt], dtype=np.float64
        ).reshape(3, 1)
        C = np.array([1.0, 0.0, -com_z_m / self.g], dtype=np.float64).reshape(1, 3)
        Qe = 100
        qx = 0.0
        Qx = qx * np.eye(3, dtype=np.float64)
        R = 1e-3

        Gi, Gx, Gd = _compute_gains(A=A, B=B, C=C, Qx=Qx, Qe=Qe)

        # State is [x, xdot, xddot]/[y, ydot, yddot]
        com_state_x = np.array([initial_com[0], 0.0, 0.0], dtype=np.float64)[:, None]
        com_state_y = np.array([initial_com[1], 0.0, 0.0], dtype=np.float64)[:, None]

        zmp_output_breaks = []
        zmp_output_samples = np.empty((2, 0), dtype=np.float64)
        com_breaks = [0.0]
        com_samples = np.copy(initial_com).reshape(3, 1)

        error_x, error_y, u_x, u_y = 0.0, 0.0, 0.0, 0.0

        for i in range(1, num_com_trajectory_points):

            t = i * self.dt
            zmp_x, zmp_y = zmp_trajectory.value(t=t)

            preview_times_list = [
                _t * self.dt for _t in range(i + 1, i + 1 + num_preview_points)
            ]
            zmp_preview_poses = zmp_trajectory.vector_values(preview_times_list)

            zmp_output_x = (C @ com_state_x).item()
            zmp_output_y = (C @ com_state_y).item()

            zmp_output_breaks.append(t)
            zmp_output_samples = np.hstack(
                (
                    zmp_output_samples,
                    np.array([zmp_output_x, zmp_output_y]).reshape(2, 1),
                ),
            )

            error_x = zmp_x - zmp_output_x
            error_y = zmp_y - zmp_output_y

            u_x = (
                -Gi * error_x - Gx @ com_state_x - Gd @ zmp_preview_poses[0, :]
            ).item()
            u_y = (
                -Gi * error_y - Gx @ com_state_y - Gd @ zmp_preview_poses[1, :]
            ).item()

            com_state_x = A @ com_state_x + u_x * B
            com_state_y = A @ com_state_y + u_y * B

            com_breaks.append(t)
            com_samples = np.hstack(
                (
                    com_samples,
                    np.array(
                        [
                            com_state_x[0, 0],
                            com_state_y[0, 0],
                            com_z_m,
                        ]
                    ).reshape(3, 1),
                ),
            )

        zmp_output_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=zmp_output_breaks,
            samples=zmp_output_samples,
        )
        com_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=com_breaks,
            samples=com_samples,
        )

        if debug:
            fig = plt.figure("Preview controller COM Trajectory")
            ax = fig.gca()

            _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            _plot_com_trajectory_on_ax(
                ax=ax,
                ax1=ax1,
                ax2=ax2,
                zmp_output_trajectory=zmp_output_trajectory,
                com_trajectory=com_trajectory,
            )
            plt.show()

        return zmp_output_trajectory, com_trajectory

    def compute_full_zmp_result(
        self,
        xy_path: XYPath,
        initial_com: XYZPoint,
        first_footstep: FootstepType = FootstepType.RIGHT,
        debug: bool = False,
    ) -> ZMPPlannerResult:

        (
            left_foot_trajectory,
            right_foot_trajectory,
            zmp_trajectory,
        ) = self.plan_zmp_trajectory(
            xy_path=xy_path,
            first_footstep=first_footstep,
            debug=False,
        )
        zmp_output_trajectory, com_trajectory = self.plan_com_trajectory(
            initial_com=initial_com,
            zmp_trajectory=zmp_trajectory,
            debug=False,
        )

        if debug:
            fig = plt.figure("ZMP trajectories")
            ax = fig.gca()

            _, axes = plt.subplots(nrows=2, ncols=2)
            _plot_zmp_trajectory_on_ax(
                ax=ax,
                left_foot_trajectory=left_foot_trajectory,
                right_foot_trajectory=right_foot_trajectory,
                zmp_trajectory=zmp_trajectory,
                xy_path=xy_path,
                left_foot_polygon=self.left_foot_polygon,
                right_foot_polygon=self.right_foot_polygon,
                initialize_axes=True,
            )
            _plot_foot_trajectory_on_ax(
                ax1=axes[0, 1],
                ax2=axes[1, 1],
                left_foot_trajectory=left_foot_trajectory,
                right_foot_trajectory=right_foot_trajectory,
                initialize_axes=True,
            )
            _plot_com_trajectory_on_ax(
                ax=ax,
                ax1=axes[0, 0],
                ax2=axes[1, 0],
                zmp_output_trajectory=zmp_output_trajectory,
                com_trajectory=com_trajectory,
                initialize_axes=False,
            )
            plt.show()

        return ZMPPlannerResult(
            left_foot_trajectory=left_foot_trajectory,
            right_foot_trajectory=right_foot_trajectory,
            zmp_trajectory=zmp_trajectory,
            zmp_output_trajectory=zmp_output_trajectory,
            com_trajectory=com_trajectory,
        )
