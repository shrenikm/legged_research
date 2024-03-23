from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import attr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pydrake.math import RotationMatrix
from pydrake.trajectories import PiecewisePolynomial, Trajectory
from scipy.linalg import solve_discrete_are

from common.attr_utils import AttrsValidators
from common.constants import ACC_DUE_TO_GRAVITY
from common.custom_types import (
    GainsVector,
    NpArrayNNf64,
    NpVectorNf64,
    PolygonArray,
    XYPath,
    XYPoint,
    XYZPoint,
)
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


def _plot_zmp_trajectory_on_ax(
    ax: Axes,
    oriented_zmp_trajectory: PiecewisePolynomial,
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

    zmp_poses = np.empty((0, 3), dtype=np.float64)

    for t in oriented_zmp_trajectory.get_segment_times():
        zmp_poses = np.vstack((zmp_poses, oriented_zmp_trajectory.value(t).reshape(3)))
        footstep_pose = oriented_zmp_trajectory.value(t)
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

    # Plot ZMP/COP poses.
    ax.plot(
        zmp_poses[:, 0],
        zmp_poses[:, 1],
        color="cornflowerblue",
        label="ZMP trajectory",
    )
    ax.legend(loc="upper right")


def _plot_com_trajectory_on_ax(
    ax: Axes,
    unoriented_zmp_output_trajectory: PiecewisePolynomial,
    com_trajectory: PiecewisePolynomial,
    initialize_axes: bool = True,
    ax2: Optional[Axes] = None,
    ax3: Optional[Axes] = None,
) -> None:
    """
    Plots the com trajectory on the axis ax.
    If ax2/ax3 are given, it plots the x/y trajectories against time.
    """
    if initialize_axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect(1.0)

    com_poses = np.empty((0, 3), dtype=np.float64)
    unoriented_zmp_output_poses = np.empty((0, 2), dtype=np.float64)

    t = 0.0
    sample_time = 0.1
    sample_times = [
        i * sample_time for i in range(int(com_trajectory.end_time() / sample_time))
    ]
    for t in sample_times:
        unoriented_zmp_output_poses = np.vstack(
            (
                unoriented_zmp_output_poses,
                unoriented_zmp_output_trajectory.value(t).reshape(2),
            )
        )
        com_poses = np.vstack((com_poses, com_trajectory.value(t).reshape(3)))

    # Plot ZMP output poses
    ax.plot(
        unoriented_zmp_output_poses[:, 0],
        unoriented_zmp_output_poses[:, 1],
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

    if ax2 is not None:
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("x (m)")
        ax2.plot(
            sample_times,
            com_poses[:, 0],
            color="mediumslateblue",
            label="COM x coordinates",
        )
        ax2.plot(
            sample_times,
            unoriented_zmp_output_poses[:, 0],
            color="olive",
            linestyle="dotted",
            label="ZMP x coordinates",
        )
        ax2.legend(loc="upper right")

    if ax3 is not None:
        ax3.set_xlabel("t (s)")
        ax3.set_ylabel("y (m)")
        ax3.plot(
            sample_times,
            com_poses[:, 1],
            color="mediumslateblue",
            label="COM y coordinates",
        )
        ax3.plot(
            sample_times,
            unoriented_zmp_output_poses[:, 1],
            color="olive",
            linestyle="dotted",
            label="ZMP y coordinates",
        )
        ax3.legend(loc="upper right")

    ax.legend(loc="upper right")


@attr.frozen
class ZMPPlannerResult:
    oriented_zmp_trajectory: PiecewisePolynomial
    unoriented_zmp_output_trajectory: PiecewisePolynomial
    # TODO: Can do better than piecewise here.
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

    distance_between_feet: float = attr.ib(
        validator=AttrsValidators.positive_validator()
    )
    max_orientation_delta: float = attr.ib(
        validator=AttrsValidators.positive_validator()
    )
    left_foot_polygon: PolygonArray
    right_foot_polygon: PolygonArray

    dt: float = attr.ib(default=0.001, validator=AttrsValidators.positive_validator())
    g: float = attr.ib(init=False, default=ACC_DUE_TO_GRAVITY)

    def plan_zmp_trajectory(
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
        # Conservative
        breaks.append(breaks[-1] + swing_phase_time_s)
        samples = np.hstack((samples, samples[:, -1].reshape(3, 1)))
        breaks.append(breaks[-1] + stance_phase_time_s)
        # Non conservative, more continuous
        # breaks.append(breaks[-1] + 2. * swing_phase_time_s + stance_phase_time_s)

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

            # Conservative.
            breaks.append(breaks[-1] + swing_phase_time_s)
            samples = np.hstack((samples, samples[:, -1].reshape(3, 1)))
            breaks.append(breaks[-1] + stance_phase_time_s)
            # Non conservative, more continuous
            # breaks.append(breaks[-1] + swing_phase_time_s + stance_phase_time_s)

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

        oriented_zmp_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=breaks,
            samples=samples,
        )

        if debug:
            fig = plt.figure("Naive Footstep Trajectory")
            ax = fig.gca()
            _plot_zmp_trajectory_on_ax(
                ax=ax,
                oriented_zmp_trajectory=oriented_zmp_trajectory,
                xy_path=xy_path,
                left_foot_polygon=self.left_foot_polygon,
                right_foot_polygon=self.right_foot_polygon,
                initialize_axes=True,
            )
            plt.show()

        return oriented_zmp_trajectory

    def plan_com_trajectory(
        self,
        initial_com: XYZPoint,
        oriented_zmp_trajectory: PiecewisePolynomial,
        preview_time_s: float = 2.0,
        debug: bool = False,
    ) -> Tuple[PiecewisePolynomial, PiecewisePolynomial]:

        assert initial_com.size == 3
        assert oriented_zmp_trajectory.start_time() == 0.0

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
        num_preview_points = int(preview_time_s / self.dt)
        num_zmp_trajectory_points = int(oriented_zmp_trajectory.end_time() / self.dt)
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

        unoriented_zmp_output_breaks = []
        unoriented_zmp_output_samples = np.empty((2, 0), dtype=np.float64)
        com_breaks = []
        com_samples = np.empty((3, 0), dtype=np.float64)

        error_x, error_y, u_x, u_y = 0.0, 0.0, 0.0, 0.0

        for i in range(num_com_trajectory_points):

            t = i * self.dt
            zmp_x, zmp_y, _ = oriented_zmp_trajectory.value(t=t)

            preview_times_list = [
                _t * self.dt for _t in range(i + 1, i + 1 + num_preview_points)
            ]
            zmp_preview_poses = oriented_zmp_trajectory.vector_values(
                preview_times_list
            )

            unoriented_zmp_output_x = (C @ com_state_x).item()
            unoriented_zmp_output_y = (C @ com_state_y).item()

            unoriented_zmp_output_breaks.append(t)
            unoriented_zmp_output_samples = np.hstack(
                (
                    unoriented_zmp_output_samples,
                    np.array(
                        [unoriented_zmp_output_x, unoriented_zmp_output_y]
                    ).reshape(2, 1),
                ),
            )

            error_x = zmp_x - unoriented_zmp_output_x
            error_y = zmp_y - unoriented_zmp_output_y

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

        unoriented_zmp_output_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=unoriented_zmp_output_breaks,
            samples=unoriented_zmp_output_samples,
        )
        com_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=com_breaks,
            samples=com_samples,
        )

        if debug:
            fig = plt.figure("Preview controller COM Trajectory")
            ax = fig.gca()
            _plot_com_trajectory_on_ax(
                ax=ax,
                unoriented_zmp_output_trajectory=unoriented_zmp_output_trajectory,
                com_trajectory=com_trajectory,
            )
            plt.show()

        return unoriented_zmp_output_trajectory, com_trajectory

    def compute_full_zmp_result(
        self,
        xy_path: XYPath,
        stride_length_m: float,
        swing_phase_time_s: float,
        stance_phase_time_s: float,
        initial_com: XYZPoint,
        preview_time_s: float = 2.0,
        first_footstep: FootstepType = FootstepType.RIGHT,
        debug: bool = False,
    ) -> ZMPPlannerResult:

        oriented_zmp_trajectory = self.plan_zmp_trajectory(
            xy_path=xy_path,
            stride_length_m=stride_length_m,
            swing_phase_time_s=swing_phase_time_s,
            stance_phase_time_s=stance_phase_time_s,
            first_footstep=first_footstep,
            debug=False,
        )
        unoriented_zmp_output_trajectory, com_trajectory = self.plan_com_trajectory(
            initial_com=initial_com,
            oriented_zmp_trajectory=oriented_zmp_trajectory,
            preview_time_s=preview_time_s,
            debug=False,
        )

        if debug:
            fig, (ax, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
            _plot_zmp_trajectory_on_ax(
                ax=ax,
                oriented_zmp_trajectory=oriented_zmp_trajectory,
                xy_path=xy_path,
                left_foot_polygon=self.left_foot_polygon,
                right_foot_polygon=self.right_foot_polygon,
            )
            _plot_com_trajectory_on_ax(
                ax=ax,
                unoriented_zmp_output_trajectory=unoriented_zmp_output_trajectory,
                com_trajectory=com_trajectory,
                ax2=ax2,
                ax3=ax3,
            )
            plt.show()

        return ZMPPlannerResult(
            oriented_zmp_trajectory=oriented_zmp_trajectory,
            unoriented_zmp_output_trajectory=unoriented_zmp_output_trajectory,
            com_trajectory=com_trajectory,
        )
