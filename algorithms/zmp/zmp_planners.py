from __future__ import annotations

from enum import Enum, auto

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
from common.custom_types import PolygonArray, XYPath, XYPoint, XYZPoint
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

    zmp_poses = np.empty((0, 3), dtype=np.float64)

    for t in zmp_trajectory.get_segment_times():
        zmp_poses = np.vstack((zmp_poses, zmp_trajectory.value(t).reshape(3)))
        footstep_pose = zmp_trajectory.value(t)
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
    ax.plot(zmp_poses[:, 0], zmp_poses[:, 1], color="cornflowerblue")


def _plot_com_trajectory_on_ax(
    ax: Axes,
    com_trajectory: PiecewisePolynomial,
    initialize_axes: bool = True,
) -> None:
    if initialize_axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect(1.0)

    com_poses = np.empty((0, 3), dtype=np.float64)

    t = 0.0
    sample_time = 0.1
    while t <= com_trajectory.end_time():
        com_poses = np.vstack((com_poses, com_trajectory.value(t).reshape(3)))
        t += sample_time

    # Plot COM poses (Just x and y).
    ax.plot(com_poses[:, 0], com_poses[:, 1], color="salmon")


@attr.frozen
class ZMPPlannerResults:
    zmp_trajectory: PiecewisePolynomial
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

        zmp_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=breaks,
            samples=samples,
        )

        if debug:
            fig = plt.figure("Naive Footstep Trajectory")
            ax = fig.gca()
            _plot_zmp_trajectory_on_ax(
                ax=ax,
                zmp_trajectory=zmp_trajectory,
                xy_path=xy_path,
                left_foot_polygon=self.left_foot_polygon,
                right_foot_polygon=self.right_foot_polygon,
                initialize_axes=True,
            )
            plt.show()

        return zmp_trajectory

    def plan_com_trajectory(
        self,
        initial_com: XYZPoint,
        zmp_trajectory: PiecewisePolynomial,
        debug: bool = False,
    ) -> PiecewisePolynomial:

        assert initial_com.size == 3
        assert zmp_trajectory.start_time() == 0.0

        com_z_m = initial_com[2]
        preview_time_s = 1.0
        # num_preview_points = int(zmp_trajectory.end_time() / self.dt)
        num_preview_points = int(preview_time_s / self.dt)

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
        Qe = 1e-3
        qx = 1e-3
        Qx = qx * np.eye(3, dtype=np.float64)
        R = 1e-3

        F_bar = np.vstack((C @ A, A))
        I_bar = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4, 1)
        A_bar = np.hstack((I_bar, F_bar))
        B_bar = np.vstack((C @ B, B))
        Q_bar = np.block([[Qe, np.zeros((1, 3))], [np.zeros((3, 1)), Qx]])

        P_bar = solve_discrete_are(a=A_bar, b=B_bar, q=Q_bar, r=R)
        K_bar = np.linalg.inv(R + B_bar.T @ P_bar @ B_bar) @ (B_bar.T @ P_bar @ A_bar)
        Gi = K_bar[0, 0]
        Gx = K_bar[0, 1:]

        Ac_bar = A_bar - B_bar @ K_bar
        X_bar = -Ac_bar.T @ P_bar @ I_bar
        Gd = np.zeros(num_preview_points, dtype=np.float64)
        Gd[0] = -Gi
        for i in range(1, num_preview_points):
            Gd[i] = np.linalg.inv(R + B_bar.T @ P_bar @ B_bar) @ B_bar.T @ X_bar
            X_bar = Ac_bar.T @ X_bar


    def plan_com_trajectory2(
        self,
        initial_com: XYZPoint,
        zmp_trajectory: PiecewisePolynomial,
        debug: bool = False,
    ) -> PiecewisePolynomial:
        """
        Integrates the following equations to compute the trajectory (t, x_com, y_com) from (t, x_zmp, y_zmp)

        x_com_ddot = (g / h) * (x_com - x_zmp)
        y_com_ddot = (g / h) * (y_com - y_zmp)

        Where h = z_com - z_zmp = z_com (as z_zmp = 0) = height of COM

        We can use the same integration scheme for both x and y as only the
        com coordinates are different between them.
        We can denote the equations in general as:

        u'' = au - b
        Where a = (g / h), b = (g / h) * x_zmp

        u'' = au - b
        u' = v
        v' = au - b

        We assume u(0) (x and y) = given and v(0) = u'(0) = 0 (zero velocities
        for both x and y)

        v'_(k) = a_k * u_k - b_k
        u'_(k) = v_k
        v_(k+1) = v_k + dt * v'_k
        u_(k+1) = u_k + dt * u'_k

        We can vectorize and solve for both x and y simulatenously as they are
        decoupled.

        """
        assert initial_com.size == 3
        assert zmp_trajectory.start_time() == 0.0

        com_z_m = initial_com[2]
        u_current = np.copy(initial_com[:2])
        v_current = np.zeros(2, dtype=np.float64)
        a_current = self.g / com_z_m

        breaks = [0.0]
        samples = np.array([initial_com[0], initial_com[1], com_z_m]).reshape(3, 1)

        t = 0.0
        while t <= zmp_trajectory.end_time():
            b_current = a_current * zmp_trajectory.value(t=t)[:2].reshape(2)

            # vp = v_prime = v'
            up_current = v_current
            vp_current = a_current * u_current - b_current
            # Integrate.
            u_current = u_current + self.dt * up_current
            v_current = v_current + self.dt * vp_current

            t += self.dt

            print(t)
            print(a_current, b_current)
            print(up_current, vp_current)
            print(u_current, v_current)
            print("===")
            # input()

            # Add to the trajectory.
            breaks.append(t)
            samples = np.hstack(
                (
                    samples,
                    np.array([u_current[0], u_current[1], com_z_m]).reshape(3, 1),
                )
            )
        com_trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks=breaks,
            samples=samples,
        )
        print(com_trajectory.start_time(), com_trajectory.end_time())
        print(com_trajectory.value(com_trajectory.start_time()))
        print(com_trajectory.value(com_trajectory.end_time()))
        input()

        if debug:
            fig = plt.figure("Naive Footstep Trajectory")
            ax = fig.gca()
            _plot_com_trajectory_on_ax(
                ax=ax,
                com_trajectory=com_trajectory,
            )
            plt.show()

        return com_trajectory
