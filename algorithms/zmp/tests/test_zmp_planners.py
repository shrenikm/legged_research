import numpy as np
import pytest

from algorithms.zmp.zmp_planners import FootstepType, NaiveZMPPlanner
from common.custom_types import PolygonArray
from common.testing_utils import execute_pytest_file


@pytest.fixture(scope="module")
def left_foot_polygon() -> PolygonArray:
    # TODO: Make function.
    return np.array(
        [
            [-0.1, 0.05],
            [0.1, 0.05],
            [0.1, -0.05],
            [-0.1, -0.05],
        ]
    )


@pytest.fixture(scope="module")
def right_foot_polygon() -> PolygonArray:
    # TODO: Make function.
    return np.array(
        [
            [-0.1, 0.05],
            [0.1, 0.05],
            [0.1, -0.05],
            [-0.1, -0.05],
        ]
    )


def test_invalid_zmp_planner_construction(
    left_foot_polygon: PolygonArray,
    right_foot_polygon: PolygonArray,
) -> None:

    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=-0.5,
            foot_lift_height_m=0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=0.5,
            stance_phase_time_s=0.5,
            distance_between_feet=0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=2.0,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=0.5,
            foot_lift_height_m=-0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=0.5,
            stance_phase_time_s=0.5,
            distance_between_feet=0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=2.0,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=0.5,
            foot_lift_height_m=0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=-0.5,
            stance_phase_time_s=0.5,
            distance_between_feet=0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=2.0,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=0.5,
            foot_lift_height_m=0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=0.5,
            stance_phase_time_s=-0.5,
            distance_between_feet=0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=2.0,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=0.5,
            foot_lift_height_m=0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=0.5,
            stance_phase_time_s=0.5,
            distance_between_feet=-0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=2.0,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=0.5,
            foot_lift_height_m=0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=0.5,
            stance_phase_time_s=0.5,
            distance_between_feet=0.5,
            max_orientation_delta=-np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=2.0,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=0.5,
            foot_lift_height_m=0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=0.5,
            stance_phase_time_s=0.5,
            distance_between_feet=0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=-2.0,
            dt=0.001,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            stride_length_m=0.5,
            foot_lift_height_m=0.2,
            default_foot_height_m=0.0,
            swing_phase_time_s=0.5,
            stance_phase_time_s=0.5,
            distance_between_feet=0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            preview_time_s=2.0,
            dt=-0.001,
        )


def test_naive_zmp_planner(
    left_foot_polygon: PolygonArray,
    right_foot_polygon: PolygonArray,
    debug: bool = True,
) -> None:

    nfp = NaiveZMPPlanner(
        stride_length_m=0.5,
        foot_lift_height_m=0.1,
        default_foot_height_m=0.0,
        swing_phase_time_s=0.5,
        stance_phase_time_s=0.5,
        distance_between_feet=0.5,
        max_orientation_delta=np.deg2rad(30.0),
        left_foot_polygon=left_foot_polygon,
        right_foot_polygon=right_foot_polygon,
        preview_time_s=2.0,
        dt=1e-2,
    )

    # Testing ZMP/COP trajectory generation for different kinds of paths.

    # Straight path in positive x.
    straight_x_path = np.arange(0.0, 5.0, 0.05)
    straight_y_path = np.zeros(straight_x_path.size, dtype=np.float64)
    straight_xy_path = np.vstack((straight_x_path, straight_y_path)).T
    zmp_result = nfp.compute_full_zmp_result(
        xy_path=straight_xy_path,
        initial_com=np.hstack((straight_xy_path[0], 1.0)),
        first_footstep=FootstepType.RIGHT,
        debug=debug,
    )
    zt = zmp_result.zmp_trajectory
    np.testing.assert_array_almost_equal(
        zt.value(zt.end_time()).reshape(2),
        np.array([4.95, -0.25]),
        decimal=6,
    )

    # Diagonal path angled 45 degrees.
    diagonal_x_path = np.arange(0.0, 5.0, 0.05)
    diagonal_y_path = np.arange(0.0, 5.0, 0.05)
    diagonal_xy_path = np.vstack((diagonal_x_path, diagonal_y_path)).T
    zmp_result = nfp.compute_full_zmp_result(
        xy_path=diagonal_xy_path,
        initial_com=np.hstack((diagonal_xy_path[0], 1.0)),
        first_footstep=FootstepType.RIGHT,
        debug=debug,
    )
    zt = zmp_result.zmp_trajectory
    np.testing.assert_array_almost_equal(
        zt.value(zt.end_time()).reshape(2),
        np.array([5.126777, 4.773223]),
        decimal=6,
    )

    # Path that turns left 90 degrees.
    turn_x_path1 = np.arange(0.0, 5.0, 0.05)
    turn_x_path2 = np.full_like(turn_x_path1, fill_value=5.0)
    turn_x_path = np.hstack((turn_x_path1, turn_x_path2))
    turn_y_path1 = np.full_like(turn_x_path1, fill_value=0.0)
    turn_y_path2 = np.arange(0.0, 5.0, 0.05)
    turn_y_path = np.hstack((turn_y_path1, turn_y_path2))
    turn_xy_path = np.vstack((turn_x_path, turn_y_path)).T
    zmp_result = nfp.compute_full_zmp_result(
        xy_path=turn_xy_path,
        initial_com=np.hstack((turn_xy_path[0], 1.0)),
        first_footstep=FootstepType.RIGHT,
        debug=debug,
    )
    zt = zmp_result.zmp_trajectory
    np.testing.assert_array_almost_equal(
        zt.value(zt.end_time()).reshape(2),
        np.array([5.25, 4.95]),
        decimal=6,
    )


if __name__ == "__main__":
    execute_pytest_file()
