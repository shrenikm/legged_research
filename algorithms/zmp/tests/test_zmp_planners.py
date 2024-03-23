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


def test_footstep_type() -> None:

    assert FootstepType.LEFT.invert() == FootstepType.RIGHT
    assert FootstepType.RIGHT.invert() == FootstepType.LEFT


def test_invalid_zmp_planner_construction(
    left_foot_polygon: PolygonArray,
    right_foot_polygon: PolygonArray,
) -> None:

    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            distance_between_feet=-0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            distance_between_feet=0.5,
            max_orientation_delta=-np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
        )
    with pytest.raises(AssertionError):
        NaiveZMPPlanner(
            distance_between_feet=0.5,
            max_orientation_delta=np.deg2rad(30.0),
            left_foot_polygon=left_foot_polygon,
            right_foot_polygon=right_foot_polygon,
            dt=-0.001,
        )


def test_naive_zmp_planner(
    left_foot_polygon: PolygonArray,
    right_foot_polygon: PolygonArray,
    debug: bool = True,
) -> None:

    nfp = NaiveZMPPlanner(
        distance_between_feet=0.5,
        max_orientation_delta=np.deg2rad(30.0),
        left_foot_polygon=left_foot_polygon,
        right_foot_polygon=right_foot_polygon,
        dt=1e-2,
    )

    # Testing ZMP/COP trajectory generation for different kinds of paths.

    # Straight path in positive x.
    straight_x_path = np.arange(0.0, 10.0, 0.05)
    straight_y_path = np.zeros(straight_x_path.size, dtype=np.float64)
    straight_xy_path = np.vstack((straight_x_path, straight_y_path)).T
    zmp_result = nfp.compute_full_zmp_result(
        xy_path=straight_xy_path,
        stride_length_m=0.5,
        swing_phase_time_s=0.5,
        stance_phase_time_s=0.1,
        initial_com=np.hstack((straight_xy_path[0], 0.22)),
        preview_time_s=2.,
        first_footstep=FootstepType.RIGHT,
        debug=debug,
    )
    ozt = zmp_result.oriented_zmp_trajectory
    np.testing.assert_array_almost_equal(
        ozt.value(ozt.end_time()).reshape(3),
        np.array([9.95, -0.25, 0.0]),
        decimal=6,
    )

    ## Diagonal path angled 45 degrees.
    #diagonal_x_path = np.arange(0.0, 10.0, 0.05)
    #diagonal_y_path = np.arange(0.0, 10.0, 0.05)
    #diagonal_xy_path = np.vstack((diagonal_x_path, diagonal_y_path)).T
    #zmp_result = nfp.compute_full_zmp_result(
    #    xy_path=diagonal_xy_path,
    #    stride_length_m=0.5,
    #    swing_phase_time_s=1.0,
    #    stance_phase_time_s=1.0,
    #    initial_com=np.hstack((diagonal_xy_path[0], 1.0)),
    #    first_footstep=FootstepType.RIGHT,
    #    debug=debug,
    #)
    #ozt = zmp_result.oriented_zmp_trajectory
    #np.testing.assert_array_almost_equal(
    #    ozt.value(ozt.end_time()).reshape(3),
    #    np.array([9.7732233, 10.1267767, 0.78539816]),
    #    decimal=6,
    #)

    ## Path that turns left 90 degrees.
    #turn_x_path1 = np.arange(0.0, 10.0, 0.05)
    #turn_x_path2 = np.full_like(turn_x_path1, fill_value=10.0)
    #turn_x_path = np.hstack((turn_x_path1, turn_x_path2))
    #turn_y_path1 = np.full_like(turn_x_path1, fill_value=0.0)
    #turn_y_path2 = np.arange(0.0, 10.0, 0.05)
    #turn_y_path = np.hstack((turn_y_path1, turn_y_path2))
    #turn_xy_path = np.vstack((turn_x_path, turn_y_path)).T
    #zmp_result = nfp.compute_full_zmp_result(
    #    xy_path=turn_xy_path,
    #    stride_length_m=0.5,
    #    swing_phase_time_s=1.0,
    #    stance_phase_time_s=1.0,
    #    initial_com=np.hstack((turn_xy_path[0], 1.0)),
    #    first_footstep=FootstepType.RIGHT,
    #    debug=debug,
    #)
    #ozt = zmp_result.oriented_zmp_trajectory
    #np.testing.assert_array_almost_equal(
    #    ozt.value(ozt.end_time()).reshape(3),
    #    np.array([10.25, 9.95, 1.570796]),
    #    decimal=6,
    #)


if __name__ == "__main__":
    execute_pytest_file()
