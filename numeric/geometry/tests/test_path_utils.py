import numpy as np
import pytest

from common.custom_types import XYPath
from common.testing_utils import execute_pytest_file
from numeric.geometry.path_utils import (
    compute_oriented_xy_path,
    compute_xytheta_side_poses,
    normalize_angles,
    segment_path_index,
    segment_path_indices,
)


@pytest.fixture(scope="module")
def xy_path() -> XYPath:
    num_points = 10
    x_coords = np.array([1.1 * i for i in range(num_points)], dtype=np.float64)
    xy_path = np.zeros((num_points, 2), dtype=np.float64)
    xy_path[:, 0] = x_coords

    return xy_path


def test_normalize_angle() -> None:

    # Angles that require no remap.
    np.testing.assert_equal(
        normalize_angles(0.0),
        0.0,
    )
    np.testing.assert_equal(
        normalize_angles(1.0),
        1.0,
    )
    np.testing.assert_equal(
        normalize_angles(-1.0),
        -1.0,
    )
    np.testing.assert_equal(
        normalize_angles(2.0),
        2.0,
    )
    np.testing.assert_equal(
        normalize_angles(-2.0),
        -2.0,
    )

    # Angles required to be remapped.
    np.testing.assert_equal(
        normalize_angles(4.0),
        -(2 * np.pi - 4.0),
    )
    np.testing.assert_equal(
        normalize_angles(-4.5),
        (2 * np.pi - 4.5),
    )

    # Full wrapped.
    np.testing.assert_equal(
        normalize_angles(-2 * np.pi + 1.0),
        1.0,
    )
    np.testing.assert_equal(
        normalize_angles(2 * np.pi - 1.0),
        -1.0,
    )
    np.testing.assert_equal(
        normalize_angles(6 * np.pi + 4.5),
        -(2 * np.pi - 4.5),
    )
    np.testing.assert_equal(
        normalize_angles(-6 * np.pi - 4.0),
        (2 * np.pi - 4.0),
    )
    np.testing.assert_array_equal(
        normalize_angles(
            angles=np.array([np.pi / 2.0, 5 * np.pi / 4.0, -5.0 * np.pi / 4.0])
        ),
        np.array([np.pi / 2.0, -3 * np.pi / 4.0, 3.0 * np.pi / 4.0]),
    )


def test_segment_path_index(xy_path: XYPath) -> None:

    # Invalid arguments.
    with pytest.raises(AssertionError):
        segment_path_index(
            xy_path=np.zeros((10, 3)),
            segment_length=1,
        )
    with pytest.raises(AssertionError):
        segment_path_index(
            xy_path=np.ones((1, 2)),
            segment_length=1,
        )

    # Degenerate case.
    # Even for an infeasible argument (0 length, we get 1 as the output) as it'll
    # find the smallest index such that the segment formed by it and the start
    # index is >= length
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=0,
        )
        == 1
    )

    # Simple cases.
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=1,
        )
        == 1
    )
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=4,
        )
        == 4
    )

    # Offset start case.
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=2,
            start_index=5,
        )
        == 7
    )

    # Segment too large for remaining path cases.
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=2.0,
            start_index=8,
        )
        == 9
    )
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=7.0,
            start_index=5,
        )
        == 9
    )
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=20.0,
        )
        == 9
    )
    # For the last point.
    assert (
        segment_path_index(
            xy_path=xy_path,
            segment_length=0.0,
            start_index=9,
        )
        == 9
    )


def test_segment_path_indices(xy_path: XYPath) -> None:
    # Invalid arguments.
    with pytest.raises(AssertionError):
        segment_path_indices(
            xy_path=np.zeros((10, 3)),
            segment_length=1,
        )
    with pytest.raises(AssertionError):
        segment_path_indices(
            xy_path=np.ones((1, 2)),
            segment_length=1,
        )

    # Simple cases.
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=1,
    ) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=2,
    ) == [2, 4, 6, 8, 9]
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=4,
    ) == [4, 8, 9]

    # Offset cases.
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=1,
        start_index=5,
    ) == [6, 7, 8, 9]
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=3,
        start_index=2,
    ) == [5, 8, 9]
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=3,
        start_index=6,
    ) == [9]

    # Segment too large for remaining path cases.
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=20,
    ) == [9]
    assert segment_path_indices(
        xy_path=xy_path,
        segment_length=10,
        start_index=5,
    ) == [9]
    # For the last point.
    assert (
        segment_path_indices(
            xy_path=xy_path,
            segment_length=2,
            start_index=9,
        )
        == []
    )


def test_compute_oriented_xy_path() -> None:

    # Invalid inputs.
    with pytest.raises(AssertionError):
        compute_oriented_xy_path(
            xy_path=np.zeros((10, 3)),
        )
    with pytest.raises(AssertionError):
        compute_oriented_xy_path(
            xy_path=np.zeros((1, 2)),
        )

    xy_path = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 3.0],
            [0.0, 3.0],
            [0.0, 2.0],
        ]
    )
    oriented_xy_path = compute_oriented_xy_path(
        xy_path=xy_path,
    )
    expected_oriented_xy_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 4.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, np.pi / 2.0],
            [2.0, 2.0, 3 * np.pi / 4.0],
            [1.0, 3.0, np.pi],
            [0.0, 3.0, -np.pi / 2.0],
            [0.0, 2.0, -np.pi / 2.0],
        ]
    )
    np.testing.assert_array_equal(
        oriented_xy_path,
        expected_oriented_xy_path,
    )


def test_compute_xytheta_side_poses() -> None:

    # Invalid inputs.
    with pytest.raises(AssertionError):
        compute_xytheta_side_poses(
            xytheta_pose=np.zeros(2),
            half_distance_m=1.0,
        )
    with pytest.raises(AssertionError):
        compute_xytheta_side_poses(
            xytheta_pose=np.zeros((1, 3)),
            half_distance_m=1.0,
        )
    with pytest.raises(AssertionError):
        compute_xytheta_side_poses(
            xytheta_pose=np.zeros(3),
            half_distance_m=-1.0,
        )

    l, r = compute_xytheta_side_poses(
        xytheta_pose=np.array([0.0, 0.0, 0.0]),
        half_distance_m=0.0,
    )
    np.testing.assert_array_almost_equal(
        l,
        np.array([0.0, 0.0, 0.0]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        r,
        np.array([0.0, 0.0, 0.0]),
        decimal=6,
    )

    l, r = compute_xytheta_side_poses(
        xytheta_pose=np.array([0.0, 0.0, 0.0]),
        half_distance_m=0.5,
    )
    np.testing.assert_array_almost_equal(
        l,
        np.array([0.0, 0.5, 0.0]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        r,
        np.array([0.0, -0.5, 0.0]),
        decimal=6,
    )

    l, r = compute_xytheta_side_poses(
        xytheta_pose=np.array([0.0, 0.0, np.pi / 2.0]),
        half_distance_m=1.0,
    )
    np.testing.assert_array_almost_equal(
        l,
        np.array([-1.0, 0.0, np.pi / 2.0]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        r,
        np.array([1.0, 0.0, np.pi / 2.0]),
        decimal=6,
    )

    l, r = compute_xytheta_side_poses(
        xytheta_pose=np.array([0.0, 0.0, -np.pi / 2.0]),
        half_distance_m=1.0,
    )
    np.testing.assert_array_almost_equal(
        l,
        np.array([1.0, 0.0, -np.pi / 2.0]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        r,
        np.array([-1.0, 0.0, -np.pi / 2.0]),
        decimal=6,
    )

    l, r = compute_xytheta_side_poses(
        xytheta_pose=np.array([-2.0, 3.0, 3. * np.pi / 4.0]),
        half_distance_m=1.0,
    )
    np.testing.assert_array_almost_equal(
        l,
        np.array([-2.707107, 2.292893, 3. * np.pi / 4.0]),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        r,
        np.array([-1.292893, 3.707107, 3. * np.pi / 4.0]),
        decimal=6,
    )


if __name__ == "__main__":
    execute_pytest_file()
