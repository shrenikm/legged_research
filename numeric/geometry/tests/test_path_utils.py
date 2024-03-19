import numpy as np
import pytest

from common.custom_types import XYPath
from common.testing_utils import execute_pytest_file
from numeric.geometry.path_utils import segment_path_index, segment_path_indices


@pytest.fixture(scope="module")
def xy_path() -> XYPath:
    num_points = 10
    x_coords = np.array([1.1 * i for i in range(num_points)], dtype=np.float64)
    xy_path = np.zeros((num_points, 2), dtype=np.float64)
    xy_path[:, 0] = x_coords

    return xy_path


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


if __name__ == "__main__":
    execute_pytest_file()
