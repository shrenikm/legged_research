import numpy as np
import pytest

from common.testing_utils import execute_pytest_file
from numeric.geometry.shape_utils import rectangle_polygon_array


def test_rectangle_polygon_array() -> None:

    with pytest.raises(AssertionError):
        rectangle_polygon_array(
            size_x=0.0,
            size_y=1.0,
        )
        rectangle_polygon_array(
            size_x=1.0,
            size_y=-1.0,
        )

    r = rectangle_polygon_array(
        size_x=3.0,
        size_y=2.0,
    )
    re = np.array(
        [
            [-1.5, 1.0],
            [1.5, 1.0],
            [1.5, -1.0],
            [-1.5, -1.0],
        ]
    )
    np.testing.assert_array_equal(r, re)

    r = rectangle_polygon_array(
        size_x=3.0,
        size_y=2.0,
        center_xy=np.array([3.0, 2.0]),
    )
    re = np.array(
        [
            [1.5, 3.0],
            [4.5, 3.0],
            [4.5, 1.0],
            [1.5, 1.0],
        ]
    )
    np.testing.assert_array_equal(r, re)


if __name__ == "__main__":
    execute_pytest_file()
