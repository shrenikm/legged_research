from typing import Optional

import numpy as np

from common.custom_types import PolygonArray, XYPoint


def rectangle_polygon_array(
    size_x: float,
    size_y: float,
    center_xy: Optional[XYPoint] = None,
) -> PolygonArray:
    """
    Array of size 4x2, starting from top left to
    bottom left clockwise.
    If center_xy is None, it is assumed to be (0., 0.)
    """
    assert size_x > 0.
    assert size_y > 0.
    if center_xy is None:
        center_xy = np.zeros(2, dtype=np.float64)
    return np.array(
        [
            [center_xy[0] - 0.5 * size_x, center_xy[1] + 0.5 * size_y],
            [center_xy[0] + 0.5 * size_x, center_xy[1] + 0.5 * size_y],
            [center_xy[0] + 0.5 * size_x, center_xy[1] - 0.5 * size_y],
            [center_xy[0] - 0.5 * size_x, center_xy[1] - 0.5 * size_y],
        ],
        dtype=np.float64,
    )
