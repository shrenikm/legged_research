from typing import List, Tuple, Union

import numpy as np

from common.custom_types import AnglesVector, XYPath, XYThetaPath, XYThetaPose


def normalize_angles(angles: Union[float, AnglesVector]) -> Union[float, AnglesVector]:
    """
    Mapping the angle/angles to be between -2pi and 2pi
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def segment_path_index(
    xy_path: XYPath,
    segment_length: float,
    start_index: int = 0,
) -> int:
    """
    Starting from start_index, returns the smallest index (called segment_index) on the path such that
    path_length(xy_path[start_index: segment_index + 1]) >= segment_length
    """
    assert xy_path.shape[0] > 1
    assert xy_path.shape[1] == 2

    remaining_path = xy_path[start_index:, :]
    remaining_path_successive_lengths = np.linalg.norm(
        np.diff(remaining_path, axis=0),
        axis=1,
    )
    cumulative_lengths = np.cumsum(remaining_path_successive_lengths)
    possible_indices = np.where(cumulative_lengths >= segment_length)[0]
    if len(possible_indices) == 0:
        return len(xy_path) - 1
    else:
        return start_index + possible_indices[0] + 1


def segment_path_indices(
    xy_path: XYPath,
    segment_length: float,
    start_index: int = 0,
) -> List[int]:
    """
    Starting from the start_index, returns a list of indices such that segment formed by the consecutive
    indices all have path length >= segment_length.
    TODO: Not a very efficient implementation.
    """
    assert xy_path.shape[0] > 1
    assert xy_path.shape[1] == 2

    segment_index = start_index
    segment_indices = []
    while segment_index < xy_path.shape[0] - 1:
        segment_index = segment_path_index(
            xy_path=xy_path,
            segment_length=segment_length,
            start_index=segment_index,
        )
        segment_indices.append(segment_index)

    return segment_indices


def compute_oriented_xy_path(
    xy_path: XYPath,
) -> XYThetaPath:

    assert xy_path.shape[0] > 1
    assert xy_path.shape[1] == 2

    n = xy_path.shape[0]
    xytheta_path = np.zeros((n, 3), dtype=np.float64)
    xytheta_path[:, :2] = xy_path

    for i in range(n - 1):
        # Note that this angle is already normalized as the domain of atan2
        # lies in [-pi, pi]
        xytheta_path[i, 2] = np.arctan2(
            xy_path[i + 1, 1] - xy_path[i, 1],
            xy_path[i + 1, 0] - xy_path[i, 0],
        )
    xytheta_path[n - 1, 2] = xytheta_path[n - 2, 2]

    return xytheta_path


def compute_xytheta_side_poses(
    xytheta_pose: XYThetaPose,
    half_distance_m: float,
) -> Tuple[XYThetaPose, XYThetaPose]:
    """
    For a given (x, y, theta) pose, computes and returns a tuple of two poses
    that will lie on either side of the line at (x, y) oriented by theta.
    The line joining the xy coordinates of the left and right poses will be
    normal to theta.
    """
    assert xytheta_pose.ndim == 1
    assert xytheta_pose.size == 3
    assert half_distance_m >= 0.

    x, y, theta = xytheta_pose
    left_pose = np.array(
        [
            x - half_distance_m * np.sin(theta),
            y + half_distance_m * np.cos(theta),
            theta,
        ],
        dtype=np.float64,
    )
    right_pose = np.array(
        [
            x + half_distance_m * np.sin(theta),
            y - half_distance_m * np.cos(theta),
            theta,
        ],
        dtype=np.float64,
    )

    return left_pose, right_pose
