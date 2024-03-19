from typing import List

import numpy as np

from common.custom_types import XYPath


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
