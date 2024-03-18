import sys
from contextlib import contextmanager
from typing import Generator

from pydrake.geometry import Meshcat

from common.logging_utils import LRLogger


@contextmanager
def auto_meshcat_visualization(
    meshcat: Meshcat,
    record: bool,
) -> Generator[None, None, None]:
    """
    Sets up meshcat recording if a meshcat instance is given.

    Usage:
        with auto_meshcat_visualization(record=...):
            simulator.AdvanceTo(...)

    TODO: Maybe "record" needs to be renamed.
    If record is True, records and publishes the recording instead of playing it live.
    If False, plays the recording live.
    """
    if meshcat is not None and record:
        meshcat.StartRecording(set_visualizations_while_recording=False)
    try:
        yield
    except KeyboardInterrupt as e:
        # If interrupted, we log, run the post hook cleanup and then exit.
        LRLogger("AutoMeshcatVisualization").info(f"Simulation interrupted: {e}")
        sys.exit(1)

    if meshcat is not None and record:
        meshcat.StopRecording()
        meshcat.PublishRecording()
