import numpy as np
import pytest
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex

from common.model_utils import (
    LeggedModelType,
    ObjectModelType,
    add_legged_model_to_plant,
    get_default_positions_for_legged_model_type,
    get_description_dirname_for_legged_model_type,
    get_description_subdir_for_legged_model_type,
    get_legged_model_urdf_path,
    get_num_actuators_for_legged_model_type,
    get_num_positions_for_legged_model_type,
    get_num_velocities_for_legged_model_type,
    get_object_model_urdf_path,
    get_object_models_directory_path,
    get_robot_models_directory_path,
)
from common.testing_utils import execute_pytest_file


def test_get_object_models_directory_path() -> None:

    assert isinstance(get_object_models_directory_path(), str)


def test_get_robot_models_directory_path() -> None:

    assert isinstance(get_robot_models_directory_path(), str)


def test_get_description_dirname_for_legged_model_type() -> None:

    assert (
        get_description_dirname_for_legged_model_type(
            legged_model_type=LeggedModelType.H1,
        )
        == "h1_description"
    )


def test_get_description_subdir_for_legged_model_type() -> None:

    assert (
        get_description_subdir_for_legged_model_type(
            legged_model_type=LeggedModelType.H1,
        )
        == "drake_urdf"
    )


def test_get_object_model_urdf_path() -> None:
    plane_sdf_filepath = get_object_model_urdf_path(
        object_model_type=ObjectModelType.PLANE,
    )
    assert isinstance(plane_sdf_filepath, str)
    assert plane_sdf_filepath.endswith(".sdf")


def test_get_legged_model_urdf_path() -> None:

    h1_urdf_filepath = get_legged_model_urdf_path(
        legged_model_type=LeggedModelType.H1,
    )
    assert isinstance(h1_urdf_filepath, str)
    assert h1_urdf_filepath.endswith(".urdf")


def test_get_num_positions_for_legged_model_type() -> None:
    assert (
        get_num_positions_for_legged_model_type(
            legged_model_type=LeggedModelType.H1,
        )
        == 26
    )


def test_get_num_velocities_for_legged_model_type() -> None:
    assert (
        get_num_velocities_for_legged_model_type(
            legged_model_type=LeggedModelType.H1,
        )
        == 25
    )


def test_get_num_actuators_for_legged_model_type() -> None:
    assert (
        get_num_actuators_for_legged_model_type(
            legged_model_type=LeggedModelType.H1,
        )
        == 19
    )


def test_get_default_positions_for_legged_model_type() -> None:
    h1_default_positions = get_default_positions_for_legged_model_type(
        legged_model_type=LeggedModelType.H1,
    )
    assert h1_default_positions.size == get_num_positions_for_legged_model_type(
        legged_model_type=LeggedModelType.H1,
    )
    np.testing.assert_array_equal(
        h1_default_positions,
        np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.98,
                0.0,
                0.0,
                -0.4,
                0.8,
                -0.4,
                0.0,
                0.0,
                -0.4,
                0.8,
                -0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


def test_add_legged_model_to_plant() -> None:

    for legged_model_type in LeggedModelType:

        # With parser.
        plant = MultibodyPlant(time_step=0.001)
        model = add_legged_model_to_plant(
            plant=plant,
            legged_model_type=legged_model_type,
        )
        assert isinstance(model, ModelInstanceIndex)
        print(model)


if __name__ == "__main__":
    execute_pytest_file()