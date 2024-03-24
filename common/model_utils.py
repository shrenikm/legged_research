import os
from typing import Optional

import numpy as np
from pydrake.multibody.parsing import PackageMap, Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex

from common.class_utils import StrEnum
from common.custom_types import DirName, DirPath, FilePath, PositionsVector

OBJECT_MODELS_DIRNAME = "object_models"
ROBOT_MODELS_DIRNAME = "robot_models"

H1_DESCRIPTION_DIRNAME = "h1_description"
DRAKE_URDF_DIRNAME = "drake_urdf"


class ObjectModelType(StrEnum):
    PLANE_HALFSPACE = "plane_halfspace.sdf"
    PLANE_BOX = "plane_box.urdf"


class LeggedModelType(StrEnum):
    H1 = "h1.urdf"


def get_object_models_directory_path() -> DirPath:
    current_directory_path = os.path.dirname(
        os.path.expanduser(os.path.realpath(__file__))
    )
    models_directory_path = os.path.join(
        current_directory_path,
        "..",
        OBJECT_MODELS_DIRNAME,
    )
    return os.path.realpath(models_directory_path)


def get_robot_models_directory_path() -> DirPath:
    current_directory_path = os.path.dirname(
        os.path.expanduser(os.path.realpath(__file__))
    )
    robot_models_directory_path = os.path.join(
        current_directory_path,
        "..",
        ROBOT_MODELS_DIRNAME,
    )
    return os.path.realpath(robot_models_directory_path)


def add_robot_models_to_package_map(package_map: PackageMap) -> None:
    """
    Add all the robot models/descriptions inside robot_models/ into the package map.
    """
    robot_models_directory_path = get_robot_models_directory_path()

    all_robot_model_dirnames_and_paths = [
        (dirname, dirpath)
        for dirname in os.listdir(robot_models_directory_path)
        if os.path.isdir(dirpath := os.path.join(robot_models_directory_path, dirname))
    ]

    for package_name, package_path in all_robot_model_dirnames_and_paths:
        package_map.Add(
            package_name=package_name,
            package_path=package_path,
        )


def get_description_dirname_for_legged_model_type(
    legged_model_type: LeggedModelType,
) -> DirPath:
    return {LeggedModelType.H1: H1_DESCRIPTION_DIRNAME}[legged_model_type]


def get_description_subdir_for_legged_model_type(
    legged_model_type: LeggedModelType,
) -> DirPath:
    return {LeggedModelType.H1: os.path.join(DRAKE_URDF_DIRNAME)}[legged_model_type]


def get_object_model_urdf_path(object_model_type: ObjectModelType) -> FilePath:
    urdf_filename = object_model_type.value

    return os.path.join(
        get_object_models_directory_path(),
        urdf_filename,
    )


def get_legged_model_urdf_path(legged_model_type: LeggedModelType) -> FilePath:

    urdf_filename = legged_model_type.value

    return os.path.join(
        get_robot_models_directory_path(),
        get_description_dirname_for_legged_model_type(
            legged_model_type=legged_model_type
        ),
        get_description_subdir_for_legged_model_type(
            legged_model_type=legged_model_type
        ),
        urdf_filename,
    )


def get_num_positions_for_legged_model_type(
    legged_model_type: LeggedModelType,
) -> int:
    return {
        LeggedModelType.H1: 26,
    }[legged_model_type]


def get_num_velocities_for_legged_model_type(
    legged_model_type: LeggedModelType,
) -> int:
    return {
        LeggedModelType.H1: 25,
    }[legged_model_type]


def get_num_actuators_for_legged_model_type(
    legged_model_type: LeggedModelType,
) -> int:
    return {
        LeggedModelType.H1: 19,
    }[legged_model_type]


def get_default_positions_for_legged_model_type(
    legged_model_type: LeggedModelType,
) -> PositionsVector:

    h1_default_positions = np.zeros(
        get_num_positions_for_legged_model_type(legged_model_type=LeggedModelType.H1),
        dtype=np.float64,
    )
    # Setting the unit quaternion of the floating base.
    h1_default_positions[0] = 1.0
    # Set z height so that the robot stands on the ground.
    h1_default_positions[6] = 0.98

    # Bend the hip pitch, knees and ankle of both legs.
    h1_default_positions[9] = -0.4
    h1_default_positions[10] = 0.8
    h1_default_positions[11] = -0.4
    h1_default_positions[14] = -0.4
    h1_default_positions[15] = 0.8
    h1_default_positions[16] = -0.4

    return {
        LeggedModelType.H1: h1_default_positions,
    }[legged_model_type]


def get_left_ankle_frame_name(legged_model_type: LeggedModelType) -> str:
    if legged_model_type == LeggedModelType.H1:
        return "left_ankle_link"
    else:
        raise NotImplementedError


def get_right_ankle_frame_name(legged_model_type: LeggedModelType) -> str:
    if legged_model_type == LeggedModelType.H1:
        return "right_ankle_link"
    else:
        raise NotImplementedError


def add_legged_model_to_plant_and_finalize(
    plant: MultibodyPlant,
    legged_model_type: LeggedModelType,
    parser: Optional[Parser] = None,
) -> ModelInstanceIndex:
    """
    Adds the model specified by the model type to the plant using the parser.
    If the parser is not given, one is constructed from the plant and robot_models is added to its package map.
    If the parser given does not contain the robot_models in the package map, it is added as well.
    """

    if parser is None:
        parser = Parser(plant)
        package_map = parser.package_map()
        add_robot_models_to_package_map(package_map=package_map)
    else:
        package_map = parser.package_map()
        description_dirname = get_description_subdir_for_legged_model_type(
            legged_model_type=legged_model_type,
        )
        if not package_map.Contains(description_dirname):
            add_robot_models_to_package_map(package_map=package_map)

    # Add the legged model.
    legged_model = parser.AddModels(
        get_legged_model_urdf_path(legged_model_type=legged_model_type),
    )[0]
    # Add the plane.
    # TODO: PLANE_HALFSPACE segfaults for some reason.
    parser.AddModels(
        get_object_model_urdf_path(object_model_type=ObjectModelType.PLANE_BOX),
    )

    # We assume that the plane model already has itself welded to the world frame in the description file.
    plant.Finalize()

    plant.SetDefaultPositions(
        model_instance=legged_model,
        q_instance=get_default_positions_for_legged_model_type(
            legged_model_type=legged_model_type,
        ),
    )

    return legged_model
