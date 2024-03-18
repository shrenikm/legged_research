import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, StartMeshcat
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ContactModel,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import Context
from pydrake.visualization import AddDefaultVisualization

from common.drake_utils import auto_meshcat_visualization
from common.model_utils import LeggedModelType, add_legged_model_to_plant_and_finalize


def simulate_passive_robot(
    legged_model_type: LeggedModelType,
) -> None:
    meshcat = StartMeshcat()
    meshcat.DeleteAddedControls()
    builder = DiagramBuilder()

    plant: MultibodyPlant

    config = MultibodyPlantConfig(
        time_step=0.001,
    )
    plant, scene_graph = AddMultibodyPlant(
        config=config,
        builder=builder,
    )
    plant.set_contact_model(ContactModel.kHydroelasticWithFallback)

    legged_model = add_legged_model_to_plant_and_finalize(
        plant=plant,
        legged_model_type=legged_model_type,
    )
    plant.GetBodyByName("left_ankle_link")

    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()

    simulator = Simulator(system=diagram)
    plant.get_actuation_input_port(model_instance=legged_model).FixValue(
        context=plant.GetMyContextFromRoot(root_context=simulator.get_context()),
        value=np.zeros(plant.num_actuators(), dtype=np.float64),
    )

    with auto_meshcat_visualization(meshcat=meshcat, record=True):
        simulator.AdvanceTo(
            boundary_time=20.0,
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    simulate_passive_robot(
        legged_model_type=legged_model_type,
    )
