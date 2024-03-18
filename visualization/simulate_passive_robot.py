import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, StartMeshcat
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization

from common.drake_utils import auto_meshcat_visualization
from common.model_utils import LeggedModelType, add_legged_model_to_plant_and_finalize


def simulate_passive_robot(
    legged_model_type: LeggedModelType,
) -> None:
    meshcat = StartMeshcat()
    builder = DiagramBuilder()

    plant: MultibodyPlant
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)

    legged_model = add_legged_model_to_plant_and_finalize(
        plant=plant,
        legged_model_type=legged_model_type,
    )

    meshcat.DeleteAddedControls()
    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()

    simulator = Simulator(system=diagram)
    #plant.get_actuation_input_port(model_instance=legged_model).FixValue(
    #    context=plant.GetMyContextFromRoot(root_context=simulator.get_context()),
    #    value=np.zeros(plant.num_actuators(), dtype=np.float64),
    #)

    with auto_meshcat_visualization(meshcat=meshcat, record=True):
        simulator.AdvanceTo(
            boundary_time=5.0,
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    simulate_passive_robot(
        legged_model_type=legged_model_type,
    )
