from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    JointSliders,
    StartMeshcat,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import RigidBody
from pydrake.visualization import AddDefaultVisualization, AddFrameTriadIllustration

from common.model_utils import LeggedModelType, add_legged_model_to_plant_and_finalize


def visualize_robot(
    legged_model_type: LeggedModelType,
    show_frames: bool = False,
) -> None:
    meshcat = StartMeshcat()
    builder = DiagramBuilder()

    plant: MultibodyPlant
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)

    legged_model = add_legged_model_to_plant_and_finalize(
        plant=plant,
        legged_model_type=legged_model_type,
    )

    print("Num positions:", plant.num_positions())
    print("Num velocities:", plant.num_velocities())
    print("Num actuators:", plant.num_actuators())

    meshcat.DeleteAddedControls()

    sliders = builder.AddSystem(JointSliders(meshcat, plant))
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    if show_frames:
        print("Body frames that are going to be illustrated:")
        for body_index in plant.GetBodyIndices(model_instance=legged_model):
            body: RigidBody = plant.get_body(body_index)
            print(body.name())
            # Add frame triad.
            AddFrameTriadIllustration(
                scene_graph=scene_graph,
                plant=plant,
                body=body,
                length=0.15,
                radius=0.001,
            )

    diagram = builder.Build()
    sliders.Run(diagram, None)


if __name__ == "__main__":
    legged_model_type = LeggedModelType.H1
    show_frames = False
    visualize_robot(
        legged_model_type=legged_model_type,
        show_frames=show_frames,
    )
