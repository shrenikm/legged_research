import os

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


def visualize_robot(
    show_frames: bool = False,
) -> None:
    builder = DiagramBuilder()

    plant: MultibodyPlant
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)

    base_path = os.path.realpath(os.path.dirname(__file__))
    h1_path = os.path.join(base_path, "..", "robot_models", "h1_description")
    parser = Parser(plant)
    package_map = parser.package_map()
    package_map.Add(
        package_name="h1_description",
        package_path=h1_path,
    )

    assert package_map.Contains("h1_description")
    h1_model = parser.AddModels(h1_path)[0]

    plant.Finalize()
    print("Num positions:", plant.num_positions())
    print("Num velocities:", plant.num_velocities())
    print("Num actuators:", plant.num_actuators())

    meshcat.DeleteAddedControls()

    sliders = builder.AddSystem(JointSliders(meshcat, plant))
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    if show_frames:
        print("Body frames that are going to be illustrated:")
        for body_index in plant.GetBodyIndices(model_instance=h1_model):
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

    meshcat = StartMeshcat()

    show_frames = False
    visualize_robot(
        show_frames=show_frames,
    )
