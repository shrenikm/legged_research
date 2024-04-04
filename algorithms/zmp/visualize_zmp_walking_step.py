from pydrake.all import DiagramBuilder, StartMeshcat
from pydrake.geometry import MeshcatVisualizer, SceneGraph
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import LeafSystem
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.trajectories import PiecewisePolynomial

from algorithms.zmp.ik_planners import solve_straight_line_walking
from common.custom_types import NpArrayMNf64
from common.drake_utils import auto_meshcat_visualization
from common.model_utils import LeggedModelType, add_legged_model_to_plant_and_finalize


class TimeSpacedPositions(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        positions: NpArrayMNf64,
        wait_time: float,
    ):
        LeafSystem.__init__(self)
        assert plant.num_positions() == positions.shape[0]

        self.wait_time = wait_time
        self.positions = positions
        self._current_k = 0

        self.DeclareVectorOutputPort(
            "position", plant.num_positions(), self.compute_position
        )

    def compute_position(self, context, output):
        current_time = context.get_time()

        if current_time >= (self._current_k + 1) * self.wait_time:
            self._current_k = min(self._current_k + 1, self.positions.shape[1] - 1)
        output.SetFromVector(self.positions[:, self._current_k])


def visualize_zmp_walking_step(
    legged_model_type: LeggedModelType,
) -> None:
    meshcat = StartMeshcat()
    meshcat.DeleteAddedControls()
    builder = DiagramBuilder()

    plant_time_step = 0.001
    plant: MultibodyPlant
    plant = MultibodyPlant(time_step=plant_time_step)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph=scene_graph)

    legged_model = add_legged_model_to_plant_and_finalize(
        plant=plant,
        legged_model_type=legged_model_type,
    )

    positions_traj: PiecewisePolynomial = solve_straight_line_walking(
        legged_model_type=legged_model_type,
        plant_time_step=plant_time_step,
        path_length=2.5,
        ik_sample_time=0.05,
    )
    positions = positions_traj.vector_values(
        positions_traj.get_segment_times(),
    )

    geometry_pose = builder.AddSystem(
        MultibodyPositionToGeometryPose(plant=plant),
    )

    wait_time = 0.1
    time_spaced_positions = builder.AddSystem(
        TimeSpacedPositions(
            plant=plant,
            positions=positions,
            wait_time=wait_time,
        ),
    )

    builder.Connect(
        geometry_pose.get_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        time_spaced_positions.get_output_port(),
        geometry_pose.get_input_port(),
    )

    MeshcatVisualizer.AddToBuilder(
        builder=builder,
        scene_graph=scene_graph,
        meshcat=meshcat,
    )

    diagram = builder.Build()
    simulator = Simulator(system=diagram)
    sim_time = wait_time * positions.shape[1]
    with auto_meshcat_visualization(meshcat=meshcat, record=True):
        simulator.AdvanceTo(
            boundary_time=sim_time,
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    visualize_zmp_walking_step(
        legged_model_type=legged_model_type,
    )
