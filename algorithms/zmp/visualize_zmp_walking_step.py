import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, StartMeshcat
from pydrake.geometry import MeshcatVisualizer, SceneGraph
from pydrake.multibody.all import AddUnitQuaternionConstraintOnPlant
from pydrake.multibody.inverse_kinematics import (
    AngleBetweenVectorsConstraint,
    ComPositionConstraint,
    InverseKinematics,
    PointToPointDistanceConstraint,
    UnitQuaternionConstraint,
)
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ContactModel,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.symbolic import Expression
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import Context, LeafSystem
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.visualization import AddDefaultVisualization

from algorithms.zmp.ik_planners import ZMPIKPlanner
from algorithms.zmp.zmp_planners import FootstepType, NaiveZMPPlanner
from common.custom_types import NpArrayMNf64
from common.drake_utils import auto_meshcat_visualization
from common.model_utils import (
    LeggedModelType,
    add_legged_model_to_plant_and_finalize,
    get_left_foot_frame_name,
    get_left_foot_polygon,
    get_right_foot_frame_name,
    get_right_foot_polygon,
)


# TODO: Move out.
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


def _solve_walking_step(
    legged_model_type: LeggedModelType,
) -> NpArrayMNf64:
    ik_plant = MultibodyPlant(0.001)
    add_legged_model_to_plant_and_finalize(
        plant=ik_plant,
        legged_model_type=legged_model_type,
    )
    ik_plant_context = ik_plant.CreateDefaultContext()
    initial_com = ik_plant.CalcCenterOfMassPositionInWorld(ik_plant_context)
    print("com:", initial_com)

    lf_name = get_left_foot_frame_name(legged_model_type=legged_model_type)
    rf_name = get_right_foot_frame_name(legged_model_type=legged_model_type)

    left_foot_position = ik_plant.EvalBodyPoseInWorld(
        ik_plant_context,
        ik_plant.GetBodyByName(name=lf_name),
    ).translation()
    right_foot_position = ik_plant.EvalBodyPoseInWorld(
        ik_plant_context,
        ik_plant.GetBodyByName(name=rf_name),
    ).translation()

    default_foot_height_m = left_foot_position[2]
    distance_between_feet = np.abs(left_foot_position[1] - right_foot_position[1])
    x_start = left_foot_position[0]

    left_foot_polygon = get_left_foot_polygon(
        legged_model_type=legged_model_type,
    )
    right_foot_polygon = get_right_foot_polygon(
        legged_model_type=legged_model_type,
    )
    nzp = NaiveZMPPlanner(
        stride_length_m=0.2,
        foot_lift_height_m=0.05,
        default_foot_height_m=default_foot_height_m,
        swing_phase_time_s=1.0,
        stance_phase_time_s=0.5,
        distance_between_feet=distance_between_feet,
        max_orientation_delta=np.deg2rad(30.0),
        left_foot_polygon=left_foot_polygon,
        right_foot_polygon=right_foot_polygon,
        preview_time_s=2.0,
        dt=1e-2,
    )
    path_length = 2.0
    x_path = np.arange(x_start, x_start + path_length, step=0.05)
    xy_path = np.vstack((x_path, np.zeros_like(x_path))).T
    zmp_result = nzp.compute_full_zmp_result(
        xy_path=xy_path,
        initial_com=initial_com,
        first_footstep=FootstepType.RIGHT,
        #debug=True,
    )

    ikzp = ZMPIKPlanner(
        legged_model_type=legged_model_type,
        plant=ik_plant,
        plant_context=ik_plant_context,
        sample_time_s=0.1,
        alpha=0.1,
    )
    return ikzp.compute_positions(
        zmp_result=zmp_result,
    )


def simulate_passive_robot(
    legged_model_type: LeggedModelType,
) -> None:
    meshcat = StartMeshcat()
    meshcat.DeleteAddedControls()
    builder = DiagramBuilder()

    plant: MultibodyPlant
    plant = MultibodyPlant(time_step=0.001)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph=scene_graph)

    legged_model = add_legged_model_to_plant_and_finalize(
        plant=plant,
        legged_model_type=legged_model_type,
    )

    positions = _solve_walking_step(
        legged_model_type=legged_model_type,
    )
    print(positions.shape)

    geometry_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant=plant))

    wait_time = 0.5
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

    # AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()

    simulator = Simulator(system=diagram)
    # plant.get_actuation_input_port(model_instance=legged_model).FixValue(
    #    context=plant.GetMyContextFromRoot(root_context=simulator.get_context()),
    #    value=np.zeros(plant.num_actuators(), dtype=np.float64),
    # )

    sim_time = wait_time * positions.shape[1]
    with auto_meshcat_visualization(meshcat=meshcat, record=True):
        simulator.AdvanceTo(
            boundary_time=sim_time,
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    simulate_passive_robot(
        legged_model_type=legged_model_type,
    )
