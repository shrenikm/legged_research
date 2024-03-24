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
    nq = ik_plant.num_positions()
    ik_plant_context = ik_plant.CreateDefaultContext()
    initial_q = ik_plant.GetDefaultPositions()
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

    left_foot_polygon = get_left_foot_polygon(
        legged_model_type=legged_model_type,
    )
    right_foot_polygon = get_right_foot_polygon(
        legged_model_type=legged_model_type,
    )
    nfp = NaiveZMPPlanner(
        stride_length_m=0.5,
        swing_phase_time_s=1.0,
        stance_phase_time_s=0.5,
        distance_between_feet=np.abs(left_foot_position[1] - right_foot_position[1]),
        max_orientation_delta=np.deg2rad(30.0),
        left_foot_polygon=left_foot_polygon,
        right_foot_polygon=right_foot_polygon,
        preview_time_s=2.0,
        dt=1e-2,
    )
    x_path = np.arange(0.0, 2.0, step=0.05)
    xy_path = np.vstack((x_path, np.zeros_like(x_path))).T
    zmp_result = nfp.compute_full_zmp_result(
        xy_path=xy_path,
        initial_com=initial_com,
        first_footstep=FootstepType.RIGHT,
        # debug=True,
    )

    stance_start_time = 0.0
    stance_end_time = zmp_result.oriented_zmp_trajectory.get_segment_times()[1]
    print(stance_start_time, stance_end_time)
    dt = 0.1
    nk = int(nfp.stance_phase_time_s / dt)

    alpha_pos = np.array([0.1, 0.0, 0.0])
    prog = MathematicalProgram()
    q_vars_matrix = prog.NewContinuousVariables(rows=nq, cols=nk, name="q_vars")
    com_vars_matrix = prog.NewContinuousVariables(rows=3, cols=nk, name="com_vars")

    for j in range(nk):
        q_vars = q_vars_matrix[:, j]
        com_vars = com_vars_matrix[:, j]
        com_desired = zmp_result.com_trajectory.value(t=j * dt).reshape(3)
        print(j, com_desired, initial_com)

        c1 = PointToPointDistanceConstraint(
            plant=ik_plant,
            frame1=ik_plant.GetFrameByName(lf_name),
            p_B1P1=alpha_pos,
            frame2=ik_plant.world_frame(),
            p_B2P2=left_foot_position + alpha_pos,
            distance_lower=0.0,
            distance_upper=0.01,
            # plant_context=ik_plant.CreateDefaultContext(),
            plant_context=ik_plant_context,
        )
        c2 = PointToPointDistanceConstraint(
            plant=ik_plant,
            frame1=ik_plant.GetFrameByName(lf_name),
            p_B1P1=-alpha_pos,
            frame2=ik_plant.world_frame(),
            p_B2P2=left_foot_position - alpha_pos,
            distance_lower=0.0,
            distance_upper=0.01,
            # plant_context=ik_plant.CreateDefaultContext(),
            plant_context=ik_plant_context,
        )
        c3 = PointToPointDistanceConstraint(
            plant=ik_plant,
            frame1=ik_plant.GetFrameByName(rf_name),
            p_B1P1=alpha_pos,
            frame2=ik_plant.world_frame(),
            p_B2P2=right_foot_position + alpha_pos,
            distance_lower=0.0,
            distance_upper=0.01,
            # plant_context=ik_plant.CreateDefaultContext(),
            plant_context=ik_plant_context,
        )
        c4 = PointToPointDistanceConstraint(
            plant=ik_plant,
            frame1=ik_plant.GetFrameByName(rf_name),
            p_B1P1=-alpha_pos,
            frame2=ik_plant.world_frame(),
            p_B2P2=right_foot_position - alpha_pos,
            distance_lower=0.0,
            distance_upper=0.01,
            # plant_context=ik_plant.CreateDefaultContext(),
            plant_context=ik_plant_context,
        )
        c5 = AngleBetweenVectorsConstraint(
            plant=ik_plant,
            # TODO: Function.
            frameA=ik_plant.GetFrameByName("torso_link"),
            a_A=np.array([0.0, 0.0, 1.0]),
            frameB=ik_plant.world_frame(),
            b_B=np.array([0.0, 0.0, 1.0]),
            angle_lower=0.0,
            angle_upper=0.05,
            # plant_context=ik_plant.CreateDefaultContext(),
            plant_context=ik_plant_context,
        )
        c6 = ComPositionConstraint(
            plant=ik_plant,
            model_instances=None,
            expressed_frame=ik_plant.world_frame(),
            plant_context=ik_plant_context,
        )

        prog.AddConstraint(c1, vars=q_vars)
        prog.AddConstraint(c2, vars=q_vars)
        prog.AddConstraint(c3, vars=q_vars)
        prog.AddConstraint(c4, vars=q_vars)
        prog.AddConstraint(c5, vars=q_vars)
        prog.AddConstraint(c6, vars=np.hstack((q_vars, com_vars)))

        # Fix everything other than the leg joints.
        for i in range(17, nq):
            prog.AddConstraint(q_vars[i] == 0)
        # COM constraint.
        for i in range(3):
            prog.AddConstraint(com_vars[i] == com_desired[i])

    for i in range(nq):
        prog.AddConstraint(q_vars_matrix[i, 0] == initial_q[i])
    for j in range(nk - 1):
        for i in range(7, 17):
            prog.AddCost(1.0 * (q_vars_matrix[i, j + 1] - q_vars_matrix[i, j]) ** 2.0)

    for j in range(nk):
        prog.SetInitialGuess(q_vars_matrix[0:4, j], np.array([1.0, 0.0, 0.0, 0.0]))

    result = Solve(prog=prog)
    print("Success:", result.is_success())
    print("Solution result:", result.get_solution_result())

    return result.GetSolution(q_vars_matrix)


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

    time_spaced_positions = builder.AddSystem(
        TimeSpacedPositions(
            plant=plant,
            positions=positions,
            wait_time=1.0,
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

    #AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()

    simulator = Simulator(system=diagram)
    #plant.get_actuation_input_port(model_instance=legged_model).FixValue(
    #    context=plant.GetMyContextFromRoot(root_context=simulator.get_context()),
    #    value=np.zeros(plant.num_actuators(), dtype=np.float64),
    #)

    with auto_meshcat_visualization(meshcat=meshcat, record=True):
        simulator.AdvanceTo(
            boundary_time=10.0,
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    simulate_passive_robot(
        legged_model_type=legged_model_type,
    )
