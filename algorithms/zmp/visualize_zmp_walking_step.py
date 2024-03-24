import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, StartMeshcat
from pydrake.multibody.all import AddUnitQuaternionConstraintOnPlant
from pydrake.multibody.inverse_kinematics import (
    ComPositionConstraint,
    InverseKinematics,
    UnitQuaternionConstraint,
)
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ContactModel,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.symbolic import Expression
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

    com = plant.CalcCenterOfMassPositionInWorld(plant.CreateDefaultContext())
    print(com)
    tl = plant.EvalBodyPoseInWorld(
        plant.CreateDefaultContext(),
        plant.GetBodyByName("left_ankle_link"),
    )
    tr = plant.EvalBodyPoseInWorld(
        plant.CreateDefaultContext(),
        plant.GetBodyByName("right_ankle_link"),
    )
    pl = tl.translation()
    pr = tr.translation()
    print(pl)
    print(pr)
    desired_com = np.copy(com)
    # desired_com[0] = 0.1
    desired_com[1] = 0.1
    desired_com[2] = 0.949

    ik_plant = MultibodyPlant(0.001)
    add_legged_model_to_plant_and_finalize(
        plant=ik_plant,
        legged_model_type=legged_model_type,
    )
    nq = ik_plant.num_positions()
    ik_plant_context = ik_plant.CreateDefaultContext()
    initial_q = ik_plant.GetDefaultPositions()

    ik = InverseKinematics(
        plant=ik_plant,
        plant_context=ik_plant_context,
        with_joint_limits=True,
    )
    ik.prog().NewContinuousVariables(3, "com_vars")
    p_front = np.array([0.15, 0.0, 0.0])
    ik.AddPointToPointDistanceConstraint(
        frame1=ik_plant.GetFrameByName("left_ankle_link"),
        p_B1P1=p_front,
        frame2=ik_plant.world_frame(),
        p_B2P2=pl + p_front,
        distance_lower=0.0,
        distance_upper=0.0,
    )
    ik.AddPointToPointDistanceConstraint(
        frame1=ik_plant.GetFrameByName("left_ankle_link"),
        p_B1P1=-p_front,
        frame2=ik_plant.world_frame(),
        p_B2P2=pl - p_front,
        distance_lower=0.0,
        distance_upper=0.0,
    )
    ik.AddPointToPointDistanceConstraint(
        frame1=ik_plant.GetFrameByName("right_ankle_link"),
        p_B1P1=p_front,
        frame2=ik_plant.world_frame(),
        p_B2P2=pr + p_front,
        distance_lower=0.0,
        distance_upper=0.0,
    )
    ik.AddPointToPointDistanceConstraint(
        frame1=ik_plant.GetFrameByName("right_ankle_link"),
        p_B1P1=-p_front,
        frame2=ik_plant.world_frame(),
        p_B2P2=pr - p_front,
        distance_lower=0.0,
        distance_upper=0.0,
    )
    ik.AddAngleBetweenVectorsConstraint(
        frameA=ik_plant.GetFrameByName("torso_link"),
        na_A=np.array([0.0, 0.0, 1.0]),
        frameB=ik_plant.world_frame(),
        nb_B=np.array([0.0, 0.0, 1.0]),
        angle_lower=0.0,
        angle_upper=0.1,
    )
    prog = ik.get_mutable_prog()
    vars = prog.decision_variables()
    q_vars = prog.decision_variables()[:nq]
    com_vars = prog.decision_variables()[-3:]

    com_constraint = ComPositionConstraint(
        plant=ik_plant,
        model_instances=None,
        expressed_frame=ik_plant.world_frame(),
        plant_context=ik_plant_context,
    )

    prog.AddL2NormCost(A=100 * np.eye(nq), b=-initial_q, vars=q_vars)
    prog.AddConstraint(constraint=com_constraint, vars=vars)
    AddUnitQuaternionConstraintOnPlant(
        plant=ik_plant,
        q_vars=q_vars,
        prog=prog,
    )
    prog.AddConstraint(com_vars[0] == desired_com[0])
    prog.AddConstraint(com_vars[1] == desired_com[1])
    prog.AddConstraint(com_vars[2] == desired_com[2])

    initial_guess = np.zeros(nq + 3)
    initial_guess[:nq] = initial_q
    prog.SetInitialGuess(vars[:nq], initial_q)

    result = Solve(prog)
    print(result.is_success())
    print(result.GetSolution())
    print(result.GetSolution()[-3:])

    # com_constraint = ComPositionConstraint(
    #    plant=ik_plant,
    #    model_instances=None,
    #    expressed_frame=ik_plant.world_frame(),
    #    plant_context=ik_plant_context,
    # )
    # unit_quaternion_constraint = UnitQuaternionConstraint()

    # prog = MathematicalProgram()
    # q_vars = prog.NewContinuousVariables(nq, "q_vars")
    # com_vars = prog.NewContinuousVariables(3, "com_vars")
    # vars = np.hstack((q_vars, com_vars))

    # prog.AddL2NormCost(A=np.eye(nq), b=initial_q, vars=q_vars)

    # prog.AddConstraint(constraint=com_constraint, vars=vars)
    ## prog.AddConstraint(constraint=unit_quaternion_constraint, vars=q_vars[:4])
    # AddUnitQuaternionConstraintOnPlant(
    #    plant=ik_plant,
    #    q_vars=q_vars,
    #    prog=prog,
    # )
    # prog.AddConstraint(com_vars[0] == desired_com[0])
    # prog.AddConstraint(com_vars[1] == desired_com[1])
    # prog.AddConstraint(com_vars[2] == desired_com[2])

    # initial_guess = np.zeros(nq + 3)
    # initial_guess[:nq] = initial_q
    # prog.SetInitialGuess(vars[:nq], initial_q)

    # result = Solve(prog)
    # print(result.is_success())
    # print(result.GetSolution())
    # print(result.GetSolution()[-3:])

    plant.SetDefaultPositions(
        model_instance=legged_model,
        q_instance=result.GetSolution()[:nq],
        # q_instance=np.zeros(nq),
    )

    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()

    simulator = Simulator(system=diagram)
    plant.get_actuation_input_port(model_instance=legged_model).FixValue(
        context=plant.GetMyContextFromRoot(root_context=simulator.get_context()),
        value=np.zeros(plant.num_actuators(), dtype=np.float64),
    )

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
