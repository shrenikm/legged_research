import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, StartMeshcat
from pydrake.multibody.all import AddUnitQuaternionConstraintOnPlant
from pydrake.multibody.inverse_kinematics import (
    ComPositionConstraint,
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
    print(plant.GetPositionUpperLimits())
    desired_com = np.copy(com)
    desired_com[2] = 0.5

    ik_plant = MultibodyPlant(0.001)
    add_legged_model_to_plant_and_finalize(
        plant=ik_plant,
        legged_model_type=legged_model_type,
    )
    ik_plant_context = ik_plant.CreateDefaultContext()
    initial_q = plant.GetDefaultPositions()
    com_constraint = ComPositionConstraint(
        plant=ik_plant,
        model_instances=None,
        expressed_frame=ik_plant.world_frame(),
        plant_context=ik_plant_context,
    )
    unit_quaternion_constraint = UnitQuaternionConstraint()
    nq = ik_plant.num_positions()

    prog = MathematicalProgram()
    q_vars = prog.NewContinuousVariables(nq, "q_var")
    com_vars = prog.NewContinuousVariables(3, "com_var")
    vars = np.hstack((q_vars, com_vars))

    prog.AddL2NormCost(A=np.eye(nq), b=initial_q, vars=q_vars)

    prog.AddConstraint(constraint=com_constraint, vars=vars)
    # prog.AddConstraint(constraint=unit_quaternion_constraint, vars=q_vars[:4])
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
            boundary_time=20.0,
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    simulate_passive_robot(
        legged_model_type=legged_model_type,
    )
