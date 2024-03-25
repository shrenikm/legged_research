import numpy as np
from pydrake.all import DiagramBuilder, StartMeshcat
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ContactModel,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import PidController
from pydrake.systems.primitives import (
    Demultiplexer,
    Multiplexer,
    StateInterpolatorWithDiscreteDerivative,
    TrajectorySource,
)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.visualization import AddDefaultVisualization

from algorithms.zmp.ik_planners import solve_straight_line_walking
from common.drake_utils import auto_meshcat_visualization
from common.model_utils import LeggedModelType, add_legged_model_to_plant_and_finalize


def simulate_zmp_walking(
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

    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    path_length = 1.5
    ik_sample_time = 0.1
    q_traj = solve_straight_line_walking(
        legged_model_type=legged_model_type,
        plant_time_step=config.time_step,
        path_length=path_length,
        ik_sample_time=ik_sample_time,
    )

    # TODO: Cleanup
    q_source = builder.AddSystem(
        TrajectorySource(q_traj),
    )
    interp = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_positions=19,
            time_step=config.time_step,
        ),
    )
    demux_estimated = builder.AddSystem(
        Demultiplexer(output_ports_sizes=[7, 19, 6, 19]),
    )
    demux_desired = builder.AddSystem(
        Demultiplexer(output_ports_sizes=[7, 19]),
    )
    mux = builder.AddSystem(
        Multiplexer(input_sizes=[19, 19]),
    )
    # TODO: Use state projection in PID instead of demuxing.
    kp = 20.0
    ki = 60.0
    kd = 0.0
    pid: PidController = builder.AddSystem(
        PidController(
            kp=np.full(19, kp),
            ki=np.full(19, ki),
            kd=np.full(19, kd),
        ),
    )

    builder.Connect(
        plant.get_state_output_port(),
        demux_estimated.get_input_port(),
    )
    builder.Connect(
        q_source.get_output_port(),
        demux_desired.get_input_port(),
    )
    builder.Connect(
        demux_estimated.get_output_port(1),
        mux.get_input_port(0),
    )
    builder.Connect(
        demux_estimated.get_output_port(3),
        mux.get_input_port(1),
    )
    builder.Connect(
        demux_desired.get_output_port(1),
        interp.get_input_port(),
    )

    builder.Connect(
        mux.get_output_port(),
        pid.get_input_port_estimated_state(),
    )
    builder.Connect(
        interp.get_output_port(),
        pid.get_input_port_desired_state(),
    )
    builder.Connect(
        pid.get_output_port_control(),
        plant.get_actuation_input_port(),
    )

    diagram = builder.Build()
    simulator = Simulator(system=diagram)

    print(q_traj.end_time())
    with auto_meshcat_visualization(meshcat=meshcat, record=True):
        simulator.AdvanceTo(
            boundary_time=q_traj.end_time(),
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    simulate_zmp_walking(
        legged_model_type=legged_model_type,
    )
