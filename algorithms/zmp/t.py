import numpy as np
from pydrake.all import DiagramBuilder, StartMeshcat
from pydrake.geometry import Meshcat, MeshcatVisualizer
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ContactModel,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamics, PidController
from pydrake.systems.framework import Context, LeafSystem
from pydrake.systems.primitives import (
    ConstantVectorSource,
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


class GravityComp(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
    ):
        LeafSystem.__init__(self)
        self.plant = plant

        self.state_ip = self.DeclareVectorInputPort(
            name="in",
            size=plant.num_positions() + plant.num_velocities(),
        )
        self.gc_op = self.DeclareVectorOutputPort(
            "out", plant.num_actuators(), self.compute_c
        )

    def compute_c(self, context: Context, output):

        state_ip_vector = self.state_ip.Eval(context)

        pc = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(pc, state_ip_vector)
        f = self.plant.CalcGravityGeneralizedForces(
            context=pc,
        )
        f = -1 * f[-19:]
        output.SetFromVector(f)


def simulate_zmp_walking(
    legged_model_type: LeggedModelType,
) -> None:
    meshcat = Meshcat()
    meshcat.SetRealtimeRate(0.1)
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
    plant.SetVelocities(plant.CreateDefaultContext(), np.zeros(25))

    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    # TODO: Use state projection in PID instead of demuxing.
    kp = 200.0
    ki = 50.0
    kd = 0.0
    # pid: PidController = builder.AddSystem(
    #    PidController(
    #        kp=np.full(19, kp),
    #        ki=np.full(19, ki),
    #        kd=np.full(19, kd),
    #    ),
    # )

    # q = plant.GetDefaultPositions()
    # qdot = np.zeros(19)
    # s = np.hstack((q[-19:], qdot))

    demux_estimated = builder.AddSystem(
        Demultiplexer(output_ports_sizes=[7, 19, 6, 19]),
    )
    mux = builder.AddSystem(
        Multiplexer(input_sizes=[19, 19]),
    )
    # cvs = builder.AddSystem(
    #    ConstantVectorSource(
    #        source_value=s,
    #    ),
    # )

    gc_plant = MultibodyPlant(0.001)
    add_legged_model_to_plant_and_finalize(
        plant=gc_plant,
        legged_model_type=legged_model_type,
    )

    gc = builder.AddSystem(
        GravityComp(gc_plant),
    )

    # builder.Connect(
    #    plant.get_state_output_port(),
    #    demux_estimated.get_input_port(),
    # )
    # builder.Connect(
    #    demux_estimated.get_output_port(1),
    #    mux.get_input_port(0),
    # )
    # builder.Connect(
    #    demux_estimated.get_output_port(3),
    #    mux.get_input_port(1),
    # )

    # builder.Connect(
    #    mux.get_output_port(),
    #    pid.get_input_port_estimated_state(),
    # )

    # builder.Connect(
    #    cvs.get_output_port(),
    #    pid.get_input_port_desired_state(),
    # )
    # builder.Connect(
    #    pid.get_output_port_control(),
    #    plant.get_actuation_input_port(),
    # )

    #builder.Connect(
    #    plant.get_state_output_port(),
    #    gc.get_input_port(),
    #)
    #builder.Connect(
    #    gc.get_output_port(),
    #    plant.get_actuation_input_port(),
    #)

    diagram = builder.Build()
    simulator = Simulator(system=diagram)

    with auto_meshcat_visualization(meshcat=meshcat, record=True):
        simulator.AdvanceTo(
            boundary_time=5.0,
            interruptible=True,
        )


if __name__ == "__main__":

    legged_model_type = LeggedModelType.H1
    simulate_zmp_walking(
        legged_model_type=legged_model_type,
    )
