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
        # Left leg
        f[6], f[7] = f[7], f[6]
        f[7], f[8] = f[8], f[7]
        # Right leg
        f[11], f[12] = f[12], f[11]
        f[12], f[13] = f[13], f[12]
        # Left arm
        f[17], f[18] = f[18], f[17]
        # Right arm
        f[21], f[22] = f[22], f[21]

        f = 1 * f[-19:]

        output.SetFromVector(f)


class Mixer(LeafSystem):
    def __init__(
        self,
    ):
        LeafSystem.__init__(self)

        self.o_ip = self.DeclareVectorInputPort(
            name="in",
            size=19,
        )
        self.o_op = self.DeclareVectorOutputPort("out", 19, self.compute_o)

    def compute_o(self, context: Context, output):

        o_vector = self.o_ip.Eval(context)

        # Left leg
        o_vector[0], o_vector[1] = o_vector[1], o_vector[0]
        o_vector[1], o_vector[2] = o_vector[2], o_vector[1]
        # Right leg
        o_vector[5], o_vector[6] = o_vector[6], o_vector[5]
        o_vector[6], o_vector[7] = o_vector[7], o_vector[6]
        # Left arm
        o_vector[11], o_vector[12] = o_vector[12], o_vector[11]
        # Right arm
        o_vector[15], o_vector[16] = o_vector[16], o_vector[15]

        output.SetFromVector(o_vector)


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
    kp = np.zeros(19, dtype=np.float64)
    ki = np.zeros(19, dtype=np.float64)
    kd = np.zeros(19, dtype=np.float64)

    kp[:] = 5000.0
    ki[:] = 5000.0
    kd[:] = 10.0
    # Hip roll
    # kp[0] = 2000.0
    # kp[5] = 2000.0
    # ki[0] = 50.0
    # ki[5] = 50.0
    ## Hip pitch
    # kp[1] = 2000.0
    # kp[6] = 2000.0
    # ki[1] = 50.0
    # ki[6] = 50.0
    ## Hip yaw
    # kp[2] = 2000.0
    # kp[7] = 2000.0
    # ki[2] = 50.0
    # ki[7] = 50.0
    ## Knees
    # kp[3] = 2000.0
    # kp[8] = 2000.0
    # ki[3] = 100.0
    # ki[8] = 100.0
    # Ankle
    # kp[4] = 50.0
    # kp[9] = 50.0
    # ki[4] = 50.0
    # ki[9] = 50.0
    # kd[4] = 0.0
    # kd[9] = 0.0

    for ii, pn in enumerate(plant.GetPositionNames()):
        print(ii, pn)
    for ii, vn in enumerate(plant.GetVelocityNames()):
        print(ii, vn)
    for ii, an in enumerate(plant.GetActuatorNames()):
        print(ii, an)

    pid: PidController = builder.AddSystem(
        PidController(
            kp=kp,
            ki=ki,
            kd=kd,
        ),
    )
    mix: Mixer = builder.AddSystem(Mixer())

    q = plant.GetDefaultPositions()
    qdot = np.zeros(19)
    s = np.hstack((q[-19:], qdot))

    demux_estimated = builder.AddSystem(
        Demultiplexer(output_ports_sizes=[7, 19, 6, 19]),
    )
    mux = builder.AddSystem(
        Multiplexer(input_sizes=[19, 19]),
    )
    cvs = builder.AddSystem(
        ConstantVectorSource(
            source_value=s,
        ),
    )

    gc_plant = MultibodyPlant(0.001)
    add_legged_model_to_plant_and_finalize(
        plant=gc_plant,
        legged_model_type=legged_model_type,
    )

    # gc = builder.AddSystem(
    #    GravityComp(gc_plant),
    # )

    builder.Connect(
        plant.get_state_output_port(),
        demux_estimated.get_input_port(),
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
        mux.get_output_port(),
        pid.get_input_port_estimated_state(),
    )

    builder.Connect(
        cvs.get_output_port(),
        pid.get_input_port_desired_state(),
    )
    builder.Connect(
        pid.get_output_port_control(),
        mix.get_input_port(),
    )
    builder.Connect(
        mix.get_output_port(),
        plant.get_actuation_input_port(),
    )

    # builder.Connect(
    #    plant.get_state_output_port(),
    #    gc.get_input_port(),
    # )
    # builder.Connect(
    #    gc.get_output_port(),
    #    plant.get_actuation_input_port(),
    # )

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
