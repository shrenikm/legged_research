import attr
import numpy as np
from pydrake.multibody.inverse_kinematics import (
    AngleBetweenVectorsConstraint,
    ComPositionConstraint,
    PointToPointDistanceConstraint,
    UnitQuaternionConstraint,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import Context
from pydrake.trajectories import PiecewisePolynomial

from algorithms.zmp.utils import FootstepType
from algorithms.zmp.zmp_planners import NaiveZMPPlanner, ZMPPlannerResult
from common.custom_types import NpArrayMNf64, NpVector3f64, PositionsVector
from common.model_utils import (
    LeggedModelType,
    add_legged_model_to_plant_and_finalize,
    get_left_foot_frame_name,
    get_left_foot_polygon,
    get_right_foot_frame_name,
    get_right_foot_polygon,
)


@attr.frozen
class ZMPIKPlanner:

    legged_model_type: LeggedModelType
    plant: MultibodyPlant
    plant_context: Context
    sample_time_s: float
    alpha: float

    _nq: int = attr.ib(init=False)
    _lf_name: str = attr.ib(init=False)
    _rf_name: str = attr.ib(init=False)
    _alpha_pos: NpVector3f64 = attr.ib(init=False)

    @_nq.default
    def _initialize_nq(self) -> int:
        return self.plant.num_positions()

    @_lf_name.default
    def _initialize_lf_name(self) -> str:
        return get_left_foot_frame_name(legged_model_type=self.legged_model_type)

    @_rf_name.default
    def _initialize_rf_name(self) -> str:
        return get_right_foot_frame_name(legged_model_type=self.legged_model_type)

    # TODO: Set up anchor points for feet.
    @_alpha_pos.default
    def _initialize_alpha_pos(self) -> NpVector3f64:
        return np.array([self.alpha, 0.0, 0.0])

    def _solve_single_phase(
        self,
        zmp_result: ZMPPlannerResult,
        start_time: float,
        end_time: float,
        initial_q: PositionsVector,
    ) -> NpArrayMNf64:
        assert end_time > start_time
        nk = int((end_time - start_time) / self.sample_time_s)
        print(nk)

        prog = MathematicalProgram()
        q_vars_matrix = prog.NewContinuousVariables(
            rows=self._nq, cols=nk, name="q_vars"
        )
        com_vars_matrix = prog.NewContinuousVariables(rows=3, cols=nk, name="com_vars")

        for j in range(nk):
            t = start_time + j * self.sample_time_s
            q_vars = q_vars_matrix[:, j]
            com_vars = com_vars_matrix[:, j]
            com_desired = zmp_result.com_trajectory.value(t=t).reshape(3)
            left_foot_xyztheta = zmp_result.left_foot_trajectory.value(t=t).reshape(4)
            right_foot_xyztheta = zmp_result.right_foot_trajectory.value(t=t).reshape(4)
            left_foot_xyz = left_foot_xyztheta[:3]
            right_foot_xyz = right_foot_xyztheta[:3]

            p2p_distance_ub = 0.01
            a2a_ub = 0.01

            # TODO: Need to incorporate theta
            c1 = PointToPointDistanceConstraint(
                plant=self.plant,
                frame1=self.plant.GetFrameByName(self._lf_name),
                p_B1P1=self._alpha_pos,
                frame2=self.plant.world_frame(),
                p_B2P2=left_foot_xyz + self._alpha_pos,
                distance_lower=0.0,
                distance_upper=p2p_distance_ub,
                plant_context=self.plant_context,
            )
            c2 = PointToPointDistanceConstraint(
                plant=self.plant,
                frame1=self.plant.GetFrameByName(self._lf_name),
                p_B1P1=-self._alpha_pos,
                frame2=self.plant.world_frame(),
                p_B2P2=left_foot_xyz - self._alpha_pos,
                distance_lower=0.0,
                distance_upper=p2p_distance_ub,
                plant_context=self.plant_context,
            )
            c3 = PointToPointDistanceConstraint(
                plant=self.plant,
                frame1=self.plant.GetFrameByName(self._rf_name),
                p_B1P1=self._alpha_pos,
                frame2=self.plant.world_frame(),
                p_B2P2=right_foot_xyz + self._alpha_pos,
                distance_lower=0.0,
                distance_upper=p2p_distance_ub,
                plant_context=self.plant_context,
            )
            c4 = PointToPointDistanceConstraint(
                plant=self.plant,
                frame1=self.plant.GetFrameByName(self._rf_name),
                p_B1P1=-self._alpha_pos,
                frame2=self.plant.world_frame(),
                p_B2P2=right_foot_xyz - self._alpha_pos,
                distance_lower=0.0,
                distance_upper=p2p_distance_ub,
                plant_context=self.plant_context,
            )
            c5 = AngleBetweenVectorsConstraint(
                plant=self.plant,
                # TODO: Function.
                frameA=self.plant.GetFrameByName("torso_link"),
                a_A=np.array([0.0, 0.0, 1.0]),
                frameB=self.plant.world_frame(),
                b_B=np.array([0.0, 0.0, 1.0]),
                angle_lower=0.0,
                angle_upper=a2a_ub,
                plant_context=self.plant_context,
            )
            c6 = ComPositionConstraint(
                plant=self.plant,
                model_instances=None,
                expressed_frame=self.plant.world_frame(),
                plant_context=self.plant_context,
            )
            c7 = UnitQuaternionConstraint()

            prog.AddConstraint(c1, vars=q_vars)
            prog.AddConstraint(c2, vars=q_vars)
            prog.AddConstraint(c3, vars=q_vars)
            prog.AddConstraint(c4, vars=q_vars)
            prog.AddConstraint(c5, vars=q_vars)
            if j > 0:
                prog.AddConstraint(c6, vars=np.hstack((q_vars, com_vars)))
            # TODO: Maybe don't need quaternion constraint.
            prog.AddConstraint(c7, vars=q_vars[:4])

            # Fix everything other than the leg joints.
            for i in range(17, self._nq):
                prog.AddConstraint(q_vars[i] == 0)
            # COM constraint.
            for i in range(3):
                prog.AddConstraint(com_vars[i] == com_desired[i])

        for i in range(self._nq):
            prog.AddConstraint(q_vars_matrix[i, 0] == initial_q[i])
        for j in range(nk - 1):
            for i in range(7, 17):
                prog.AddCost(
                    1.0 * (q_vars_matrix[i, j + 1] - q_vars_matrix[i, j]) ** 2.0
                )

        for j in range(nk):
            prog.SetInitialGuess(q_vars_matrix[:, j], initial_q)

        result = Solve(prog=prog)
        print("Success:", result.is_success())
        print("Solution result:", result.get_solution_result())
        print(result.GetInfeasibleConstraintNames(prog))

        return result.GetSolution(q_vars_matrix)

    def compute_positions(
        self,
        zmp_result: ZMPPlannerResult,
    ) -> NpArrayMNf64:

        zt = zmp_result.zmp_trajectory

        knot_times = zt.get_segment_times()

        # Ignoring the last few know points where we won't have the com trajectory due to
        # preview control
        knot_times = knot_times[:-2]

        initial_q = self.plant.GetDefaultPositions()

        solution_q_matrix = np.empty((self._nq, 0), dtype=np.float64)

        for i in range(len(knot_times) - 1):

            start_time = knot_times[i]
            end_time = knot_times[i + 1]

            sol_q_mat = self._solve_single_phase(
                zmp_result=zmp_result,
                start_time=start_time,
                end_time=end_time,
                initial_q=initial_q,
            )
            solution_q_matrix = np.hstack((solution_q_matrix, sol_q_mat))

            initial_q = sol_q_mat[:, -1]

        return solution_q_matrix


def solve_straight_line_walking(
    legged_model_type: LeggedModelType,
    plant_time_step: float,
    path_length: float,
    ik_sample_time: float,
) -> PiecewisePolynomial:
    ik_plant = MultibodyPlant(plant_time_step)
    add_legged_model_to_plant_and_finalize(
        plant=ik_plant,
        legged_model_type=legged_model_type,
    )
    ik_plant_context = ik_plant.CreateDefaultContext()
    initial_com = ik_plant.CalcCenterOfMassPositionInWorld(ik_plant_context)

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
    x_path = np.arange(x_start, x_start + path_length, step=0.05)
    xy_path = np.vstack((x_path, np.zeros_like(x_path))).T
    zmp_result = nzp.compute_full_zmp_result(
        xy_path=xy_path,
        initial_com=initial_com,
        first_footstep=FootstepType.RIGHT,
        # debug=True,
    )

    ikzp = ZMPIKPlanner(
        legged_model_type=legged_model_type,
        plant=ik_plant,
        plant_context=ik_plant_context,
        sample_time_s=ik_sample_time,
        alpha=0.1,
    )
    positions = ikzp.compute_positions(
        zmp_result=zmp_result,
    )

    breaks = np.arange(0.0, positions.shape[1] * ik_sample_time, ik_sample_time)
    return PiecewisePolynomial.FirstOrderHold(
        breaks=breaks,
        samples=positions,
    )
