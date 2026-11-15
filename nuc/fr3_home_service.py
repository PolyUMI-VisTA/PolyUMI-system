#!/usr/bin/env python3
"""
FR3 home/set-home services — runs ON THE NUC (ROS 2 Humble).

Serves /polyumi/home (std_srvs/Trigger): a joint-space move to the stored home pose (the SRDF
`ready` pose until one is taught), planned and executed through the LOCAL move_group. This node
must run on the NUC, not the laptop: the laptop (rmw_cyclonedds 4.0.2, Kilted) and the NUC
(rmw_cyclonedds 1.3.4, Humble) can exchange small messages fine, but the large nested MoveIt
action goals (MoveGroup.Goal / ExecuteTrajectory.Goal) get corrupted across the rmw-major boundary
("invalid data size, at serdata.cpp:384" -> move_group "Catastrophic failure"). Keeping the
move_group calls same-rmw (NUC-local) avoids that.

Self-contained (no PolyUMI package deps) so it runs from a plain clone on the NUC:
    source /opt/ros/humble/setup.bash
    source ~/franka_ws/install/setup.bash   # move_group + franka must be up (fr3-bringup)
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file://$HOME/franka_ws/config/cyclonedds.xml
    python3 nuc/fr3_home_service.py

Serves /polyumi/set_home (std_srvs/Trigger) alongside it: record wherever the arm is standing
right now as the pose /polyumi/home drives to, persisted so it survives the next bringup. The
pair is what makes eval trials repeatable -- teach the start pose once by moving the arm there,
then begin every trial with /polyumi/home.

Callable from the laptop despite the rmw gap, as long as the type is given explicitly (the ROS
*graph* does not cross Humble<->Kilted, so `ros2 node list` and node-name lookups come back
empty, but service calls match on DDS endpoints and work fine):

    ros2 service call /polyumi/set_home std_srvs/srv/Trigger "{}"   # teach (does NOT move)
    ros2 service call /polyumi/home     std_srvs/srv/Trigger "{}"   # MOVES THE ARM
"""

import datetime
import math
import os
import pathlib
import tempfile
import threading

from controller_manager_msgs.srv import SwitchController
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetMotionPlan
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import yaml

# The SRDF group homing plans in.
DEFAULT_GROUP = 'fr3_arm'

PLAN_TIMEOUT_S = 5.0
# How long to let a cancelled goal actually abort before giving up on it. Short: move_group stops
# the controller immediately, so this is deceleration, not motion.
CANCEL_TIMEOUT_S = 3.0

# --- Homing (the /polyumi/home service) ---
# The SRDF's own `ready` group state for fr3_arm (franka_fr3_moveit_config, group_definition.xacro).
# Overridable via the `home_joints` param when a task wants to start somewhere else.
HOME_JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]
HOME_JOINTS = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
HOME_TOLERANCE_RAD = 0.01
HOME_PLAN_TIME_S = 5.0
# Homing crosses the workspace, so it outlasts any short move. Generous on purpose: this is a
# wedge detector, and aborting a sweep that was merely slow is worse than waiting out a stuck one.
HOME_EXECUTE_TIMEOUT_S = 120.0

# --- Teaching the start pose (the /polyumi/set_home service) ---
# Evals need every trial to begin in the same place, and that place is task-specific rather than
# the SRDF `ready` pose. /polyumi/set_home captures wherever the arm is standing now, so the
# start pose is taught by putting the arm there once instead of being typed in as radians.
#
# Persisted beside the FAULHABER gripper's limits file and for the same reason: it is taught once
# by hand and has to survive the next bringup. Set the `home_pose_file` param to '' to keep the
# taught pose in memory only.
DEFAULT_HOME_POSE_FILE = '~/.ros/polyumi_home_pose.yaml'
HOME_POSE_FILE_VERSION = 1

JOINT_STATE_TOPIC = '/joint_states'
# A taught pose is only meaningful if it describes where the arm is *now*. The broadcaster runs
# at 30 Hz+, so a cache approaching a second old means it has stopped publishing and the
# positions are from wherever the arm used to be — which would be recorded as the start pose
# with nothing on the wire to say so.
JOINT_STATE_MAX_AGE_S = 1.0

# --- Controller handover ---
# move_group executes through fr3_arm_controller; the streaming impedance controller claims the
# same <joint>/effort interfaces, so the two are mutually exclusive and homing has to borrow the
# arm back. Switching restarts the libfranka control loop (franka_hardware's
# perform_command_mode_switch calls stopRobot() then re-initialises), so it must only happen with
# the arm stationary — which, at the start and end of a home, it is.
SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'
MOVEIT_CONTROLLER = 'fr3_arm_controller'
SWITCH_TIMEOUT_S = 5.0


class Fr3HomeService(Node):
    """Serve /polyumi/home and /polyumi/set_home: drive the FR3 to a taught start pose, and teach it."""

    def __init__(self, **kwargs):
        """Declare params, load any taught pose, and create the services and move_group clients."""
        super().__init__('fr3_home_service', **kwargs)

        self.declare_parameter('planning_group', DEFAULT_GROUP)
        self.declare_parameter('home_joints', HOME_JOINTS)
        self.declare_parameter('home_pose_file', DEFAULT_HOME_POSE_FILE)

        self._group = self.get_parameter('planning_group').get_parameter_value().string_value

        self._home_joints = list(self.get_parameter('home_joints').get_parameter_value().double_array_value)
        self._home_source = 'the home_joints parameter'

        pose_file = self.get_parameter('home_pose_file').get_parameter_value().string_value
        self._home_pose_file = pathlib.Path(pose_file).expanduser() if pose_file else None
        # A pose taught on the last run outranks the parameter default: it is the more recent and
        # more specific statement of where this task starts, and the operator who recorded it has
        # no other way to make it stick across a bringup.
        taught = self._load_home_pose()
        if taught is not None:
            self._home_joints = taught
            self._home_source = f'the taught pose in {self._home_pose_file}'

        self._cbgroup = ReentrantCallbackGroup()
        self._joint_plan = self.create_client(GetMotionPlan, 'plan_kinematic_path', callback_group=self._cbgroup)
        self._exec = ActionClient(self, ExecuteTrajectory, 'execute_trajectory', callback_group=self._cbgroup)
        self._switch = self.create_client(
            SwitchController, '/controller_manager/switch_controller', callback_group=self._cbgroup
        )
        self.create_service(Trigger, '/polyumi/home', self._on_home, callback_group=self._cbgroup)
        self.create_service(Trigger, '/polyumi/set_home', self._on_set_home, callback_group=self._cbgroup)

        # Cached with our own receive time rather than the message stamp: what set_home needs to
        # know is how long since we last heard anything, and that stays true even for a publisher
        # that leaves header.stamp at zero.
        self._joint_state_lock = threading.Lock()
        self._joint_state: JointState | None = None
        self._joint_state_at = None
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint_state, 10, callback_group=self._cbgroup)

        # Latest-goal, skip-while-busy: a concurrent /polyumi/home call while one is already
        # planning/executing is dropped rather than queued.
        self._busy = threading.Lock()

        # Fail loudly at startup rather than on the first /polyumi/home call.
        if not self._joint_plan.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                'plan_kinematic_path NOT found after 10s — move_group is probably not running '
                'on this NUC. Start it first: ros2 launch nuc/launch/fr3_move_group.launch.py '
                'robot_ip:=192.168.51.20'
            )
        else:
            self.get_logger().info('move_group found (plan_kinematic_path ready).')

        if not self._exec.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'execute_trajectory action server NOT found after 10s — move_group is probably not '
                'running on this NUC. Start it first: ros2 launch nuc/launch/fr3_move_group.launch.py '
                'robot_ip:=192.168.51.20'
            )
        else:
            self.get_logger().info('move_group found (execute_trajectory ready).')

        self.get_logger().info(
            'fr3_home_service started — /polyumi/home and /polyumi/set_home are up (std_srvs/Trigger).'
        )
        # Logged loudly, every start: a pose taught weeks ago is indistinguishable from the
        # default once the arm is moving, and this line is the only place it is visible before
        # the first trial.
        self.get_logger().info(f'Home pose from {self._home_source}: {self._format_joints(self._home_joints)}')

    def _on_home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Move the arm to the home joint pose. THIS MOVES THE ARM.

        An explicit, one-shot operator request, unconditional on anything else — there is no
        streamed-chunk path left for it to be confused with.

        Joint-space, not Cartesian: a Cartesian path from an arbitrary pose back to home is
        happy to drag the gripper straight through the table.
        """
        # Read once: /polyumi/set_home rebinds this attribute (atomically) from another thread,
        # and planning against a different pose than the one length-checked would be worse than
        # either outcome on its own.
        home_joints = self._home_joints
        if len(home_joints) != len(HOME_JOINT_NAMES):
            response.success = False
            response.message = f'home_joints has {len(home_joints)} values, expected {len(HOME_JOINT_NAMES)}'
            self.get_logger().error(response.message)
            return response
        if not self._busy.acquire(blocking=False):
            response.success = False
            response.message = 'busy: a plan/execute is already in flight'
            self.get_logger().warn(f'/polyumi/home refused — {response.message}')
            return response
        try:
            self.get_logger().warn(
                f'/polyumi/home called — MOVING THE ARM to the home pose from {self._home_source}: '
                f'{self._format_joints(home_joints)}'
            )
            # Borrow the arm from the streaming controller if it holds it. Not conditional on
            # having seen it start: this bridge and the controller come up independently, and a
            # switch naming an inactive controller is a no-op, so asking is cheaper than tracking.
            handed_over = self._switch_controllers(activate=MOVEIT_CONTROLLER, deactivate=SERVO_CONTROLLER)
            try:
                trajectory = self._plan_to_joints(home_joints)
                if trajectory is None:
                    response.success = False
                    response.message = 'planning to the home pose failed — see the bridge log'
                    return response
                if self._run_execute(trajectory, HOME_EXECUTE_TIMEOUT_S):
                    response.success = True
                    response.message = f'homed to the pose from {self._home_source}'
                    self.get_logger().info('Homed.')
                else:
                    response.success = False
                    response.message = 'execution failed — see the bridge log'
                return response
            finally:
                # Hand the arm back only if we took it. Leaving the servo deactivated after a
                # failed home would look like the policy silently doing nothing — and that
                # includes a home that itself SUCCEEDED: this must still turn a true `homed`
                # response into a false one if the hand-back is what failed. Mutating `response`
                # here reaches the caller even though `return response` above already fired —
                # Python evaluates that expression to the object reference first, then runs this
                # block, then returns the (now possibly-mutated) object.
                if handed_over and not self._switch_controllers(
                    activate=SERVO_CONTROLLER, deactivate=MOVEIT_CONTROLLER
                ):
                    response.success = False
                    response.message += (
                        f' — but failed to hand the arm back to {SERVO_CONTROLLER}; it is still on '
                        f'{MOVEIT_CONTROLLER} and the policy cannot drive it. Retry the switch '
                        'manually.'
                    )
        finally:
            self._busy.release()

    # ----------------------------------------------------------------------
    # Teaching the start pose
    # ----------------------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        """Cache the latest joint state, with the time we received it."""
        with self._joint_state_lock:
            self._joint_state = msg
            self._joint_state_at = self.get_clock().now()

    def _on_set_home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Record the arm's current joint positions as the pose /polyumi/home drives to.

        THIS DOES NOT MOVE THE ARM. Put the arm where trials should start — by hand in Desk's
        guiding mode, or by any other means — and then call this to capture it. Whatever moved
        the arm there is irrelevant: all this reads is /joint_states.

        Joint positions, not a TCP pose, because that is what /polyumi/home replays: capturing
        Cartesian coordinates would leave the elbow free to come back in a different
        configuration, which is exactly the trial-to-trial variation this exists to remove.
        """
        # Refuse mid-home rather than record a pose the arm is merely passing through. This also
        # keeps the two services from interleaving a rebind with a plan.
        if not self._busy.acquire(blocking=False):
            response.success = False
            response.message = 'busy: a plan/execute is already in flight — the arm is moving'
            self.get_logger().warn(f'/polyumi/set_home refused — {response.message}')
            return response
        try:
            positions, problem = self._arm_joints_now()
            if positions is None:
                response.success = False
                response.message = f'cannot read the arm pose: {problem}'
                self.get_logger().error(f'/polyumi/set_home failed — {response.message}')
                return response

            self._home_joints = positions
            self._home_source = 'the pose taught over /polyumi/set_home'
            formatted = self._format_joints(positions)

            saved, save_problem = self._save_home_pose(positions)
            if saved:
                self._home_source = f'the taught pose in {self._home_pose_file}'
            response.success = True
            response.message = f'recorded home pose: {formatted}'
            if not saved:
                # Still a success — the pose is live for this session. Only the persistence
                # failed, and an operator who is told that can re-teach after the next bringup
                # instead of discovering it when the first trial starts somewhere else.
                response.message += f' (in memory only — not saved: {save_problem})'
                self.get_logger().warn(f'/polyumi/set_home could not persist the pose: {save_problem}')
            self.get_logger().info(f'/polyumi/set_home — {response.message}')
            return response
        finally:
            self._busy.release()

    def _arm_joints_now(self) -> tuple[list[float] | None, str]:
        """Return the 7 arm joint positions from the cached joint state, or (None, why not)."""
        with self._joint_state_lock:
            msg = self._joint_state
            received_at = self._joint_state_at
        if msg is None:
            return None, (
                f'nothing has been published on {JOINT_STATE_TOPIC} — is franka_bringup up with '
                'the joint_state_broadcaster spawned?'
            )
        age_s = (self.get_clock().now() - received_at).nanoseconds / 1e9
        if age_s > JOINT_STATE_MAX_AGE_S:
            return None, (
                f'{JOINT_STATE_TOPIC} is stale ({age_s:.1f}s since the last message, limit '
                f'{JOINT_STATE_MAX_AGE_S:.1f}s) — the cached positions are not where the arm is now'
            )
        by_name = dict(zip(msg.name, msg.position))
        missing = [name for name in HOME_JOINT_NAMES if name not in by_name]
        if missing:
            return None, f'{JOINT_STATE_TOPIC} carries no {", ".join(missing)}'
        return [float(by_name[name]) for name in HOME_JOINT_NAMES], ''

    def _load_home_pose(self) -> list[float] | None:
        """Return the taught pose from disk, or None if there is none or it is unusable."""
        if self._home_pose_file is None or not self._home_pose_file.is_file():
            return None
        try:
            doc = yaml.safe_load(self._home_pose_file.read_text())
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(f'Ignoring {self._home_pose_file}: cannot read it ({exc}).')
            return None
        if not isinstance(doc, dict):
            self.get_logger().error(f'Ignoring {self._home_pose_file}: expected a mapping.')
            return None

        names = doc.get('joint_names')
        positions = doc.get('joint_positions')
        # Names are checked, not just the count: a file written against a different joint set
        # would otherwise be replayed onto fr3_joint1..7 by position and home somewhere else
        # entirely.
        if names != HOME_JOINT_NAMES:
            self.get_logger().error(
                f'Ignoring {self._home_pose_file}: joint_names is {names}, expected {HOME_JOINT_NAMES}.'
            )
            return None
        if not isinstance(positions, list) or len(positions) != len(HOME_JOINT_NAMES):
            self.get_logger().error(
                f'Ignoring {self._home_pose_file}: joint_positions must be {len(HOME_JOINT_NAMES)} values.'
            )
            return None
        try:
            return [float(value) for value in positions]
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f'Ignoring {self._home_pose_file}: joint_positions is not numeric ({exc}).')
            return None

    def _save_home_pose(self, positions: list[float]) -> tuple[bool, str]:
        """Persist the taught pose atomically; return (saved, why not)."""
        if self._home_pose_file is None:
            return False, 'home_pose_file is empty, so persistence is off'
        doc = {
            'version': HOME_POSE_FILE_VERSION,
            'joint_names': HOME_JOINT_NAMES,
            'joint_positions': positions,
            'recorded_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        }
        header = (
            '# PolyUMI eval start pose, written by /polyumi/set_home (nuc/fr3_home_service.py).\n'
            '# /polyumi/home drives the arm here. Delete this file to fall back to the\n'
            '# home_joints parameter (the SRDF `ready` pose by default).\n'
        )
        try:
            self._home_pose_file.parent.mkdir(parents=True, exist_ok=True)
            # Write-and-rename: a crash partway through a plain write leaves a truncated file that
            # the next startup refuses to parse, silently homing to the SRDF pose instead of the
            # taught one.
            fd, tmp_path = tempfile.mkstemp(dir=str(self._home_pose_file.parent), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as handle:
                    handle.write(header)
                    yaml.safe_dump(doc, handle, default_flow_style=False, sort_keys=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self._home_pose_file)
            except BaseException:
                # mkstemp's file outlives a failed write, and a stray .tmp beside the real file
                # is one more thing to explain later.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            return False, f'{self._home_pose_file}: {exc}'
        return True, ''

    @staticmethod
    def _format_joints(positions) -> str:
        """Render joint positions for a log line or a service response."""
        return '[' + ', '.join(f'{value:.4f}' for value in positions) + '] rad'

    def _switch_controllers(self, *, activate: str, deactivate: str) -> bool:
        """
        Swap which controller drives the arm; return whether the switch actually happened.

        STRICT, not BEST_EFFORT: a partial switch would leave the arm with either two controllers
        claiming its effort interfaces or none at all, and the second is indistinguishable from a
        working system that has simply stopped moving.

        A False return is not always an error — asking to deactivate a controller that was never
        spawned fails the same way, which is the normal case when running without the streaming
        controller at all.
        """
        if not self._switch.wait_for_service(timeout_sec=SWITCH_TIMEOUT_S):
            self.get_logger().warn(
                'controller_manager switch_controller not available; leaving controllers as they are.'
            )
            return False

        request = SwitchController.Request()
        request.activate_controllers = [activate]
        request.deactivate_controllers = [deactivate]
        request.strictness = SwitchController.Request.STRICT
        future = self._switch.call_async(request)
        if not self._wait(future, SWITCH_TIMEOUT_S):
            self.get_logger().error(f'switch to {activate} timed out after {SWITCH_TIMEOUT_S}s')
            return False

        ok = bool(future.result() and future.result().ok)
        if ok:
            self.get_logger().info(f'Controller switched: {deactivate} -> {activate}')
        else:
            self.get_logger().info(
                f'Controller switch {deactivate} -> {activate} declined; '
                f'{deactivate} is probably not running. Continuing.'
            )
        return ok

    def _plan_to_joints(self, positions: list[float]) -> RobotTrajectory | None:
        """Plan a collision-checked joint-space move to `positions`; return the trajectory or None."""
        if not self._joint_plan.service_is_ready():
            self.get_logger().error(
                'plan_kinematic_path is NOT available — is move_group running on this NUC? '
                '(ros2 launch nuc/launch/fr3_move_group.launch.py robot_ip:=192.168.51.20)'
            )
            return None

        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = self._group
        mpr.num_planning_attempts = 10
        mpr.allowed_planning_time = HOME_PLAN_TIME_S
        # Speed is left alone: MotionPlanRequest documents that a scaling factor outside (0,1] —
        # 0.0 included, which is what the unset field holds — is treated as 1.0, so this plans at
        # full speed against the URDF's joint velocity limits and fr3_move_group's
        # max_acceleration. Homing is an operator repositioning the arm, not a policy motion;
        # there is nothing here to run slowly for, and those two limits are the real ceiling.
        goal = Constraints()
        for name, position in zip(HOME_JOINT_NAMES, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = HOME_TOLERANCE_RAD
            jc.tolerance_below = HOME_TOLERANCE_RAD
            jc.weight = 1.0
            goal.joint_constraints.append(jc)
        mpr.goal_constraints.append(goal)

        future = self._joint_plan.call_async(req)
        if not self._wait(future, HOME_PLAN_TIME_S + PLAN_TIMEOUT_S):
            self.get_logger().warning('Joint-space planning timed out.')
            return None
        resp = future.result()
        if resp is None or resp.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if resp is None else resp.motion_plan_response.error_code.val
            self.get_logger().warning(f'Joint-space planning failed (error_code={code}).')
            return None
        return resp.motion_plan_response.trajectory

    def _run_execute(self, trajectory: RobotTrajectory, timeout_s: float) -> bool:
        """Execute a planned trajectory via ExecuteTrajectory; block until done."""
        # Same rationale as _plan_to_joints's service_is_ready() check: send_goal_async on a
        # server that isn't there hangs until PLAN_TIMEOUT_S instead of failing immediately,
        # holding the busy lock past when the caller would otherwise give up.
        if not self._exec.server_is_ready():
            self.get_logger().error(
                'execute_trajectory action server is NOT available — is move_group running on this '
                'NUC? (ros2 launch nuc/launch/fr3_move_group.launch.py robot_ip:=192.168.51.20)'
            )
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        gf = self._exec.send_goal_async(goal)
        if not self._wait(gf, PLAN_TIMEOUT_S):
            self.get_logger().warning('Execute goal submission timed out.')
            return False
        gh = gf.result()
        if gh is None or not gh.accepted:
            self.get_logger().warning('Execute goal rejected.')
            return False
        rf = gh.get_result_async()
        if not self._wait(rf, timeout_s):
            # Cancel, and wait for the abort to land. Returning here releases the caller's busy
            # lock, and an uncancelled goal keeps driving the arm — so a follow-up /polyumi/home
            # would plan from a start state the arm has already left. Bounded, because a server
            # that ignores the cancel must not wedge the bridge instead.
            self.get_logger().warning(f'Execution timed out after {timeout_s:.0f}s — cancelling.')
            gh.cancel_goal_async()
            if not self._wait(rf, CANCEL_TIMEOUT_S):
                self.get_logger().error(
                    'Cancel did not take effect — the arm may still be moving. Stop it from the '
                    'Desk UI before sending anything else.'
                )
            return False
        res = rf.result()
        if res is None or res.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if res is None else res.result.error_code.val
            self.get_logger().warning(f'Execution failed (error_code={code}).')
            return False
        return True

    @staticmethod
    def _wait(future, timeout_s: float) -> bool:
        """Block until future completes or times out (node spins under a MultiThreadedExecutor)."""
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        return done.wait(timeout=timeout_s)


def main():
    """Spin the bridge node under a multithreaded executor."""
    rclpy.init()
    node = Fr3HomeService()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
