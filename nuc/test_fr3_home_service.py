"""
Tests for the NUC-side MoveIt bridge's /polyumi/home and /polyumi/set_home services.

Homing is the one path here that moves the arm on an explicit request rather than on a streamed
chunk, and every way it can be wrong is silent: a wrong joint name plans to a pose that is not
home, the plan-only gate would make it a no-op in the default configuration, and the shorter
chunk timeout would abort a long joint-space sweep partway across the workspace.

Runs on the laptop despite the bridge targeting the Humble NUC: only the moveit_msgs *message
definitions* are needed. No move_group is involved — the service and action clients are mocked,
so these tests exercise the bridge's logic and nothing else.

    bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \
      && source ros2_ws/install/setup.bash \
      && /usr/bin/python3 -m pytest nuc/test_fr3_home_service.py -q'
"""

from unittest.mock import MagicMock, patch

from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory
import pytest
import rclpy
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import yaml

import fr3_home_service as mb


@pytest.fixture(scope='module', autouse=True)
def ros():
    """Init rclpy once for the module; every node here is constructed without a real executor."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node(tmp_path):
    """
    Build a bridge whose move_group clients are mocks, so no server or executor is required.

    create_client is patched to hand back a fresh mock per call — the real one would make
    __init__ block for its two 10 s wait_for_service/wait_for_server timeouts.

    `home_pose_file` defaults into tmp_path: the bridge loads a taught pose at construction, and
    without this every test on a machine that has ever run /polyumi/set_home would silently home
    to whatever is in the developer's ~/.ros instead of the SRDF pose it asserts against.
    """
    nodes = []

    def _make(**overrides):
        overrides.setdefault('home_pose_file', str(tmp_path / 'home_pose.yaml'))
        params = [Parameter(k, value=v) for k, v in overrides.items()]
        with (
            patch.object(mb.Fr3HomeService, 'create_client', side_effect=lambda *a, **k: MagicMock()),
            patch.object(mb, 'ActionClient') as action_client,
        ):
            action_client.return_value.wait_for_server.return_value = True
            action_client.return_value.server_is_ready.return_value = True
            node = mb.Fr3HomeService(parameter_overrides=params)
        node.get_logger = MagicMock()
        # Futures never resolve against a mocked client, so _wait would burn its full timeout.
        node._wait = lambda future, timeout_s: True
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()


def _stub_plan_ok(node, trajectory='TRAJ'):
    """Make the joint-space planner answer SUCCESS with `trajectory`."""
    node._joint_plan.service_is_ready.return_value = True
    resp = MagicMock()
    resp.motion_plan_response.error_code.val = MoveItErrorCodes.SUCCESS
    resp.motion_plan_response.trajectory = trajectory
    node._joint_plan.call_async.return_value.result.return_value = resp
    return resp


def _capture_execute(node) -> list:
    """Replace _run_execute with a recorder, returning the list it appends (trajectory, timeout) to."""
    calls = []

    def _record(trajectory, timeout_s):
        calls.append((trajectory, timeout_s))
        return True

    node._run_execute = _record
    return calls


def _home(node) -> Trigger.Response:
    """Call the service handler directly, as rclpy would."""
    return node._on_home(Trigger.Request(), Trigger.Response())


def test_home_plans_to_the_srdf_ready_pose(make_node):
    """The goal constraint must name fr3_joint1..7 at the SRDF `ready` values, in order."""
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)

    assert _home(node).success

    request = node._joint_plan.call_async.call_args[0][0]
    constraints = request.motion_plan_request.goal_constraints[0].joint_constraints
    assert [c.joint_name for c in constraints] == [f'fr3_joint{i}' for i in range(1, 8)]
    assert [c.position for c in constraints] == pytest.approx(mb.HOME_JOINTS)
    assert request.motion_plan_request.group_name == mb.DEFAULT_GROUP


def test_home_moves_the_arm_even_in_plan_only_mode(make_node):
    """`execute` gates streamed chunks, not an explicit home request — see _on_home's docstring."""
    node = make_node(execute=False)
    _stub_plan_ok(node)
    executed = _capture_execute(node)

    assert _home(node).success
    assert len(executed) == 1, 'plan-only mode must not suppress an explicit /polyumi/home'


def test_home_uses_the_long_execute_timeout(make_node):
    """A joint-space sweep at low velocity scaling needs far longer than a short move."""
    node = make_node()
    _stub_plan_ok(node)
    executed = _capture_execute(node)

    _home(node)

    assert executed[0][1] == mb.HOME_EXECUTE_TIMEOUT_S


def test_home_refused_while_a_chunk_is_in_flight(make_node):
    """A concurrent /polyumi/home call must not cut in on one already planning/executing."""
    node = make_node()
    _stub_plan_ok(node)
    executed = _capture_execute(node)
    node._busy.acquire()
    try:
        response = _home(node)
    finally:
        node._busy.release()

    assert not response.success
    assert 'busy' in response.message
    assert executed == []


def test_home_releases_the_busy_lock_after_a_failed_plan(make_node):
    """A planning failure must not wedge the bridge — the next chunk still has to get through."""
    node = make_node()
    node._joint_plan.service_is_ready.return_value = False  # move_group missing
    _capture_execute(node)

    assert not _home(node).success
    assert node._busy.acquire(blocking=False), 'busy lock left held after a failed home'
    node._busy.release()


def test_home_rejects_a_wrong_length_home_joints(make_node):
    """A 6-value override would otherwise zip() short and silently home to a partial pose."""
    node = make_node(home_joints=[0.0] * 6)
    _stub_plan_ok(node)
    executed = _capture_execute(node)

    response = _home(node)

    assert not response.success
    assert 'expected 7' in response.message
    assert executed == []


def test_execute_timeout_cancels_the_goal_and_waits_for_the_abort(make_node):
    """
    A timed-out goal keeps driving the arm, and the caller releases the busy lock on the way out.

    Without a cancel the next chunk plans a Cartesian path from a start state the arm has already
    left. The wait after the cancel matters just as much: returning while it decelerates puts us
    back in the same race, one deceleration narrower.
    """
    node = make_node(execute=True)
    waits = []

    def _wait(_future, timeout_s):
        """Time out only on the second wait — the goal result — and succeed on the others."""
        waits.append(timeout_s)
        return len(waits) != 2

    node._wait = _wait

    assert not node._run_execute(RobotTrajectory(), timeout_s=7.0)

    goal_handle = node._exec.send_goal_async.return_value.result.return_value
    goal_handle.cancel_goal_async.assert_called_once()
    assert waits == [mb.PLAN_TIMEOUT_S, 7.0, mb.CANCEL_TIMEOUT_S], (
        'must wait for the abort to land, bounded by CANCEL_TIMEOUT_S'
    )


def test_execute_success_does_not_cancel(make_node):
    """The cancel path must not fire on the happy path — it would abort every good chunk."""
    node = make_node(execute=True)
    result = node._exec.send_goal_async.return_value.result.return_value.get_result_async.return_value
    result.result.return_value.result.error_code.val = MoveItErrorCodes.SUCCESS

    assert node._run_execute(RobotTrajectory(), timeout_s=7.0)

    node._exec.send_goal_async.return_value.result.return_value.cancel_goal_async.assert_not_called()


# ----------------------------------------------------------------------
# Controller handover
# ----------------------------------------------------------------------


def _stub_switch(node, ok=True):
    """Make /controller_manager/switch_controller answer `ok`, and record the requests it got."""
    node._switch.wait_for_service.return_value = True
    result = MagicMock()
    result.ok = ok
    node._switch.call_async.return_value.result.return_value = result
    return node._switch.call_async


def _switch_pairs(call_async) -> list:
    """Return the (activate, deactivate) pairs the bridge asked for, in order."""
    return [
        (tuple(c[0][0].activate_controllers), tuple(c[0][0].deactivate_controllers)) for c in call_async.call_args_list
    ]


def _stub_switch_sequence(node, oks: list[bool]):
    """Like _stub_switch, but each successive switch_controller call answers the next `oks` value."""
    node._switch.wait_for_service.return_value = True
    futures = []
    for ok in oks:
        result = MagicMock()
        result.ok = ok
        future = MagicMock()
        future.result.return_value = result
        futures.append(future)
    node._switch.call_async.side_effect = futures
    return node._switch.call_async


def test_home_borrows_the_arm_from_the_servo_and_gives_it_back(make_node):
    """
    Homing must borrow the arm from the streaming controller and then give it back.

    Homing goes through move_group, which drives fr3_arm_controller; the streaming impedance
    controller claims the same effort interfaces, so exactly one can hold the arm. Failing to hand
    it back would leave the policy silently commanding nothing after every home.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch(node)

    assert _home(node).success

    assert _switch_pairs(call_async) == [
        ((mb.MOVEIT_CONTROLLER,), (mb.SERVO_CONTROLLER,)),
        ((mb.SERVO_CONTROLLER,), (mb.MOVEIT_CONTROLLER,)),
    ]


def test_home_switches_strictly(make_node):
    """
    The handover must be STRICT.

    BEST_EFFORT could half-apply, leaving the arm with two controllers claiming its efforts or
    none — and "none" looks exactly like a working system that has stopped moving.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch(node)

    _home(node)

    request = call_async.call_args_list[0][0][0]
    assert request.strictness == mb.SwitchController.Request.STRICT


def test_home_hands_the_arm_back_even_when_planning_fails(make_node):
    """A failed home must not strand the servo deactivated — that is a silent dead policy."""
    node = make_node()
    node._joint_plan.service_is_ready.return_value = False  # planner unavailable
    call_async = _stub_switch(node)

    assert not _home(node).success

    assert _switch_pairs(call_async)[-1] == ((mb.SERVO_CONTROLLER,), (mb.MOVEIT_CONTROLLER,))


def test_home_reports_failure_if_the_servo_cannot_be_reactivated(make_node):
    """
    A successful home must not report success if the hand-back to the servo then fails.

    Otherwise the caller sees `homed: true` while the arm is left on fr3_arm_controller, unable to
    take a target chunk from the policy — the same silent-dead-policy failure the handover exists
    to avoid, just arriving one step later than the failure modes above cover.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch_sequence(node, [True, False])  # borrow succeeds, hand-back fails

    response = _home(node)

    assert not response.success
    assert 'failed to hand the arm back' in response.message
    assert _switch_pairs(call_async) == [
        ((mb.MOVEIT_CONTROLLER,), (mb.SERVO_CONTROLLER,)),
        ((mb.SERVO_CONTROLLER,), (mb.MOVEIT_CONTROLLER,)),
    ]


def test_home_without_the_servo_running_does_not_switch_back(make_node):
    """
    A declined handover must not be followed by a switch back.

    Running the MoveIt executor alone is the normal pre-Phase-4 configuration. There the first
    switch is declined (no such controller), and activating the servo afterwards would be wrong —
    it was never holding the arm.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch(node, ok=False)

    assert _home(node).success, 'a declined handover must not fail the home itself'

    assert _switch_pairs(call_async) == [((mb.MOVEIT_CONTROLLER,), (mb.SERVO_CONTROLLER,))]


# ----------------------------------------------------------------------
# Teaching the start pose (/polyumi/set_home)
# ----------------------------------------------------------------------

TAUGHT = [0.11, -0.22, 0.33, -1.44, 0.55, 1.66, 0.77]


def _joint_state(names, positions) -> JointState:
    """Build a JointState carrying these joint names and positions."""
    msg = JointState()
    msg.name = list(names)
    msg.position = [float(p) for p in positions]
    return msg


def _feed(node, positions=TAUGHT, names=None):
    """Deliver a joint state to the bridge as the broadcaster would."""
    node._on_joint_state(_joint_state(names if names is not None else mb.HOME_JOINT_NAMES, positions))


def _set_home(node) -> Trigger.Response:
    """Call the teach handler directly, as rclpy would."""
    return node._on_set_home(Trigger.Request(), Trigger.Response())


def test_set_home_records_the_current_joint_positions(make_node):
    """The taught pose is whatever the arm is standing at when the service is called."""
    node = make_node()
    _feed(node)

    response = _set_home(node)

    assert response.success
    assert node._home_joints == pytest.approx(TAUGHT)


def test_set_home_reads_joints_by_name_not_by_index(make_node):
    """
    /joint_states also carries the gripper's joints, in no guaranteed order.

    Taking the first seven positions would record finger widths as arm angles, and the resulting
    home would be a plan to somewhere the operator never put the arm.
    """
    node = make_node()
    node._on_joint_state(
        _joint_state(
            ['fr3_finger_joint1', 'fr3_finger_joint2'] + list(reversed(mb.HOME_JOINT_NAMES)),
            [0.04, 0.04] + list(reversed(TAUGHT)),
        )
    )

    assert _set_home(node).success
    assert node._home_joints == pytest.approx(TAUGHT)


def test_set_home_does_not_move_the_arm(make_node):
    """Teaching is a read. Planning or executing here would move the arm on a record request."""
    node = make_node()
    _stub_plan_ok(node)
    executed = _capture_execute(node)
    _feed(node)

    assert _set_home(node).success

    assert executed == []
    node._joint_plan.call_async.assert_not_called()


def test_home_plans_to_the_taught_pose(make_node):
    """After teaching, /polyumi/home must drive to the taught pose rather than the SRDF one."""
    node = make_node()
    _feed(node)
    _set_home(node)
    _stub_plan_ok(node)
    _capture_execute(node)

    assert _home(node).success

    constraints = node._joint_plan.call_async.call_args[0][0].motion_plan_request.goal_constraints[0].joint_constraints
    assert [c.joint_name for c in constraints] == mb.HOME_JOINT_NAMES
    assert [c.position for c in constraints] == pytest.approx(TAUGHT)


def test_taught_pose_survives_a_restart(make_node, tmp_path):
    """
    A pose taught once has to outlive the next bringup, or evals re-teach it every session.

    The second node is a fresh construction against the same file — the restart, in miniature.
    """
    pose_file = str(tmp_path / 'persist.yaml')
    _feed(taught := make_node(home_pose_file=pose_file))
    assert _set_home(taught).success

    restarted = make_node(home_pose_file=pose_file)

    assert restarted._home_joints == pytest.approx(TAUGHT)


def test_taught_pose_outranks_the_home_joints_parameter(make_node, tmp_path):
    """The file is the more recent, more specific statement of where this task starts."""
    pose_file = str(tmp_path / 'persist.yaml')
    _feed(taught := make_node(home_pose_file=pose_file))
    _set_home(taught)

    restarted = make_node(home_pose_file=pose_file, home_joints=[9.0] * 7)

    assert restarted._home_joints == pytest.approx(TAUGHT)


def test_no_taught_pose_falls_back_to_the_srdf_ready_pose(make_node):
    """With nothing taught, the default behaviour must be exactly what it was before."""
    node = make_node()

    assert node._home_joints == pytest.approx(mb.HOME_JOINTS)


def test_set_home_refuses_a_stale_joint_state(make_node):
    """
    A stale cache describes where the arm *used to be*, and recording it is silent.

    The broadcaster dying is the realistic case: joint_states simply stops, the last message
    stays cached, and nothing on the wire distinguishes it from a stationary arm.
    """
    node = make_node()
    _feed(node)
    node._joint_state_at = node.get_clock().now() - Duration(seconds=mb.JOINT_STATE_MAX_AGE_S + 5.0)

    response = _set_home(node)

    assert not response.success
    assert 'stale' in response.message
    assert node._home_joints == pytest.approx(mb.HOME_JOINTS), 'a refused teach must not change the pose'


def test_set_home_refuses_when_nothing_has_published_joint_states(make_node):
    """Without franka_bringup up there is nothing to record, and the message should say so."""
    node = make_node()

    response = _set_home(node)

    assert not response.success
    assert mb.JOINT_STATE_TOPIC in response.message


def test_set_home_refuses_a_joint_state_missing_an_arm_joint(make_node):
    """Six of seven joints would otherwise be recorded with the seventh left at its old value."""
    node = make_node()
    _feed(node, positions=TAUGHT[:-1], names=mb.HOME_JOINT_NAMES[:-1])

    response = _set_home(node)

    assert not response.success
    assert 'fr3_joint7' in response.message


def test_set_home_refused_while_a_home_is_in_flight(make_node):
    """Mid-home the arm is moving, so the cached pose is one it is passing through."""
    node = make_node()
    _feed(node)
    node._busy.acquire()
    try:
        response = _set_home(node)
    finally:
        node._busy.release()

    assert not response.success
    assert 'busy' in response.message
    assert node._home_joints == pytest.approx(mb.HOME_JOINTS)


def test_set_home_releases_the_busy_lock_after_a_refusal(make_node):
    """A failed teach must not wedge the bridge — /polyumi/home still has to work afterwards."""
    node = make_node()

    assert not _set_home(node).success

    assert node._busy.acquire(blocking=False), 'busy lock left held after a failed set_home'
    node._busy.release()


def test_set_home_still_succeeds_in_memory_when_the_file_cannot_be_written(make_node, tmp_path):
    """
    An unwritable file must not lose the pose the operator just taught for this session.

    It does have to be said out loud, though: the difference only shows up after the next
    bringup, by which time the trial has already started somewhere else.
    """
    unwritable = tmp_path / 'nope'
    unwritable.write_text('not a directory')
    node = make_node(home_pose_file=str(unwritable / 'pose.yaml'))
    _feed(node)

    response = _set_home(node)

    assert response.success
    assert 'in memory only' in response.message
    assert node._home_joints == pytest.approx(TAUGHT)


def test_persistence_can_be_turned_off(make_node):
    """An empty home_pose_file keeps the taught pose in memory and writes nothing."""
    node = make_node(home_pose_file='')
    _feed(node)

    response = _set_home(node)

    assert response.success
    assert node._home_joints == pytest.approx(TAUGHT)
    assert node._home_pose_file is None


def test_a_pose_file_naming_other_joints_is_ignored(make_node, tmp_path):
    """
    Replaying a file written for a different joint set would home somewhere else entirely.

    Positions are matched to names by order, so a file listing seven *other* joints is not a
    partial answer — it is seven wrong angles applied to fr3_joint1..7.
    """
    pose_file = tmp_path / 'foreign.yaml'
    pose_file.write_text(
        yaml.safe_dump({'joint_names': [f'panda_joint{i}' for i in range(1, 8)], 'joint_positions': TAUGHT})
    )

    node = make_node(home_pose_file=str(pose_file))

    assert node._home_joints == pytest.approx(mb.HOME_JOINTS)


def test_a_truncated_pose_file_is_ignored(make_node, tmp_path):
    """A half-written file must not take the bridge down at startup."""
    pose_file = tmp_path / 'truncated.yaml'
    pose_file.write_text('joint_names: [fr3_joint1, fr3_jo')

    node = make_node(home_pose_file=str(pose_file))

    assert node._home_joints == pytest.approx(mb.HOME_JOINTS)


def test_a_wrong_length_pose_file_is_ignored(make_node, tmp_path):
    """Six values would zip() short against seven names and home to a partial pose."""
    pose_file = tmp_path / 'short.yaml'
    pose_file.write_text(yaml.safe_dump({'joint_names': mb.HOME_JOINT_NAMES, 'joint_positions': TAUGHT[:6]}))

    node = make_node(home_pose_file=str(pose_file))

    assert node._home_joints == pytest.approx(mb.HOME_JOINTS)


def test_the_written_file_records_the_joint_names_alongside_the_values(make_node, tmp_path):
    """The names are what makes the file safe to reload; a bare list of angles is not."""
    pose_file = tmp_path / 'written.yaml'
    node = make_node(home_pose_file=str(pose_file))
    _feed(node)

    assert _set_home(node).success

    doc = yaml.safe_load(pose_file.read_text())
    assert doc['joint_names'] == mb.HOME_JOINT_NAMES
    assert doc['joint_positions'] == pytest.approx(TAUGHT)
    assert 'recorded_at' in doc


def test_teaching_twice_overwrites_rather_than_appends(make_node, tmp_path):
    """Re-teaching is the normal way to nudge a start pose between eval blocks."""
    pose_file = tmp_path / 'twice.yaml'
    node = make_node(home_pose_file=str(pose_file))
    _feed(node)
    _set_home(node)
    second = [v + 0.05 for v in TAUGHT]
    _feed(node, positions=second)

    assert _set_home(node).success

    assert yaml.safe_load(pose_file.read_text())['joint_positions'] == pytest.approx(second)
    assert make_node(home_pose_file=str(pose_file))._home_joints == pytest.approx(second)
