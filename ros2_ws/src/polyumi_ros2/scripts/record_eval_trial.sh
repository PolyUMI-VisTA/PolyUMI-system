#!/usr/bin/env bash
# Record one evaluation trial as a rosbag, on lamb, into a per-policy folder.
#
#   ./record_eval_trial.sh <policy> <trial_number> [duration_s]
#   ./record_eval_trial.sh dp 1
#   ./record_eval_trial.sh vista 7 45
#
# Ctrl-C ends a recording early; without a duration it records until you do.
#
# WHERE IT WRITES. Bags land on lamb under $EVAL_ROOT (default /data/eval_results_9_6), because
# ros2 bag has to run where the topics are and lamb has 11 TB free. The Seagate is on the laptop,
# so copying there is a separate step -- see the rsync line this prints when it finishes. Writing
# a multi-hundred-MB bag straight to a network mount would drop messages, not just run slow.
#
# WHAT IT RECORDS, and why not more. The whole point is that a trial can be re-judged later
# without the robot, so this keeps every observation the policy saw, every command it issued, and
# the diagnostics that explain a bad rollout -- and deliberately drops the one topic that would
# dwarf all of them.
set -euo pipefail

POLICY="${1:?usage: record_eval_trial.sh <policy: dp|vista|sparsh_x> <trial_number> [duration_s]}"
TRIAL="${2:?missing trial number}"
DURATION="${3:-}"

EVAL_ROOT="${EVAL_ROOT:-/data/eval_results_9_6}"
TASK="${TASK:-lightbulb}"
OUT_DIR="${EVAL_ROOT}/${TASK}/${POLICY}"
BAG="${OUT_DIR}/trial_${TRIAL}"

if [ -e "${BAG}" ]; then
    echo "error: ${BAG} already exists -- refusing to overwrite a recorded trial." >&2
    echo "       Pick another trial number, or move the old one aside." >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"

# The topics, grouped by why each is here.
TOPICS=(
    # --- What the policy saw. Without these a trial cannot be re-scored or replayed. ---
    # COMPRESSED, never /gopro/image_raw: raw is 1920x1080 RGB at 60 Hz, ~370 MB/s, which is
    # larger than everything else here combined by two orders of magnitude and is losslessly
    # recoverable from the JPEG for every purpose an eval has.
    /gopro/image_raw/compressed
    /gopro/camera_info
    # The tactile pair. Harmless to list for a dp run -- ros2 bag records what exists and does not
    # fail on a topic that never appears -- so one topic list covers all three policies.
    /pi/camera/image/compressed
    /pi/audio/raw

    # --- What the policy did. ---
    /polyumi/target_poses_traj      # the executed chunk
    /polyumi/target_gripper
    /polyumi/target_poses_preview   # published even when execute_motion:=false, so dry runs record
    /polyumi/target_gripper_preview

    # --- Where the arm actually went, which is NOT the same as what was commanded. ---
    /joint_states
    /fr3_gripper/joint_states
    /faulhaber_gripper/motor_current_ma          # contact/force proxy on the gripper
    /faulhaber_gripper/waypoint_tracking_error_mm

    # --- Why a rollout went wrong. Cheap, and the first thing you want on a failed trial. ---
    /polyumi/diag/obs_age_s
    /polyumi/diag/inference_latency_s
    /polyumi/diag/inference_model_s
    /polyumi/diag/inference_overhead_s
    /polyumi/diag/image_age_s
    /polyumi/diag/gripper_state_age_s
    /polyumi/diag/gripper_width_m
    /polyumi/diag/n_published_arm
    /polyumi/diag/n_published_gripper
    /polyumi/diag/n_stale_arm
    /polyumi/diag/n_stale_gripper

    # --- Frames. Without TF the poses above cannot be placed in the world. ---
    /tf
    /tf_static

    # --- Node-level errors, which is where a stale-channel warning shows up. ---
    /rosout
)

echo "==> ${POLICY} trial ${TRIAL} -> ${BAG}"
CMD=(ros2 bag record --storage mcap --output "${BAG}" "${TOPICS[@]}")
# `timeout`, not ros2 bag's -d: that flag SPLITS the bag at a duration, it does not stop the
# recording, so using it for a fixed-length trial would record forever in numbered parts.
[ -n "${DURATION}" ] && CMD=(timeout --signal=INT "${DURATION}" "${CMD[@]}")

# Ctrl-C (and `timeout --signal=INT`) is the normal way to stop a recording and exits non-zero, so
# the exit status cannot distinguish "you stopped it" from "it never started". Check the artefact
# instead: a trial that silently recorded nothing is the one failure that must not look like
# success, because it is only discovered when the eval is over and the robot has been put away.
"${CMD[@]}" || true

# Checked against the CAMERA, not against the bag merely being non-empty: the recorder publishes
# its own /rosout, so every bag has messages in it no matter how wrong the run was. A trial
# without the wrist camera cannot be re-scored or replayed and is not a trial, whereas one missing
# a diagnostic is merely thinner. This is the cheapest check that actually distinguishes them.
CAMERA_TOPIC=/gopro/image_raw/compressed
if [ ! -d "${BAG}" ] || ! ros2 bag info "${BAG}" 2>/dev/null | grep -q "Topic: ${CAMERA_TOPIC} .*Count: [1-9]"; then
    echo "ERROR: ${BAG} captured no ${CAMERA_TOPIC} messages -- not a usable trial." >&2
    echo "       Is the inference stack up? Check with: ros2 topic hz ${CAMERA_TOPIC}" >&2
    rm -rf "${BAG}"   # leave no directory that a later trial number would refuse to overwrite
    exit 1
fi

echo
echo "==> recorded: $(du -sh "${BAG}" | cut -f1) at ${BAG}"
ros2 bag info "${BAG}" 2>/dev/null | grep -E "^(Duration|Messages|Topic information)" | head -3 || true
echo "==> pull to the Seagate from the LAPTOP with:"
echo "    rsync -avP lamb:${EVAL_ROOT}/ \"/media/aravind-linux/Seagate Portable Drive/eval results_9_6/\""
