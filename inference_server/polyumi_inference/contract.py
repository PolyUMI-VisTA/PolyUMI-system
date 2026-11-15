"""
What the PolyUMI policy requires of an observation.

:mod:`polyumi_inference.wire` decides whether a frame is *readable*. This module decides whether it
is *servable* -- the channels, dimensions and shapes a checkpoint was trained on.

It exists as one function because two servers must agree on it exactly. ``dummy_server`` is the
bringup path: every frame it accepts is one a checkpoint will later be handed, so a frame it accepts
and the real server rejects is a bug discovered on hardware. Both call
:func:`validate_observation`, so that agreement is a shared call rather than two matching listings.
"""

from __future__ import annotations

from typing import Iterable

from polyumi_inference.errors import WireFormatError
from polyumi_inference.types import AGENT_POS_DIM, Observation

#: The wrist camera, named for the dataset field it comes from.
CAMERA_CHANNEL = 'camera0_rgb'
#: Absolute end-effector pose ``[x, y, z, qx, qy, qz, qw, gripper]``. A wire concept: the server
#: splits it into three separate ``shape_meta`` fields before the policy sees it.
AGENT_POS_CHANNEL = 'agent_pos'

#: The finger camera, named for the dataset field it comes from.
FINGER_CHANNEL = 'finger_rgb'
#: The contact mic. Alone among the channels this is NOT one entry per observation step: its
#: leading dim is the policy's ``audio_obs_horizon``, a training-time constant that need not equal
#: ``n_obs_steps`` (10 against 2 for the Vista family; 17 before the 2026-09-07 realignment),
#: because the mic updates far faster than
#: the control loop and the policy consumes a window of rows rather than one row per step.
MIC0_CHANNEL = 'mic_0'
#: Samples in one ``mic_0`` row: 33.5 ms at 16 kHz. Fixed by the dataset's own blocking, so it is
#: the same for every checkpoint, unlike the row count.
MIC0_SAMPLES_PER_ROW = 536

#: Channels the policy cannot run without. Named for the dataset's own fields, so wiring a new
#: modality is adding a name here and in shape_meta rather than reshaping the request.
REQUIRED_CHANNELS = (CAMERA_CHANNEL, AGENT_POS_CHANNEL)


def validate_observation(obs: Observation, required: Iterable[str] = REQUIRED_CHANNELS) -> None:
    """
    Check an observation against the policy's contract, or raise.

    :raises WireFormatError: with a message naming the specific disagreement. The caller turns
        these into 422s: an observation the policy cannot consume is a bad request.
    """
    obs.require(required)

    # The header's window length and each array's leading dim are two independent claims about the
    # same thing; a mismatch means the client packed something other than what it says it packed.
    # ndim first: a 0-d channel (shape ()) has no [0] to read, and shape[0] on it raises IndexError
    # rather than the WireFormatError this is supposed to turn a bad frame into.
    for name in required:
        # mic_0 is exempt: it is a window of audio rows, not one entry per step, so its leading
        # dim is audio_obs_horizon. Checked on its own terms below.
        if name == MIC0_CHANNEL:
            continue
        arr = obs[name]
        if arr.ndim == 0 or arr.shape[0] != obs.n_obs_steps:
            raise WireFormatError(f'{name} leading dim must be n_obs_steps={obs.n_obs_steps}, got {list(arr.shape)}')

    if CAMERA_CHANNEL in obs:
        image = obs[CAMERA_CHANNEL]
        if image.ndim != 4 or image.shape[-1] != 3:
            raise WireFormatError(f'{CAMERA_CHANNEL} must be [To,H,W,3], got {list(image.shape)}')

    if AGENT_POS_CHANNEL in obs:
        agent_pos = obs[AGENT_POS_CHANNEL]
        if agent_pos.ndim != 2 or agent_pos.shape[1] != AGENT_POS_DIM:
            raise WireFormatError(f'{AGENT_POS_CHANNEL} must be [To,{AGENT_POS_DIM}], got {list(agent_pos.shape)}')

    if FINGER_CHANNEL in obs:
        finger = obs[FINGER_CHANNEL]
        if finger.ndim != 4 or finger.shape[-1] != 3:
            raise WireFormatError(f'{FINGER_CHANNEL} must be [To,H,W,3], got {list(finger.shape)}')

    if MIC0_CHANNEL in obs:
        mic0 = obs[MIC0_CHANNEL]
        if mic0.ndim != 2 or mic0.shape[1] != MIC0_SAMPLES_PER_ROW:
            raise WireFormatError(
                f'{MIC0_CHANNEL} must be [audio_obs_horizon,{MIC0_SAMPLES_PER_ROW}], got {list(mic0.shape)}'
            )
