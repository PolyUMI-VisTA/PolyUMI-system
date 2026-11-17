"""The finger-camera modality: the pzarr's ``finger/frames``, cropped, as ``data/finger_rgb``."""

from __future__ import annotations

import logging

import numpy as np
import zarr

from polyumi_ingest.camera_preproc import crop_finger_rgb
from polyumi_ingest.config import load_finger_camera_config
from polyumi_ingest.export.dp.modality import ExportModality
from polyumi_ingest.pzarr.store import arr
from polyumi_ingest.timebase import gopro_ts_in_finger_clock

log = logging.getLogger('export.dp')

#: Key for the finger camera in the exported ReplayBuffer. UMI derives its RGB keys purely from
#: ``shape_meta``'s ``type: rgb`` — no camera name is hardcoded anywhere in its dataset or
#: encoder — so this names what the stream *is* rather than where it sits in a numbering.
FINGER_KEY = 'finger_rgb'

_FRAMES_PATH = 'finger/frames'
_TS_PATH = 'timestamps/finger'


class FingerCameraModality(ExportModality):
    """
    The gripper's finger camera on the exported step grid, as ``finger_rgb``.

    Cropped, not resized, by default: the mount occludes a fixed strip carrying no signal, and the
    encoder's input size belongs with whoever builds the policy. Crop and optional resize come from
    ``config/finger_camera.yaml`` and are applied here, so retuning them is a re-export rather than
    a re-preprocess of the corpus.

    The camera records at ~10 fps against a ~30 Hz step grid, so each step takes the newest frame
    at or before it and roughly three rows share one source frame. Nothing is interpolated, but a
    training config wants ``down_sample_steps`` set accordingly or a window fetches the same image
    twice. The measured source rate is in the buffer's meta attrs.

    It also stops recording ~0.65 s before the GoPro does (measured over 111 episodes). Steps past
    the end are excluded by :meth:`valid_steps`: the causal index clamps, so exporting them would
    pair a frozen frame with moving proprioception.

    Selection is causal so training and inference pair the same way; see :meth:`prepare_episode`.
    Looking back only, staleness runs up to a full frame period rather than half, which
    ``max_staleness_s`` has to cover.
    """

    name = FINGER_KEY
    #: Nothing beyond what the visuomotor export already needs. The frames are written at ingest
    #: and the only annotation read is step 1's chirp offset, which the default ``--type dp``
    #: requires anyway for its start trim.
    required_steps = frozenset()

    def __init__(self, output_size: tuple[int, int] | None = None) -> None:
        """
        Read the crop geometry once, so every episode in a buffer is cut the same way.

        ``output_size`` overrides ``config/finger_camera.yaml`` for this export only; ``None``
        means use the configured value. The override exists because the resolution is a property
        of the DATASET, not of the rig: one corpus may want native frames and another a 224x224
        buffer that a policy's shared image_shape can accept, from the same recordings. The size
        used is written into the buffer's meta attrs either way, so a dataset stays self-describing.
        """
        cfg = load_finger_camera_config()
        crop = cfg['crop']
        self.crop = {
            'x_min': int(crop['x_min']),
            'x_max': None if crop['x_max'] is None else int(crop['x_max']),
            'y_min': int(crop['y_min']),
            'y_max': None if crop['y_max'] is None else int(crop['y_max']),
        }
        cfg_size = cfg['output_size'] if output_size is None else output_size
        self.output_size = None if cfg_size is None else (int(cfg_size[0]), int(cfg_size[1]))
        self.max_staleness_s = float(cfg['max_staleness_s'])

        self._frames: zarr.Array | None = None
        self._idx: np.ndarray | None = None
        self._staleness: np.ndarray | None = None
        self._episode_key = ''
        self._shape: tuple[int, int, int] | None = None
        self._source_rate_hz: float | None = None

    def prepare_episode(self, ep: zarr.Group, episode_key: str, gopro_ts: np.ndarray, stride: int) -> None:
        """Map every GoPro frame onto the finger frame nearest it, and note how near that was."""
        for path in (_FRAMES_PATH, _TS_PATH):
            if path not in ep:
                raise RuntimeError(
                    f'{episode_key}: no {path} — this session has no finger camera, so it cannot '
                    f'be exported with `pingest export --type polyumi`. Use the default '
                    f'`--type dp` for a visuomotor dataset, or mark the episode unusable in the '
                    f'catalog.'
                )
        frames = arr(ep, _FRAMES_PATH)
        finger_ts = np.asarray(arr(ep, _TS_PATH)[:], dtype=np.float64)
        if frames.shape[0] != len(finger_ts):
            raise RuntimeError(
                f'{episode_key}: {_FRAMES_PATH} has {frames.shape[0]} frame(s) but {_TS_PATH} has '
                f'{len(finger_ts)}. The pzarr is inconsistent; rebuild it with `pingest process`.'
            )

        # Sample-exact pairing of an image with a pose, so an unshifted grid is not a degraded
        # result but a wrong one: the two devices stamp against unrelated epochs, and without step
        # 1's chirp offset the finger frame chosen for a step is arbitrary.
        gopro_ts_finger = gopro_ts_in_finger_clock(ep, require_offset=True)
        # CAUSAL, not nearest: the newest finger frame at or before each step. Inference can only
        # look backwards -- the frame after the observation instant has not been captured yet --
        # so choosing the nearest frame here would train the policy on a pairing it can never be
        # given. At 10 fps the nearest frame sits up to half a period (50 ms) in the future, which
        # would be a systematic bias toward fresher tactile data at training time than at run time,
        # not a symmetric error. This matches contact_audio's `block_alignment: causal` and the
        # backwards-only lookup in policy_client_node._frame_at.
        #
        # -1 where no finger frame precedes the step at all (the stream starts after the GoPro).
        # Those steps get infinite staleness so valid_steps drops them, exactly as it drops the
        # steps past the end of the stream.
        self._idx = np.searchsorted(finger_ts, gopro_ts_finger, side='right') - 1
        self._staleness = np.where(
            self._idx >= 0,
            gopro_ts_finger - finger_ts[np.maximum(self._idx, 0)],
            np.inf,
        )
        self._idx = np.maximum(self._idx, 0)
        self._frames = frames
        self._episode_key = episode_key
        if len(finger_ts) >= 2:
            self._source_rate_hz = 1.0 / float(np.median(np.diff(finger_ts)))

        shape = crop_finger_rgb(np.asarray(frames[0]), output_size=self.output_size, **self.crop).shape
        if self._shape is not None and shape != self._shape:
            raise RuntimeError(
                f'{episode_key}: the crop yields {shape} here but {self._shape} in an earlier '
                f'episode of this export — the sessions were recorded at different finger-camera '
                f'resolutions, and one buffer cannot hold both.'
            )
        self._shape = shape

    def valid_steps(self, steps: np.ndarray) -> np.ndarray:
        """Mark the steps whose preceding finger frame is close enough in time to be that step."""
        assert self._staleness is not None, 'prepare_episode must run before valid_steps'
        return self._staleness[steps] <= self.max_staleness_s

    def segment_arrays(self, gidx: np.ndarray) -> dict[str, np.ndarray]:
        """Crop the finger frame belonging to each exported step."""
        assert self._frames is not None and self._idx is not None and self._staleness is not None
        # Backstop, not the enforcement path: valid_steps already removed anything this stale
        # before segmentation. It fires only if that mask were bypassed, and a frozen frame is
        # worth an exception rather than a silently plausible observation.
        stale = self._staleness[gidx]
        worst = int(np.argmax(stale))
        if stale[worst] > self.max_staleness_s:
            raise RuntimeError(
                f'{self._episode_key}: step at GoPro frame {int(gidx[worst])} is '
                f'{stale[worst]:.3f}s after its most recent finger frame, over the '
                f'{self.max_staleness_s:.3f}s limit in config/finger_camera.yaml, yet reached '
                f'segmentation — the validity mask was bypassed.'
            )

        # Each finger frame lands in ~3 consecutive steps at 10 fps against a ~30 Hz grid, so
        # decode the distinct ones once and expand. Decoding per step would triple the JPEG-XL
        # work for byte-identical output.
        rows = self._idx[gidx]
        uniq, inverse = np.unique(rows, return_inverse=True)
        decoded = np.stack(
            [crop_finger_rgb(np.asarray(self._frames[int(i)]), output_size=self.output_size, **self.crop) for i in uniq]
        )
        return {FINGER_KEY: decoded[inverse].astype(np.uint8)}

    def segment_provenance(self, gidx: np.ndarray) -> dict:
        """Record how well the finger stream actually covered this segment."""
        assert self._idx is not None and self._staleness is not None
        stale = self._staleness[gidx]
        return {
            'max_staleness_s': float(stale.max()),
            'median_staleness_s': float(np.median(stale)),
            'n_source_frames': int(len(np.unique(self._idx[gidx]))),
        }

    def meta_attrs(self) -> dict:
        """Make the buffer self-describing about the crop it was built under."""
        return {
            'finger_rgb_crop': dict(self.crop),
            'finger_rgb_output_size': list(self.output_size) if self.output_size else None,
            'finger_rgb_shape': list(self._shape) if self._shape else None,
            'finger_rgb_source_rate_hz': self._source_rate_hz,
            'finger_rgb_max_staleness_s': self.max_staleness_s,
            'finger_rgb_source': _FRAMES_PATH,
        }
