"""
The inference ``mic_0`` contract must reproduce what the exporter bakes into training.

``polyumi_ros2.audio_preproc`` cuts ``mic_0`` rows from a live piezo buffer;
``polyumi_ingest.export.dp.audio.PiezoMicModality`` cuts them from an episode's precomputed
per-frame blocks. A policy only compares like with like, so these have to agree — and unlike the
camera contracts, they cannot agree *by construction*, because the two sides are anchored
differently:

* the exporter anchors each block by ``searchsorted`` on a **59.94 fps GoPro frame's** timestamp,
  then concatenates ``stride`` of them;
* the inference node has no 59.94 fps frame stamps — its camera is the Elgato at ~30 Hz, one
  frame per *step* — so it takes the row as one contiguous span ending at the observation instant.

Those coincide exactly when consecutive blocks abut, and differ only at the seam when the
exporter's deliberate block overlap (width 268 for a ~266.93-sample frame interval) makes them
overlap by a sample. Both cases are pinned below, rather than asserting a single approximate
equality that would hide a real regression in the overlap.

The ROS module is loaded **by path**: it lives in the ROS package (Python 3.12,
``/usr/bin/python3``) which this suite's interpreter cannot import from, but the module itself is
numpy-only by design. Same reasoning as ``camera_preproc_golden.py``.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from polyumi_ingest.export.dp.audio import PiezoMicModality

_ROS_MODULE = (
    pathlib.Path(__file__).resolve().parents[2]
    / 'ros2_ws'
    / 'src'
    / 'polyumi_ros2'
    / 'polyumi_ros2'
    / 'audio_preproc.py'
)


def _load_ros_audio_preproc():
    """Import the ROS-side contract by path — it is numpy-only so this interpreter can run it."""
    if not _ROS_MODULE.is_file():
        raise FileNotFoundError(
            f'inference audio contract not found at {_ROS_MODULE}. It is the counterpart this '
            f'test exists to compare against; if it moved, fix this path rather than skipping.'
        )
    spec = importlib.util.spec_from_file_location('_ros_audio_preproc', _ROS_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audio_preproc = _load_ros_audio_preproc()


def _exporter_rows(piezo: np.ndarray, gopro_ts: np.ndarray, piezo_ts: np.ndarray, gidx: np.ndarray, stride: int):
    """
    Run the real exporter path: step 6's block anchoring, then ``PiezoMicModality``'s row cut.

    Block building is step 6's ``searchsorted`` + fixed-width gather, inlined here because that
    step needs a zarr episode; the row assembly is the shipped code, called directly.
    """
    modality = PiezoMicModality()
    width = modality.block_width
    starts = np.searchsorted(piezo_ts, gopro_ts, side='left').astype(np.int64)
    blocks = np.zeros((len(starts), width), dtype=np.float32)
    for i, s in enumerate(starts):
        chunk = piezo[s : s + width]
        blocks[i, : len(chunk)] = chunk
    modality._blocks = blocks
    modality._stride = stride
    return modality.segment_arrays(gidx)['mic_0'], starts


def _synthetic_piezo(n: int, seed: int = 0) -> np.ndarray:
    """Build a deterministic, non-repeating signal — a constant tone would hide off-by-one errors."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n).astype(np.float32)


def test_row_width_matches_the_exporter_geometry():
    """The inference row width is the exporter's ``stride * block_width``, not an assumption."""
    modality = PiezoMicModality()
    assert audio_preproc.MIC0_SAMPLES_PER_ROW == 2 * modality.block_width
    assert audio_preproc.MIC0_SAMPLE_RATE_HZ == modality.sample_rate_hz
    assert modality.alignment == 'causal', 'this contract assumes causal alignment'


def test_exact_when_blocks_abut():
    """
    With blocks exactly abutting, the two anchorings must agree bit for bit.

    Frame interval == block width is the degenerate case that isolates the row *assembly* from the
    seam behaviour tested below: any disagreement here is a real bug, not a sampling artefact.
    """
    modality = PiezoMicModality()
    width = modality.block_width
    stride = 2
    n_frames = 40
    piezo = _synthetic_piezo(width * (n_frames + 4))
    piezo_ts = np.arange(len(piezo), dtype=np.float64) / audio_preproc.MIC0_SAMPLE_RATE_HZ
    # One frame every `width` samples => starts[i+1] - starts[i] == width exactly.
    gopro_ts = np.arange(n_frames, dtype=np.float64) * width / audio_preproc.MIC0_SAMPLE_RATE_HZ

    gidx = np.arange(6, n_frames, stride)
    expected, starts = _exporter_rows(piezo, gopro_ts, piezo_ts, gidx, stride)

    for row_i, g in enumerate(gidx):
        end_index = int(starts[g - 1]) + width  # one past the last sample of block g-1
        got = audio_preproc.mic0_rows(piezo, end_index, n_rows=1)[0]
        np.testing.assert_array_equal(got, expected[row_i])


def test_matches_exporter_within_the_block_seam_at_real_frame_rate():
    """
    At the real 59.94 fps the exporter's blocks overlap ~1 sample; the rows agree elsewhere.

    ``samples_per_gopro_frame`` is ``ceil(16000/59.94) + 1``, deliberately a little wider than the
    frame interval so consecutive blocks never leave a hole. The cost is that the second block
    repeats the last sample or two of the first, which a contiguous span does not. Bounding that
    to the seam is the real assertion: a drift in anchoring would move many samples, not one.
    """
    modality = PiezoMicModality()
    width = modality.block_width
    stride = 2
    fps = 59.94
    n_frames = 60
    piezo = _synthetic_piezo(int(n_frames * audio_preproc.MIC0_SAMPLE_RATE_HZ / fps) + 4 * width)
    piezo_ts = np.arange(len(piezo), dtype=np.float64) / audio_preproc.MIC0_SAMPLE_RATE_HZ
    gopro_ts = np.arange(n_frames, dtype=np.float64) / fps

    gidx = np.arange(6, n_frames, stride)
    expected, starts = _exporter_rows(piezo, gopro_ts, piezo_ts, gidx, stride)

    for row_i, g in enumerate(gidx):
        first_start = int(starts[g - stride])
        second_start = int(starts[g - 1])
        got = audio_preproc.mic0_rows(piezo, first_start + 2 * width, n_rows=1)[0]

        # The older half is the exporter's first block, sample for sample: both start at the
        # same anchor, so any disagreement here is an assembly bug rather than the seam.
        np.testing.assert_array_equal(got[:width], expected[row_i][:width])

        # The newer half is the exporter's second block, displaced by exactly the difference
        # between the block width and the true frame interval — the overlap the fixed width
        # buys to guarantee blocks never leave a hole. Assert the displacement is that and
        # nothing larger; a drift in anchoring would move many samples, not one or two.
        shift = (first_start + width) - second_start
        assert abs(shift) <= 2, f'row {row_i}: block seam displaced {shift} samples'
        np.testing.assert_array_equal(
            got[width : 2 * width - shift] if shift >= 0 else got[width:],
            expected[row_i][width + shift : 2 * width] if shift >= 0 else expected[row_i][width:],
        )


def test_rows_are_contiguous_and_ordered_oldest_first():
    """The window is one gapless span ending at the instant — what the mel frontend flattens."""
    piezo = _synthetic_piezo(20_000)
    n_rows = 10
    end = 15_000
    rows = audio_preproc.mic0_rows(piezo, end, n_rows=n_rows)
    span = n_rows * audio_preproc.MIC0_SAMPLES_PER_ROW
    np.testing.assert_array_equal(rows.reshape(-1), piezo[end - span : end])


def test_never_reads_future_audio():
    """Causality: nothing after the observation instant may appear in a row."""
    piezo = _synthetic_piezo(20_000)
    end = 12_000
    marked = piezo.copy()
    marked[end:] = 999.0  # anything from the future is unmistakable
    rows = audio_preproc.mic0_rows(marked, end, n_rows=10)
    assert not (rows == 999.0).any()


def test_short_history_is_zero_padded_at_the_front():
    """
    Episode start has no past, and the exporter fills it with silence rather than repeating.

    Repeating the nearest audio would splice a copy of a real contact event into the waveform and
    read as a genuine one.
    """
    piezo = _synthetic_piezo(1000)
    n_rows = 4
    rows = audio_preproc.mic0_rows(piezo, 1000, n_rows=n_rows)
    flat = rows.reshape(-1)
    pad = n_rows * audio_preproc.MIC0_SAMPLES_PER_ROW - 1000
    assert pad > 0, 'this test needs a window longer than the buffer'
    np.testing.assert_array_equal(flat[:pad], np.zeros(pad, dtype=np.float32))
    np.testing.assert_array_equal(flat[pad:], piezo)


def test_decode_piezo_selects_channel_zero_and_scales_like_the_wav_reader():
    """Channel 1 is the air mic; the scale is the pzarr builder's ``int16 / 32768``."""
    piezo_counts = np.array([0, 1, -1, 32767, -32768, 1234], dtype='<i2')
    air_counts = np.full(piezo_counts.shape, 999, dtype='<i2')
    interleaved = np.empty(piezo_counts.size * 2, dtype='<i2')
    interleaved[0::2] = piezo_counts
    interleaved[1::2] = air_counts

    got = audio_preproc.decode_piezo(interleaved.tobytes(), number_of_channels=2)

    np.testing.assert_allclose(got, piezo_counts.astype(np.float32) / 32768.0)
    assert got.dtype == np.float32
    assert got.max() < 1.0 and got.min() >= -1.0


def test_decode_piezo_rejects_a_stream_that_cannot_hold_the_piezo():
    """A mono stream is not this contract, and mis-striding it would sound plausible."""
    with pytest.raises(ValueError, match='at least'):
        audio_preproc.decode_piezo(np.zeros(8, dtype='<i2').tobytes(), number_of_channels=1)


def test_decode_piezo_rejects_a_partial_frame():
    """A truncated frame means the stream is mis-strided, which sounds like plausible audio."""
    with pytest.raises(ValueError, match='whole number'):
        audio_preproc.decode_piezo(np.zeros(5, dtype='<i2').tobytes(), number_of_channels=2)
