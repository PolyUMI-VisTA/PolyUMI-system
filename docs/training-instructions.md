# Training a policy

This is the step **after** ingest and preprocessing. The full pipeline is:

```
pingest fetch → process-all → pp (preprocessing steps 1–5) → export → TRAIN (this doc)
```

Training runs a fork of [UMI](https://github.com/real-stanford/universal_manipulation_interface)
inside a **Docker** image, so it needs no host conda (which fights the ROS install) and is portable
to the GPU workstation. The same image serves both training and inference — the serving environment
must match the training one because checkpoints are dill-pickled and unpickle against the exact
dependency tree.

There are **two forks** under `external/`, selected with `POLICY` (below): the visuomotor diffusion
policy (`dp`, the default) and Rickmer's Vista multimodal zoo (`vista`). Most of this doc is
written for `dp` — the flags, mounts and rootless gotchas are identical for both — with the
Vista-specific parts in "Choosing a policy" and "Vista".

The visuomotor policy takes vision plus proprioception, which is what `export --type dp` (the
default) produces. Vista adds the finger camera and the contact mic, from
`export --type polyumi`; the data contract and the ManiWAV recipe behind the audio channel are in
[maniwav-audio-policy.md](maniwav-audio-policy.md).

## Prerequisites

- A **GPU workstation** with an NVIDIA GPU. (Training a diffusion policy with a timm vision
  encoder wants meaningfully more than the laptop's ~4 GB.)
- **Docker** with the NVIDIA container toolkit. Ours runs **rootless** (no sudo) — see the
  gotchas below.
- An **exported dataset**: `pingest export <scene> --output <name>.zarr.zip`. Copy the
  `.zarr.zip` to the workstation.
- A **Weights & Biases** API key (`WANDB_API_KEY`) if you want online logging. `WANDB_ENTITY`
  and `WANDB_PROJECT` are optional, forwarded only if set in the calling shell
  (`${oc.env:WANDB_ENTITY,null}` / `${oc.env:WANDB_PROJECT,polyumi}`) — unset, they fall back to
  your wandb login's default entity and the `polyumi` project, not to a hard-coded team.

## Quick start

From the PolyUMI repo root on the workstation:

```bash
# Build the image and run a short overfit smoke test (loss should drop):
DATASET=/abs/path/to/your.zarr.zip \
WANDB_API_KEY=... \
./train_policy.sh training.num_epochs=3 task.dataset.val_ratio=0 logging.mode=offline
```

That proves the image builds, the GPU is visible and the dataset loads. `train_policy.sh` builds
`external/polyumi_diffusion_policy` and runs it with the rootless-safe flags and the
dataset/output mounts; extra arguments pass straight through to `train.py` as **Hydra overrides**,
and outputs land in `data/dp_outputs/` (`OUTPUT_DIR`). Real runs are below.

## Worked examples

These are the invocations the shipped checkpoints were actually trained with. `WANDB_API_KEY` is
elided throughout; set it in the calling shell.

**Visuomotor (`dp`), the default.** `CONFIG_NAME` picks the workspace config, and so the task;
everything else is identical between runs, which is why only that one variable changes below. It
is an env var rather than a Hydra override because Hydra cannot set `--config-name` that way, and
it replaces the old `DP_CONFIG`, which was never wired to anything.

```bash
# Standard visuomotor, the default config:
DATASET=/abs/path/to/gears_v2_polyumi_clean.zarr.zip ./train_policy.sh

# A tactile+audio task on the same corpus — same command, different CONFIG_NAME:
CONFIG_NAME=train_diffusion_unet_timm_polyumi_tactile_audio_workspace \
DATASET=/abs/path/to/gears_v2_polyumi_clean.zarr.zip ./train_policy.sh
```

**Slip-control ablations.** One workspace config per sensor combination — `_slip_v` (vision only),
`_slip_t`, `_slip_a`, `_slip_vt`, `_slip_va`, `_slip_vta` — all against the same export, so the
only thing that differs between runs is which streams the task declares:

```bash
CONFIG_NAME=train_diffusion_unet_timm_polyumi_slip_vta_workspace \
DATASET=/abs/path/to/slip_polyumi_224.zarr.zip ./train_policy.sh

CONFIG_NAME=train_diffusion_unet_timm_polyumi_slip_t_workspace \
DATASET=/abs/path/to/slip_polyumi_224.zarr.zip ./train_policy.sh
```

**Vista (`POLICY=vista`).** Selected by `--model`, not `CONFIG_NAME` — the fork's suite runner
passes its own `--config-name` per model. Hydra overrides go after a `--` separator:

```bash
POLICY=vista DATASET=/abs/path/to/gears_v2_polyumi_clean.zarr.zip \
    ./train_policy.sh --model mitas

POLICY=vista DATASET=/abs/path/to/gears_v2_polyumi_clean.zarr.zip \
    ./train_policy.sh --model see_hear_feel

POLICY=vista DATASET=/abs/path/to/gears_v2_polyumi_clean.zarr.zip \
    ./train_policy.sh --model sparsh_x

# Resume, and sensor ablations, via the -- separator:
POLICY=vista DATASET=... ./train_policy.sh --model mitas -- training.resume=true
POLICY=vista DATASET=... ./train_policy.sh --model mitas -- ablation=v
```

Serving one of these: the image-side mechanics are under "Inference" below; which flags each
policy family needs on the arm is [crb-fr3-inference.md](crb-fr3-inference.md) §3.

## Choosing a policy

`POLICY` picks which fork gets built and run. It defaults to `dp`, so every command above is
unchanged.

| `POLICY` | Fork | Image | Entrypoint | Dataset |
|---|---|---|---|---|
| `dp` *(default)* | `external/polyumi_diffusion_policy` | `polyumi-dp` | `docker/train.sh` | `export --type dp` |
| `vista` | `external/polyumi_vista_policy` | `polyumi-vista` | `docker/train_day0suite.sh` | `export --type polyumi` |

The wiring lives in **`config/policy.<name>.env`** in this repo — one file per fork, holding its
directory under `external/`, its image tag, its container entrypoint, where `DATASET` mounts, and
its default `CONFIG_NAME`. These are facts about how PolyUMI *drives* a fork, so they belong here
rather than in the submodule: a fork cannot know its own path in the parent, and the values have to
be readable before there is anything checked out to read. Same pattern as `config/env.<hostname>.sh`.

Hyperparameters are **not** here. They stay in each fork's own Hydra tree
(`vista/config/`, `diffusion_policy/config/`); `config/policy.*.env` only names which workspace yaml
to run. Adding a third fork is a submodule plus one new `config/policy.<name>.env`.

Building without starting a run — useful for the first, slow build of a new fork:

```bash
POLICY=vista ./build_policy_image.sh
```

## Building the image

`polyumi-dp` is built in two stages, by `build_policy_image.sh` at the repo root — which
`train_policy.sh` and `serve_policy.sh` both source, so the two roles always run the same image
(a checkpoint is dill-pickled and must unpickle against the exact dep tree it was trained with):

| Stage | Context | What | Cost |
|---|---|---|---|
| `<image>-base` | the selected fork | conda env, torch, the policy code | ~23 min cold, then cached |
| `<image>` | `inference_server` | `polyumi_inference` + `pytest` layered on top | seconds |

Two stages because a fork's Dockerfile builds with the *fork directory* as its context and so
cannot see `inference_server/`, which both ends of the inference protocol import. Layer caching
makes this cheaper than one build would be: editing the shared library rebuilds only stage two.
The stage-one solve is **not** shared between forks — each has its own conda env — so the first
build of a new policy pays the ~23 minutes again.

To build by hand:

```bash
docker build -t polyumi-dp-base external/polyumi_diffusion_policy
docker build -t polyumi-dp --build-arg BASE=polyumi-dp-base \
    -f docker/polyumi_inference.Dockerfile inference_server
```

## The two entrypoints

The image (`polyumi-dp`) has one env and two thin wrappers, both in the fork's `docker/`:

| Command | Wrapper | Purpose |
|---|---|---|
| `bash docker/train.sh` *(default)* | runs `python train.py --config-name=train_diffusion_unet_timm_polyumi_workspace` | training |
| `bash docker/serve.sh` | runs `uvicorn serve_policy:app` on `:8000` | inference server (see below) |

## Rootless Docker gotchas

Rootless is fine for this workload, but five things differ from rootful Docker. `train_policy.sh`
handles the ones it can and exposes the rest as env vars.

1. **GPU flag.** The script defaults to `--gpus all`. If that fails under rootless, use the CDI
   form: `GPU_FLAG="--device nvidia.com/gpu=all" ./train_policy.sh ...`. **Verify GPU visibility
   first** (below) — this is the most common snag.
2. **Output file ownership.** Rootless remaps container UIDs. The script defaults to
   `USER_FLAG="--user 0:0"` (container root), which under rootless Docker maps back to *your*
   host user — so the bind-mounted output dir is writable and checkpoints come back owned by
   you. (Running as the image's default `mambauser` instead maps to a subuid that can't write
   your dir — the failure you'd otherwise hit.) On a **rootful** Docker host, override with
   `USER_FLAG="--user $(id -u):$(id -g)"`. The script also sets `HOME=/tmp` and cache dirs so
   nothing needs a writable home regardless.
3. **Shared memory.** The PyTorch DataLoader needs more than Docker's 64 MB default `/dev/shm`
   or workers crash with a bus error. The script sets `--shm-size=8g` (override with `SHM_SIZE`).
   Do **not** use `--ipc=host` under rootless.
4. **Networking (inference).** Publish with `-p 8002:8000`, matching the wrapper's default. Do **not** use `--network host` —
   under rootless it shares the rootless net namespace, not the real host.
5. **Storage / no registry.** Build on the workstation from the fork checkout; rootless image
   storage lives under your home. Pruning the fork (a later task) shrinks the image.

## Verification

Run these in order; each isolates the next failure.

```bash
# 1. GPU visible (do this first — the likeliest rootless snag):
#    (both stages; see "Building the image" above)
docker build -t polyumi-dp-base external/polyumi_diffusion_policy
docker build -t polyumi-dp --build-arg BASE=polyumi-dp-base \
    -f docker/polyumi_inference.Dockerfile inference_server
docker run --rm --gpus all polyumi-dp nvidia-smi          # or --device nvidia.com/gpu=all
docker run --rm polyumi-dp \
    micromamba run -n umi python -c "import polyumi_inference"   # the shared protocol library
docker run --rm --gpus all polyumi-dp \
    micromamba run -n umi python -c "import torch; print(torch.cuda.is_available())"   # -> True

# 2. Dataset loads under UmiDataset (validates the exporter's schema against the real reader):
docker run --rm -v /abs/path/to/your.zarr.zip:/data/dataset.zarr.zip:ro polyumi-dp \
    micromamba run -n umi python -c "
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)  # as train.py does, for latency_steps
from diffusion_policy.dataset.umi_dataset import UmiDataset
# Nest under 'task' so the \${task.*} interpolations in the config resolve.
cfg = OmegaConf.create({'task': OmegaConf.load('diffusion_policy/config/task/polyumi.yaml')})
ds = UmiDataset(shape_meta=cfg.task.shape_meta, dataset_path='/data/dataset.zarr.zip',
                pose_repr=cfg.task.pose_repr, cache_dir=None)
print('episodes:', ds.replay_buffer.n_episodes, 'obs keys:', sorted(ds[0]['obs'].keys()))
"

# 3. Overfit smoke — loss must descend:
DATASET=/abs/path/to/your.zarr.zip ./train_policy.sh \
    training.num_epochs=5 task.dataset.val_ratio=0 logging.mode=offline
```

Watch `train_action_mse_error` (and `train_loss`) descend in the console or in W&B (project
`polyumi`). The checkpointed weights are the **EMA** model (`workspace.ema_model`), not the raw
`model` — evaluation loads the EMA weights.

## Inference (after you have a checkpoint)

The same image serves the policy over the exact HTTP contract the ROS-side
`policy_client_node` already speaks to `inference_server/polyumi_inference/dummy_server.py`. Use the
`serve_policy.sh` wrapper (the inference counterpart of `train_policy.sh` — it builds the image
and wires the rootless-safe flags, checkpoint mount, and HF cache):

```bash
CKPT=/abs/path/to/epoch=0070-train_loss=0.021.ckpt ./serve_policy.sh
# CKPT=... PORT=8001 ./serve_policy.sh    # override the published port
```

> **Do not run `external/polyumi_diffusion_policy/docker/serve.sh` on the host** — it is the
> *in-container* entrypoint (`exec uvicorn …`) and fails with `exec: uvicorn: not found` because
> the `umi` conda env only exists inside the image. `serve_policy.sh` runs it via `docker run`.

The equivalent raw command, if you'd rather not use the wrapper:

```bash
docker run --rm -it --gpus all --user 0:0 -p 8002:8000 \
    -e HOME=/tmp -e HF_HOME=/hf_cache \
    -e CKPT_PATH=/data/model.ckpt \
    -v /abs/path/to/epoch=0070-train_loss=0.021.ckpt:/data/model.ckpt:ro \
    -v "${HOME}/.cache/huggingface:/hf_cache:rw" \
    polyumi-dp bash docker/serve.sh
```

The `-v ...huggingface:/hf_cache` mount + `HF_HOME=/hf_cache` persist the timm ViT encoder
weights (~600 MB) so the server doesn't re-download them on every start — the same cache
`train_policy.sh` uses. `--gpus all` and `--user 0:0` mirror the training flags (GPU access;
rootless-safe uid). Mounting the checkpoint *file* to a clean path sidesteps the `=` in its name.

Set the client's `inference_server_url` to `http://<workstation>:8002/predict_cartesian/`
(or the port selected with `PORT`). The launch file defaults to the dummy server on port 8000.

`serve_policy.py` serves the EMA weights and converts the relative action chunk back to absolute
EEF poses. It has one endpoint beyond `/predict_cartesian/`: the client must `POST /reset
{agent_pos: [8]}` at the start of each rollout to set the episode-start pose, which
`robot0_eef_rot_axis_angle_wrt_start` is measured against. Absent a reset it falls back to the
current pose and warns. Running it on the arm is [crb-fr3-inference.md](crb-fr3-inference.md).

## Vista

`POLICY=vista` runs the multimodal models using vision (`camera0_rgb`), finger camera
(`finger_rgb`), contact mic (`mic_0`) and proprioception. Architecture and model details are in the fork's
[`docs/VISTA_AGENT_GUIDE.md`](../external/polyumi_vista_policy/docs/VISTA_AGENT_GUIDE.md).

The submodule is **private** (`RickmerKrohn/polyumi_vista_policy`), so `git submodule update --init
external/polyumi_vista_policy` needs collaborator access.

**lamb's copy is hand-managed: `deploy_lamb.sh` does not sync it.** Preserve the working copy
that existing checkpoints require, and install a matching checkout manually on a fresh lamb.

The suite runner supports six models: `see_hear_feel`, `sparsh_x`, `polytouch`, `qformer`,
`mitas` and `vta_diffusion`. Without `--model`, it trains **all six sequentially**. Use the
[worked examples](#worked-examples) to select one model and pass Hydra overrides after `--`.
These configs use `training.num_epochs`. `DRY_RUN=1` prints the `train_vista.py` commands
without training.

### Running the fork's tests

The image can run its own test suite — `pytest` is layered in by
`docker/polyumi_inference.Dockerfile`:

```bash
docker run --rm --user 0:0 \
    -e HOME=/tmp -e HF_HOME=/hf_cache -e MPLCONFIGDIR=/tmp/mpl \
    -v "${HOME}/.cache/huggingface:/hf_cache:rw" \
    polyumi-vista \
    python -m pytest test/test_sensor_ablation.py test/test_vista_shapes.py -q
```

CPU-only, no GPU flag needed. The HF-cache mount is for the AST audio encoder, which the `va` and
`vta` ablation groups pull from the hub on first run.

## What is out of scope here

Robot calibration (`T_gopro_to_fingertip`, `latency.gopro`, the `eef_frame` static TF) is separate
from training. The task config sets `dataset_frequeny: 0`, which zeroes every `latency_steps`,
so these latency settings do not affect training. Before deployment, follow
[calibration-instructions.md](calibration-instructions.md) for the hardware in use.
