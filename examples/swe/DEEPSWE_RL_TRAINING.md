# DeepSWE-style RL training: preparation and runbook

This document walks through preparing the **same dataset family** and **same training entrypoint** used for DeepSWE (rLLM agent PPO with verl + vLLM), and how to point training at **your** base model.

Official references:

- Blog: [DeepSWE: Training a Fully Open-sourced State-of-the-Art Coding Agent by Scaling RL](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33)
- Dataset: [R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) on Hugging Face
- Framework: [rLLM](https://github.com/rllm-org/rllm) (active); the older [agentica-project/rllm](https://github.com/agentica-project/rllm) tree is archived

---

## 1. What you are running

| Piece | Role |
|--------|------|
| `scripts/data/swe_dataset.py` | Downloads HF SWE-style datasets and writes **Parquet** files under `data/swe/` (or `--local_dir`) in the format `train_agent_ppo` expects. |
| `python -m rllm.trainer.verl.train_agent_ppo` | Distributed **PPO** training with **verl**, **Ray**, actor/rollout/ref model, and rLLM’s SWE agent/env. |
| `examples/swe/train_deepswe_32b.sh` | Shell wrapper for the published **multi-node** recipe (`trainer.nnodes=8`, **64 GPUs** total). |
| `examples/swe/train_deepswe_32b_1x8gpu.sh` | Same Hydra recipe with **`trainer.nnodes=1`** — **one machine, eight GPUs** (see §2.1). |

**Not** used for this Parquet+PPO path: `examples/swe/prepare_swe_data.py` (that registers datasets for the older in-process `AgentTrainer` API, not verl Parquet training).

---

## 2. Prerequisites

### 2.1 Hardware requirements

#### Single workstation: **1 node × 8 GPUs** (your setup)

The **rLLM + verl** stack does **not** require multiple physical machines. Ray treats `trainer.nnodes` as the number of **Ray nodes** in the job; on one box you want **`trainer.nnodes=1`** and **`trainer.n_gpus_per_node=8`**.

- **Use this repo’s script:** [`train_deepswe_32b_1x8gpu.sh`](train_deepswe_32b_1x8gpu.sh) — it is the same command line as [`train_deepswe_32b.sh`](train_deepswe_32b.sh) except `trainer.nnodes=1` instead of `8`.
- **Do not** run [`train_deepswe_32b.sh`](train_deepswe_32b.sh) unchanged on one machine: it sets `trainer.nnodes=8`, which requests **eight Ray nodes** (typically eight hosts), i.e. **64 GPUs** — that is a configuration mistake for a single server, not a framework limitation.
- **Resource pool:** the code builds one GPU pool per Ray node (`[n_gpus_per_node] * nnodes`; see `init_resource_pool_mgr` in [`rllm/trainer/verl/train_agent_ppo.py`](../../rllm/trainer/verl/train_agent_ppo.py)). With `nnodes=1` and `n_gpus_per_node=8`, you get **one pool of eight GPUs** on localhost — supported.
- **Hybrid engine:** [`agent_ppo_trainer.py`](../../rllm/trainer/verl/agent_ppo_trainer.py) expects `actor_rollout_ref.hybrid_engine=True` for this path (actor + rollout fused). That layout is compatible with a **single** eight-GPU pool; you still need enough **VRAM** and **host RAM** for the chosen model and context (see below).
- **Orchestration:** multi-node **interconnect and shared NFS** are unnecessary on one host; Parquet and checkpoints can live on **local disk**. You still need **Docker or Kubernetes** for R2E-Gym `SWEEnv` unless you have changed the env backend elsewhere.
- **Reality check:** the default recipe keeps **Qwen3-32B**, **32k** max response length, **`rollout.n=8`**, and **tensor parallel size 8**. That is **very demanding** even on **8×80GB** cards; if you hit OOM or timeouts, reduce context, `rollout.n`, batch sizes, or switch to a smaller base model and retune — the codebase allows those Hydra edits on one node the same as on many.

#### Published DeepSWE / example README scale (target reproduction)

These numbers come from the DeepSWE example README and the stock training shell; they describe the **intended** cluster class, not a strict minimum for every possible code path.

| Resource | Requirement | Notes |
|----------|-------------|--------|
| **GPUs (CUDA)** | **≥ 64** data-parallel / cluster GPUs for the default script | `trainer.n_gpus_per_node=8` and `trainer.nnodes=8` in `train_deepswe_32b.sh`. |
| **GPU generation** | **High-memory datacenter GPUs** (e.g. **80 GB H100** or **80 GB A100** class) strongly recommended | **Qwen3-32B** training plus **vLLM** rollout with `data.max_response_length=32768`, `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000`, and `rollout.n=8` implies very large activation and KV footprints. Lower-memory cards usually need a different model, smaller context, lower `rollout.n`, higher TP, or aggressive re-tuning. |
| **Tensor parallelism (rollout)** | **8** (`actor_rollout_ref.rollout.tensor_model_parallel_size=8`) | One vLLM engine spans **8 GPUs** on a node for generation. Your **per-node GPU count** must be at least this value (unless you change TP). |
| **Sequence / actor parallelism** | **8** (`actor_rollout_ref.actor.ulysses_sequence_parallel_size=8`) | Must remain **compatible** with your per-node layout and verl’s expectations when you change topology. |
| **CPUs per node (SWE + Ray)** | **Very large CPU count** for full reproduction | Example README: on the order of **~200 CPUs per node** in their Kubernetes setup, to schedule many **R2E-Gym** repo environments and Ray tasks alongside training. |
| **System RAM** | **Large** (hundreds of GB per node is plausible at this scale) | FSDP with **optimizer and parameter offload** (`actor_rollout_ref.actor.fsdp_config.param_offload` / `optimizer_offload`) reduces GPU memory but **increases host RAM** traffic. |
| **Node local disk** | **Multi-terabyte per node** for the published setup | Example README: **6 TB+ per node** to hold **thousands** of per-repo **Docker images** pulled for R2E-Gym tasks. |
| **Parallel containers** | On the order of **512 Docker containers** in parallel (per README) | Requires orchestration (Kubernetes at scale), image registry bandwidth, and cgroup/disk headroom—not a laptop workload. |
| **Network** | **Low-latency, high-bandwidth** interconnect between nodes | Ray and verl move tensors and logs across the cluster; slow NFS or WAN-mounted checkpoints worsen training and can dominate wall time. |

For **one machine and eight GPUs**, ignore multi-node network requirements; keep the **GPU memory** and **CPU/RAM for SWE + Ray** rows in mind.

#### Minimum practical lab (illustrative only)

There is **no officially supported “small GPU count”** profile in the stock multi-node script. For experimentation you might:

- Use a **smaller base model** than 32B and shrink **max token lengths**, **batch sizes**, and **`rollout.n`**.
- Reduce **`tensor_model_parallel_size`** and **`trainer.n_gpus_per_node`** together consistently.
- Run **fewer parallel envs** (lower `rllm` / env parallelism in Hydra) and accept lower statistical throughput.

Expect to **iterate on Hydra** and possibly hit OOM or timeout until the configuration matches your **VRAM**, **host RAM**, and **CPU** envelope.

#### Data preparation machine

`scripts/data/swe_dataset.py` is **CPU- and RAM-heavy** for [R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) because each row’s `extra_info` embeds large JSON. A machine with **≥ 32 GB RAM** is a reasonable starting point; use `SWE_DATASET_BATCH_SIZE` (see §5.1) if you see the process killed by the OOM killer.

#### CUDA / driver

Match **PyTorch, CUDA, vLLM, and flash-attn** wheels to your driver and GPU architecture (see `pyproject.toml` pins for the `[verl]` extra). Wrong combinations fail at import or kernel launch.

### 2.2 Orchestration and topology

Before a full run, decide:

- How many **Ray nodes** (`trainer.nnodes`) and **GPUs per node** (`trainer.n_gpus_per_node`) you actually have, and whether that supports **`tensor_model_parallel_size=8`** as written (needs **≤** `n_gpus_per_node` on each Ray node that runs the vLLM worker group).
- Whether **SWEEnv** will use **Docker** or **Kubernetes**; at DeepSWE scale, **Kubernetes** is the expected backend, but **Docker** on a single strong workstation is a common dev path. **`SWEEnv` defaults to `backend=kubernetes` in code** — on a box without in-cluster or `~/.kube/config`, R2E-Gym fails with `Invalid kube-config file` / `Service host/port is not set`. The bundled `train_deepswe_32b*.sh` scripts set **`+rllm.env.env_args.backend=docker`** (leading **`+`** is required so Hydra can add the key under struct `env_args`) for local Docker; on a real cluster use **`+rllm.env.env_args.backend=kubernetes`** and ensure kubeconfig (or in-cluster credentials) is available to every worker.
- **Multi-node only:** whether every node sees the same **shared filesystem** for Parquet paths and checkpoints (or you replicate artifacts). **Single-node:** local paths are enough.

You will need **`trainer.nnodes=1`** for one physical machine (see [`train_deepswe_32b_1x8gpu.sh`](train_deepswe_32b_1x8gpu.sh)). For multiple hosts, set `trainer.nnodes` to the number of machines and **edit Hydra overrides** so `actor_rollout_ref.rollout.tensor_model_parallel_size`, `actor_rollout_ref.actor.ulysses_sequence_parallel_size`, FSDP micro-batches, and related fields match your cluster.

### 2.3 Software

- **Python**: rLLM’s `pyproject.toml` requires **Python ≥ 3.10** (many installs use 3.10–3.12).
- **Git** and network access to **Hugging Face** (dataset and optional model download).
- **CUDA-capable GPUs** for training (vLLM rollout + FSDP actor as configured).
- **R2E-Gym** installed for the SWE environment (see §3).

### 2.4 API keys and credentials

Nothing below is **intrinsic** to rLLM’s algorithms; each key is only needed when you turn on the corresponding integration or use a **private** resource.

| When you need it | Variable or mechanism | Required? |
|------------------|------------------------|-----------|
| **Public** Hugging Face datasets and models (default paths like `R2E-Gym/R2E-Gym-Subset`, `Qwen/Qwen3-32B`) | _(none)_ | **No** — anonymous download is enough for many public assets. |
| **Gated** or **private** Hugging Face datasets or checkpoints | `HF_TOKEN` (recommended), or interactive `huggingface-cli login` | **Yes**, for that asset — set `HF_TOKEN` in the environment (or in CI secrets) **before** §5 (data export) and before workers first load `actor_rollout_ref.model.path`. The Hub also honors `HUGGING_FACE_HUB_TOKEN` in many clients; prefer **`HF_TOKEN`** for consistency with current Hugging Face tooling. |
| **Weights & Biases** logging (`trainer.logger=['console','wandb']` in the bundled scripts) | `WANDB_API_KEY` | **Yes**, if you keep **`wandb`** in `trainer.logger` — W&B will prompt or fail without a project API key. Either export `WANDB_API_KEY` on every node that runs training code, or set `trainer.logger=['console']` (or another backend) in Hydra so W&B is not used. |
| **W&B entity / project** (optional naming) | `WANDB_ENTITY`, `WANDB_PROJECT`, or align `trainer.project_name` / `trainer.experiment_name` with your W&B dashboard | **No** — convenience only. |
| **Private Docker / OCI registry** (custom SWE images not on Docker Hub) | Registry credentials (e.g. `docker login`, imagePullSecrets on Kubernetes, or your cloud’s secret store) | **Yes**, if you use private images — not part of rLLM defaults; depends on R2E-Gym / your cluster. |
| **Cloud Kubernetes** (EKS, GKE, AKS, etc.) | Cloud provider + cluster auth (e.g. IAM roles, `gcloud`/Application Default Credentials, service accounts) | **Yes**, if training uses that cluster — operational, not an rLLM API key. |
| **Ray** on a hosted or cross-account control plane | Whatever your Ray deployment docs require (e.g. Anyscale or custom) | **Only if** your site wires that in. |

**Summary:** For a minimal open setup, you may need **no keys**. Add **`HF_TOKEN`** when the dataset or base model is gated/private. Add **`WANDB_API_KEY`** (or remove `wandb` from `trainer.logger`) before §8 so logging matches your policy.

---

## 3. Clone repositories

From a working directory of your choice:

1. **rLLM** (use the maintained org):

   ```bash
   git clone https://github.com/rllm-org/rllm.git
   cd rllm
   ```

2. **R2E-Gym** (SWE execution scaffold; required for RL rollouts in this stack):

   ```bash
   git clone https://github.com/agentica-project/R2E-Gym.git
   cd R2E-Gym
   pip install -e .
   cd ../rllm
   ```

---

## 4. Create a Python environment and install rLLM + verl stack

From the **rLLM repo root**:

1. Create and activate a virtual environment (conda or `python -m venv` / `uv venv`).

2. Install the **editable model gateway** (path dependency in `pyproject.toml`):

   ```bash
   pip install -e ./rllm-model-gateway
   ```

3. Install **rLLM** with the **verl** extra (pulls `verl`, `ray`, `torch`, `vllm`, `flash-attn`, etc., per `pyproject.toml`):

   ```bash
   pip install -e ".[verl]"
   ```

4. Install **SWE-related** extras used by environments and tooling (Docker/K8s client, etc.):

   ```bash
   pip install -e ".[swe]"
   ```

If `flash-attn` or `vllm` fails to build on your platform, resolve CUDA/toolchain versions first; the training script is built around this stack.

**Note:** Older blog/README text sometimes says `pip install -e ./verl`; the current **rllm-org** packaging often uses **PyPI `verl`** via the `[verl]` extra instead of a git submodule. Follow `pyproject.toml` in your checkout.

---

## 5. Prepare training and validation Parquet files

Training reads **Parquet** paths from Hydra, e.g. `data.train_files` and `data.val_files` in the `train_deepswe_32b*.sh` scripts (same paths in both).

If your Hugging Face dataset or snapshot is **gated**, set **`HF_TOKEN`** (or log in with the CLI) before running the export commands below — see **§2.4**.

### 5.1 Run the dataset export script

From **rLLM repo root**:

```bash
python scripts/data/swe_dataset.py
```

Behavior:

- For each configured Hugging Face dataset id (see `SWE_DATASETS` in `scripts/data/swe_dataset.py`), loads the **`train`** split if present, else **`test`**.
- Writes `<DatasetNamePart>.parquet` under **`{RLLM_DIR}/data/swe/`**, where `RLLM_DIR` is the parent of the installed `rllm` package (for editable installs, that is your **clone root**).

For **only** [R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) (smaller download, matches DeepSWE training data):

```bash
python scripts/data/swe_dataset.py --datasets R2E-Gym/R2E-Gym-Subset
```

Optional arguments:

| Argument | Meaning |
|----------|---------|
| `--local_dir /path/to/dir` | Write Parquet files somewhere other than `{RLLM_DIR}/data/swe`. |
| `--hdfs_dir ...` | If set, also copies each file to HDFS using verl’s helpers (requires verl import path). |

**Memory:** Rows include large JSON fields (`extra_info`). The script writes **in batches**; if you still hit OOM, lower batch size:

```bash
export SWE_DATASET_BATCH_SIZE=16
python scripts/data/swe_dataset.py --datasets R2E-Gym/R2E-Gym-Subset
```

### 5.2 Files expected by the default training script

Both `train_deepswe_32b.sh` and `train_deepswe_32b_1x8gpu.sh` reference:

- **Train:** `${RLLM_DIR}/data/swe/R2E_Gym_Subset.parquet`  
  Produced when `R2E-Gym/R2E-Gym-Subset` is processed (name: `R2E_Gym` + `_Subset` → `R2E_Gym_Subset.parquet`).
- **Val:** `${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet`  
  Produced when `R2E-Gym/SWE-Bench-Verified` is included in the export.

So for a **minimal** setup matching the script defaults:

```bash
python scripts/data/swe_dataset.py --datasets R2E-Gym/R2E-Gym-Subset R2E-Gym/SWE-Bench-Verified
```

If you use `--local_dir`, update **`data.train_files`** and **`data.val_files`** in your Hydra overrides to those absolute paths.

---

## 6. Configure your base model (checkpoint or Hub id)

The policy/actor tokenizer and weights root are controlled by:

```text
actor_rollout_ref.model.path
```

Default in both bundled scripts: `Qwen/Qwen3-32B`.

Set it to either:

- A **Hugging Face model id** (e.g. `Qwen/Qwen3-32B` or your own uploaded repo), or  
- A **local directory** with a standard Transformers layout (`config.json`, tokenizer files, weight shards).

### 6.1 Override without editing the shell script

Hydra **appends** can override the baked-in value. Example pattern (conceptually; you pass this as extra arguments after the script’s own `python3 -m ...` line or duplicate the script):

```text
actor_rollout_ref.model.path=/absolute/path/to/your/hf_checkpoint
```

Or:

```text
actor_rollout_ref.model.path=your-org/your-model-on-hub
```

If you invoke `bash train_deepswe_32b_1x8gpu.sh` (or the multi-node script), many shells **do not** forward extra arguments to the inner `python3 -m ...` line unless you add `"$@"` to that line. To override Hydra from the CLI, run `python3 -m rllm.trainer.verl.train_agent_ppo` yourself and paste all overrides, or copy a script and edit it.

**Practical pattern:** copy `train_deepswe_32b_1x8gpu.sh` to something like `train_deepswe_my_machine.sh`, replace the line `actor_rollout_ref.model.path=...`, and adjust `trainer.n_gpus_per_node` / `tensor_model_parallel_size` if you have fewer than eight GPUs.

**Optional operator knob:** you can replace that line with:

```bash
actor_rollout_ref.model.path=${MODEL_PATH:-Qwen/Qwen3-32B}
```

Then:

```bash
export MODEL_PATH=/abs/path/or/hub/id
bash train_deepswe_my_cluster.sh
```

---

## 7. Environment variables used by the example training script

The `train_deepswe_32b*.sh` scripts set (before invoking Python):

| Variable | Purpose |
|----------|---------|
| `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | Attention backend for vLLM rollout. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` | CUDA allocator behavior. |
| `VLLM_USE_V1=1` | vLLM V1 engine path. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | Allows long max model length configuration. |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S=...` | Very large timeout for long rollouts. |

Ensure these are appropriate for your vLLM/CUDA version.

**API keys (you set these; the shell script does not):** export on the **driver and workers** as needed — full matrix in **§2.4**.

| Variable | Typical use in this workflow |
|----------|-------------------------------|
| `HF_TOKEN` | Download **gated** Hub datasets during §5; load **gated** or private `actor_rollout_ref.model.path` during §8. |
| `HUGGING_FACE_HUB_TOKEN` | Alternate name some libraries accept; **`HF_TOKEN`** is preferred for new setups. |
| `WANDB_API_KEY` | Required if `trainer.logger` includes **`wandb`** (default in both `train_deepswe_32b*.sh` scripts). |
| `WANDB_ENTITY` / `WANDB_PROJECT` | Optional W&B dashboard routing. |

---

## 8. Launch training

### 8.1 Working directory

Use the directory that contains your overrides, typically:

```bash
cd examples/swe
```

### 8.2 Run the bundled script

Before starting Ray/verl, ensure **§2.4** credentials are available everywhere the job runs: e.g. **`WANDB_API_KEY`** if logging to W&B is enabled, and **`HF_TOKEN`** if the policy model or tokenizer is loaded from a private Hub repo.

**One machine, eight GPUs** (recommended entrypoint):

```bash
bash train_deepswe_32b_1x8gpu.sh
```

**Multi-node / 64-GPU** reproduction (original DeepSWE scale):

```bash
bash ./examples/swe/train_deepswe_32b.sh
```

Some older docs refer to `deepswe_32b.sh`; the files in this tree are **`train_deepswe_32b_1x8gpu.sh`** and **`train_deepswe_32b.sh`**.

Both scripts invoke:

```bash
python3 -m rllm.trainer.verl.train_agent_ppo \
  ...many hydra overrides...
```

Ray will start according to verl/rLLM config. **Multi-node:** every node must see the same **Parquet paths** and **model path** (shared filesystem or mirrored data). **Single-node:** local paths for `${RLLM_DIR}/data/swe/*.parquet` and the model are sufficient.

### 8.3 What you must align with your infrastructure

Before expecting a stable run, review and adjust at least:

- `trainer.n_gpus_per_node`, `trainer.nnodes`
- `actor_rollout_ref.rollout.tensor_model_parallel_size`
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size`
- `actor_rollout_ref.actor.fsdp_config.*_offload`, `ppo_micro_batch_size_per_gpu`, `rollout.gpu_memory_utilization`
- `rllm.agent.trajectory_timeout` and Docker/K8s capacity
- `data.train_files` / `data.val_files` if not under default `RLLM_DIR/data/swe/`

### 8.4 How long training runs (steps, epochs, validation)

Training length is controlled by **verl’s `RayPPOTrainer`**, which sets `total_training_steps` as follows (see `verl/trainer/ppo/ray_trainer.py` in your environment):

- **Default:** `len(train_dataloader) * trainer.total_epochs` (one “step” here is one **PPO optimizer update** after a train batch of rollouts).
- **Override:** if **`trainer.total_training_steps`** is set to a positive integer, that value **replaces** the product above and training **stops** once `global_steps` reaches it (see the loop in [`rllm/trainer/verl/agent_ppo_trainer.py`](../../rllm/trainer/verl/agent_ppo_trainer.py)).

**Practical recipe for “short first run, then scale up”:**

| Hydra key | Effect |
|-----------|--------|
| **`trainer.total_training_steps=50`** (example) | Hard cap on **PPO updates** — best knob for a **quick smoke** run regardless of dataset size. |
| **`trainer.total_epochs=1`** (or small) | Fewer full passes over the train parquet; total steps still scale with `len(train_dataloader)` unless you also set `total_training_steps`. |
| **`data.train_batch_size`** | Larger batch ⇒ **fewer** dataloader batches per epoch (faster wall-clock per epoch, different RL statistics). |
| **`trainer.test_freq=1`** | Run **validation** every PPO step — useful to watch reward, but **very expensive** for SWE (Docker/K8s + rollouts). Try `2`, `5`, or `10` first. |
| **`trainer.save_freq=1`** | Save a checkpoint every step — use a small number while debugging, then increase. |
| **`trainer.val_before_train=True`** | Run validation **once** before the first update (default in many verl configs; DeepSWE scripts set **`False`**). Turn on for an immediate baseline. |
| **`trainer.val_only=True`** | Only run validation, then exit (no PPO updates) — good for **eval-only** checks. |

**Note:** `rllm.agent.max_steps` is the **maximum turns per SWE trajectory** (agent–env interaction), **not** the number of PPO training steps.

There is **no built-in “stop when validation reward plateaus”** early stopping in this path; use short **`total_training_steps`**, inspect W&B / console metrics, then **resume** with `trainer.resume_mode` / `trainer.resume_from_path` (verl checkpointing) for a longer second phase if quality looks promising.

---

## 9. Quick reference: end-to-end command sequence

Conceptual sequence (adapt paths and cluster launcher to your site):

1. Clone **rllm-org/rllm** and **R2E-Gym**; install R2E-Gym, then rLLM with `[verl]` and `[swe]`.
2. Export data:

   ```bash
   cd /path/to/rllm
   python scripts/data/swe_dataset.py --datasets R2E-Gym/R2E-Gym-Subset R2E-Gym/SWE-Bench-Verified
   ```

3. Set `MODEL_PATH` (if you parameterized a local copy of the script) **or** plan Hydra overrides for `actor_rollout_ref.model.path`.
4. For **one host and eight GPUs**, use [`train_deepswe_32b_1x8gpu.sh`](train_deepswe_32b_1x8gpu.sh) as-is or tune Hydra further; for **eight nodes**, use [`train_deepswe_32b.sh`](train_deepswe_32b.sh) and align paths across nodes.
5. From `examples/swe`, run the appropriate `bash train_deepswe_32b_*.sh` (or your copy) on the machine or cluster head node as required by your Ray setup.

---

## 10. Further reading

- Example overview: [examples/swe/README.md](README.md)
- rLLM docs: [https://docs.rllm-project.com/](https://docs.rllm-project.com/)
- DeepSWE reproduction details (evaluation, scaling): [R2E-Gym reproduction docs](https://github.com/agentica-project/R2E-Gym/tree/master/reproduction)
