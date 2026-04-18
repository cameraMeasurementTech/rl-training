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
| `examples/swe/train_deepswe_32b.sh` | Shell wrapper with Hydra overrides matching the published DeepSWE 32B recipe (multi-node, large context). |

**Not** used for this Parquet+PPO path: `examples/swe/prepare_swe_data.py` (that registers datasets for the older in-process `AgentTrainer` API, not verl Parquet training).

---

## 2. Prerequisites

### 2.1 Hardware and orchestration

The stock `train_deepswe_32b.sh` assumes **8 GPUs per node × 8 nodes** (64 GPUs), tensor parallel size 8, long responses (up to 32k tokens in config), and **SWE environments** backed by **R2E-Gym** (typically **Kubernetes** at scale, or **Docker** for smaller experiments).

Before a full run, decide:

- How many **nodes** and **GPUs per node** you actually have.
- Whether you will use **Docker** or **Kubernetes** for `SWEEnv`.
- Disk space for **Docker images** (the published setup used very large nodes; local `kind` is mentioned in the example README but is not enough for a full reproduction).

You will need to **edit Hydra overrides** (or duplicate the shell script) so `trainer.nnodes`, `trainer.n_gpus_per_node`, `actor_rollout_ref.rollout.tensor_model_parallel_size`, `actor_rollout_ref.actor.ulysses_sequence_parallel_size`, FSDP micro-batches, and related fields match your cluster.

### 2.2 Software

- **Python**: rLLM’s `pyproject.toml` requires **Python ≥ 3.10** (many installs use 3.10–3.12).
- **Git** and network access to **Hugging Face** (dataset and optional model download).
- **CUDA-capable GPUs** for training (vLLM rollout + FSDP actor as configured).
- **R2E-Gym** installed for the SWE environment (see step 4).

### 2.3 Accounts and tokens (optional)

- **Hugging Face**: If the dataset or model is gated, run `huggingface-cli login` (or set `HF_TOKEN`) before data prep or first model load.
- **Weights & Biases**: The example enables `trainer.logger=['console','wandb']`. Set `WANDB_API_KEY` or change the logger list if you do not use W&B.

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

Training reads **Parquet** paths from Hydra, e.g. `data.train_files` and `data.val_files` in `train_deepswe_32b.sh`.

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

`examples/swe/train_deepswe_32b.sh` references:

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

Default in `train_deepswe_32b.sh`: `Qwen/Qwen3-32B`.

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

If your shell script is invoked as `bash train_deepswe_32b.sh`, many environments pass extra args to the **inner** `python` line only if you change the script to `"$@"` or you run `python3 -m rllm.trainer.verl.train_agent_ppo` yourself and paste all overrides.

**Practical pattern:** copy `train_deepswe_32b.sh` to something like `train_deepswe_my_cluster.sh`, replace the single line `actor_rollout_ref.model.path=...`, and adjust GPU counts in the same file.

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

`train_deepswe_32b.sh` sets (before invoking Python):

| Variable | Purpose |
|----------|---------|
| `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | Attention backend for vLLM rollout. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` | CUDA allocator behavior. |
| `VLLM_USE_V1=1` | vLLM V1 engine path. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | Allows long max model length configuration. |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S=...` | Very large timeout for long rollouts. |

Ensure these are appropriate for your vLLM/CUDA version.

---

## 8. Launch training

### 8.1 Working directory

Use the directory that contains your overrides, typically:

```bash
cd examples/swe
```

### 8.2 Run the bundled script

The repository file name is **`train_deepswe_32b.sh`** (some older text refers to `deepswe_32b.sh`; use the actual filename).

```bash
bash train_deepswe_32b.sh
```

This invokes:

```bash
python3 -m rllm.trainer.verl.train_agent_ppo \
  ...many hydra overrides...
```

Ray will start according to verl/rLLM config; all nodes must see the same **Parquet paths** and **model path** (shared filesystem or local mirror per node).

### 8.3 What you must align with your infrastructure

Before expecting a stable run, review and adjust at least:

- `trainer.n_gpus_per_node`, `trainer.nnodes`
- `actor_rollout_ref.rollout.tensor_model_parallel_size`
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size`
- `actor_rollout_ref.actor.fsdp_config.*_offload`, `ppo_micro_batch_size_per_gpu`, `rollout.gpu_memory_utilization`
- `rllm.rllm.agent.trajectory_timeout` and Docker/K8s capacity
- `data.train_files` / `data.val_files` if not under default `RLLM_DIR/data/swe/`

---

## 9. Quick reference: end-to-end command sequence

Conceptual sequence (adapt paths and cluster launcher to your site):

1. Clone **rllm-org/rllm** and **R2E-Gym**; install R2E-Gym, then rLLM with `[verl]` and `[swe]`.
2. Export data:

   ```bash
   cd /path/to/rllm
   python scripts/data/swe_dataset.py --datasets R2E-Gym/R2E-Gym-Subset R2E-Gym/SWE-Bench-Verified
   ```

3. Set `MODEL_PATH` (if you parameterized the script) **or** plan Hydra overrides for `actor_rollout_ref.model.path`.
4. Tune `train_deepswe_32b.sh` for your GPU topology and paths.
5. From `examples/swe`, run `bash train_deepswe_32b.sh` (or your copy) on the head node / job scheduler as required by your Ray cluster.

---

## 10. Further reading

- Example overview: [examples/swe/README.md](README.md)
- rLLM docs: [https://docs.rllm-project.com/](https://docs.rllm-project.com/)
- DeepSWE reproduction details (evaluation, scaling): [R2E-Gym reproduction docs](https://github.com/agentica-project/R2E-Gym/tree/master/reproduction)
