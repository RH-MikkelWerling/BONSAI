# Server Requirements

This document logs what BONSAI actually requires to run — both its Python dependency manifest and the server/infrastructure setup it is deployed against. Since pretraining has not outperformed XGBoost, we are pivoting toward an "EveryQuery"-style pretraining approach (separate repo). This log is the baseline for that comparison: it captures the current offline server environment so the next step can identify what carries over as-is versus what needs to change.

## 1. Dependency manifest

**Package manager:** pip + setuptools (`pyproject.toml`, PEP 621). No poetry/uv/conda. **No lock file** is committed — most versions are range-constrained, not pinned.

`requires-python = ">=3.12"`, project version `2.1.0`.

**Core dependencies** (`[project.dependencies]`, all range-constrained):

| Package | Constraint |
|---|---|
| numpy | `>=1.26,<3` |
| lightning | `>=2.5,<3` |
| hydra-core | `>=1.3.2` |
| pandas | `>=2.2,<4` |
| polars | `>=1.20.0` |
| pyarrow | `>=15.0.0` |
| PyYAML | `>=6.0.3` |
| scikit-learn | `>=1.5,<2` |
| scipy | `>=1.11.0` |
| torch | `>=2.10,<3` |
| torchmetrics | `>=1.6.0` |
| tqdm | `>=4.66.0` |
| matplotlib | `>=3.9.0` |
| python-dotenv | `>=1.0.1` |

**Optional extras** (`[project.optional-dependencies]`):

| Extra | Contents | Purpose |
|---|---|---|
| `dev` | `pytest>=8,<10`, `ruff==0.15.1` (exact), `docstr-coverage`, `coverage>=7` | lint/test |
| `tabular` | `xgboost>=2.0` | the current tabular baseline being compared against |
| `visualization` | `adjustText`, `pacmap`, `statsmodels`, `umap-learn` | embedding/diagnostic plots |
| `survival` | `lifelines>=0.30.0` | survival/hematology finetuning |
| `retrieval` | `faiss-cpu>=1.8.0` | embedding retrieval |
| `tabpfn` | `tabpfn` (unpinned) | tabular baseline |
| `bayesian` | `pymc>=5.20,<7`, `arviz>=0.20,<1` | Bayesian diagnostics |
| `flash_attn` | `packaging==25.0`, `psutil==7.2.0`, `ninja==1.13.0`, `flash-attn==2.8.3` (all exact) | FlashAttention 2; needs GCC ≥9, Linux/CUDA only |

**CI-only pins** (`requirements/constraints-ci.txt`, a pip `-c` constraints file, not a full lock): pins `coverage`, `hydra-core`, `lightning`, `matplotlib`, `numpy`, `pandas`, `polars`, `pyarrow`, `pytest`, `ruff`, `scikit-learn`, `torch`, `torchmetrics` to exact versions. Used only by `.github/workflows/quality.yml` and `pipeline.yml` for reproducible CI — it does not cover extras (torch's CUDA build, flash-attn, xgboost, faiss-cpu, tabpfn, pymc) and is not used for server deployment.

**Manual install flow** (`README.md`, `SERVER_RUNBOOK.md`):
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,tabular,survival]"
# then: install the CUDA-compatible torch build required by the cluster
python -m pip install -e ".[flash_attn]"
# fallback if flash-attn can't be built: model.attn_type=sdpa
```

## 2. Server / infrastructure requirements

- **OS / Python:** Linux, Python 3.12. A Windows PowerShell venv path exists for local dev only — FlashAttention is explicitly Linux/CUDA-only and unavailable there.
- **GPU:** single NVIDIA V100 (Volta). FP16 mixed precision (`16-mixed`) — no native BF16 support, so requesting BF16 falls back/converts and can blow up memory. `torch.compile` is explicitly disabled in the production hardware profile (`configs/hardware/1gpu6cpu.yaml`) because it can materialize the full attention tensor on V100, defeating the memory-efficient fused SDPA path. A CPU fallback profile (`configs/hardware/cpu.yaml`) exists for local dev only.
- **Scheduling:** SLURM only. Example wrapper (`SERVER_RUNBOOK.md`): `--gres=gpu:1 --cpus-per-task=8 --mem=64G --time=48:00:00`. No containers anywhere in the repo — no Dockerfile, Singularity, or Apptainer config.
- **Data access:** plain local filesystem, no database. EHR data arrives as MEDS-format parquet from the `ehr2meds` pipeline, then gets tokenized into `.pt` files loaded with plain `torch.load`. Server paths follow a `/project/...`-style mount convention. Patient data, checkpoints, and results must never live inside the git worktree.
- **Environment variables** (`.env.example`, loaded via `.env` + Hydra's `${oc.env:...}` resolver): `BONSAI_CONFIG_PATH` (must be absolute), `BONSAI_MODELS`, `BONSAI_PROCESSED_DATA`, `BONSAI_PREDICTIONS`, `BONSAI_CHECKPOINT_ROOT`, `BONSAI_RESULTS_ROOT`, `EHR2MEDS_OUTPUT`, `BONSAI_COHORT_MEMBERSHIP`, `BONSAI_OUTCOMES_DIR`.
- **Experiment tracking:** no wandb, mlflow, or tensorboard anywhere in `bonsai/` or `opera/` source. The only logger wired up is Lightning's local `CSVLogger`; monitoring is a local CLI (`opera.run.monitor_training`) reading CSV/JSON artifacts on disk.
- **Offline / air-gapped:** the deployment target is explicitly called the "offline environment" in a couple of places (e.g. falling back to `--method pca` when UMAP is unavailable). However, **there is no documented package-installation mechanism for the air-gapped server** — no vendored wheels, no local package index/mirror configuration, and no lock file with hashes to drive a `pip download` / `--no-index` install. Anyone deploying today has to pre-resolve and pre-stage wheels for every ranged dependency (and the CUDA-matched torch build, and flash-attn) themselves.

## 3. Preliminary EveryQuery compatibility notes (not yet a full audit)

Based on `.tmp/everyquery-audit/pyproject.toml` — a read-only checkout of the EveryQuery repo (uv-managed, cloned 2026-08-12) already sitting in this repo's `.tmp/` scratch directory:

- **Package manager mismatch:** EveryQuery uses `uv` with a committed, fully-resolved `uv.lock`; BONSAI uses plain pip with no lock file. Worth checking whether `uv` + a pre-resolved lock is actually *easier* to stage on the air-gapped server than BONSAI's current ranged-pip approach, given the gap noted above.
- **Python:** EveryQuery requires `>=3.11`; the server already runs 3.12 — compatible, no action needed.
- **Overlapping ranges that look compatible:** `torch` (`>=2.6,<3` vs BONSAI's `>=2.10,<3`), `lightning` (`>=2.5,<3`, identical), `hydra-core`, `numpy`, `scikit-learn`, `matplotlib`, `torchmetrics`, `polars`, `pyarrow`.
- **New hard dependencies BONSAI doesn't currently install:** `meds`, `meds-transforms`, `meds-torch-data[lightning]`, `transformers`, `torchdata` (StatefulDataLoader for mid-epoch resume), `filelock`, `ipykernel`. All would need to be staged for the offline server.
- **Likely biggest blocker:** `wandb>=0.22.3,<1` is a hard runtime dependency in EveryQuery, but BONSAI's server has no wandb usage and no offline-mode wandb setup today (no internet access). Would need `WANDB_MODE=offline` plus the package itself pre-staged, or a substitution.
- **Possible win:** EveryQuery's dev dependency `hydra-submitit-launcher` targets SLURM submission via submitit — potentially a good fit given the server is already SLURM-based, worth evaluating rather than treating as a blocker.

The full compatibility audit against EveryQuery is a separate follow-up task now that this baseline is logged.
