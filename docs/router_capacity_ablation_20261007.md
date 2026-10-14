# Router capacity check: residual scale (tuning only)

The historical `time2vec_router_interacting_e2e` run is **not** a valid
interactive 0.1-scale reference: its best and last checkpoints contain no
interaction tensors despite their metadata. Train a fresh 0.1 arm under the
current matched-initialization code before interpreting the scale sweep.
Two new controlled candidates use scales 0.2 and 0.3. Their generated configs
must differ from the reference **only** at
`competing_risk.adapter_residual_scale`; the repository test checks this.
This is a *scale/specialization* test, not the disease-versus-outcome source
factorial or a clean additive-versus-interacting causal contrast.

## Before new GPU jobs

Sync the repository to all servers. Run the read-only pair audit in
`router_query_audit_20261007.md` first and inspect `current_router_code` plus
the saved checkpoint modes. The historical checkpoint does not archive a
training-time source hash, so document that provenance limit.

Use a new candidate directory; do not alter the old manifest or run folders.
The new output path below is an example and must not already exist.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export OPERA_RELEASE_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export OPERA_FAMILY_DIR="$(python -c 'import json; from pathlib import Path; print(json.loads(Path("data/results/approved_neural_candidates_router_20261006T112022Z/candidate_manifest_INTERNAL.json").read_text())["family_dir"])')"
export OPERA_CANDIDATE_DIR="$PWD/data/results/approved_neural_candidates_router_scale_20261007_v1"
export OPERA_NEURAL_RUN_ROOT="$PWD/data/models/approved_neural_router_scale_20261007_v1"
python -m opera.run.prepare_approved_neural_runs_v1 \
  --release-dir "$OPERA_RELEASE_DIR" --family-dir "$OPERA_FAMILY_DIR" \
  --output-dir "$OPERA_CANDIDATE_DIR"
```

On two separate one-GPU servers, use the *same* four exported paths above,
plus `OPERA_LOCAL_GPU=0 OPERA_LAUNCH=1`, and run one variant per server:

```bash
export OPERA_LOCAL_GPU=0 OPERA_LAUNCH=1
bash scripts/run_approved_neural_candidate_v1.sh time2vec_router_interacting_scale02_e2e
```

```bash
export OPERA_LOCAL_GPU=0 OPERA_LAUNCH=1
bash scripts/run_approved_neural_candidate_v1.sh time2vec_router_interacting_scale03_e2e
```

On a third GPU, train `time2vec_router_interacting_e2e` in the **new** run
root as the 0.1 reference. Also rerun `time2vec_router_additive_e2e` in the
same root for the clean query-mode ablation. Do not resume or overwrite the
historical October 6 checkpoints; they remain descriptive context only.

## Decision rule

Before interpreting the new arms, verify the resolved `hydra/config.yaml`,
`best.ckpt`, completion marker, encoder checkpoint, release, and seed. Compare
the three arms at a common prespecified epoch and by the same checkpoint rule;
report endpoint-group-macro CIF-AUC, 365-day per-outcome AUC and Brier, plus
family-level trade-offs. The existing metric logger records route shared mass,
entropy, and effective residual magnitude. Those metrics tell us whether
greater permitted scale actually changed the model, not whether it helped.

Evaluate all three fixed checkpoints on the **same tuning patients** for paired
predictions and route exports. Use patient-paired bootstrap intervals for the
predefined contrasts 0.2–0.1 and 0.3–0.1; the intervals are conditional on
checkpoint selection on tuning and are not held-out generalization estimates.
Use matched seeds for any confirmatory rerun, including a new 0.1 reference if
training-time code provenance of the historical run remains uncertain.

Do not choose a scale from the prettiest embedding arrows. Compare full-vector
shift and changes in CIF, and inspect whether gains are concentrated in
well-supported outcomes or come at a calibration/family cost. Keep held-out
sealed until the architecture and scale are frozen.
