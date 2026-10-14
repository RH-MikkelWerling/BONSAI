#!/usr/bin/env bash
# Build a new, auditable primary-outcome candidate from the sealed server inputs.
# Run from the Python environment after syncing this repository. Never overwrites.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="$REPO_ROOT/data/results/outcomes_final_medfix_20261005_v1"
BASE_DIR="$RUN_ROOT/opera_outcomes_smr_gated"
PANEL_DIR="$RUN_ROOT/final_panel"
LAB_CANDIDATES_DIR="$REPO_ROOT/data/results/final_lab_candidates_20261005T164505Z"
TLS_SNAPSHOT_DIR="$REPO_ROOT/data/results/final_tls_snapshot_20261005T162435Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PANEL_REVIEW_DIR="${OPERA_PANEL_REVIEW_DIR:-$RUN_ROOT/panel_decision_primary_v2_$STAMP}"

for file in \
  "$BASE_DIR/overall_survival.parquet" \
  "$BASE_DIR/hypokalemia_g3plus__audit.parquet" \
  "$PANEL_DIR/assembly_manifest_INTERNAL.json" \
  "$LAB_CANDIDATES_DIR/manifest_INTERNAL.json" \
  "$LAB_CANDIDATES_DIR/outcomes/ALT_increased_g2plus.parquet" \
  "$LAB_CANDIDATES_DIR/outcomes/ALT_increased_g3plus.parquet" \
  "$LAB_CANDIDATES_DIR/outcomes/creatinine_rise_gt1p5x_preindex_baseline.parquet" \
  "$TLS_SNAPSHOT_DIR/candidate_tls_labs_v1_INTERNAL.parquet"; do
  if [[ ! -f "$file" ]]; then
    printf 'Missing required input: %s\n' "$file" >&2
    exit 1
  fi
done

if [[ -z "${OPERA_PANEL_REVIEW_DIR:-}" ]]; then
  if [[ -e "$PANEL_REVIEW_DIR" ]]; then
    printf 'Versioned panel-review path already exists: %s\n' "$PANEL_REVIEW_DIR" >&2
    exit 1
  fi
  python -m opera.run.decide_outcome_panel \
    --outcomes-dir "$PANEL_DIR" \
    --registry opera/configs/experiment_registry.yaml \
    --rules opera/configs/outcome_panel_decision_v1.yaml \
    --output-dir "$PANEL_REVIEW_DIR"
fi
for file in \
  "$PANEL_REVIEW_DIR/summary_INTERNAL.json" \
  "$PANEL_REVIEW_DIR/panel_decisions_INTERNAL.csv" \
  "$PANEL_REVIEW_DIR/input_files_sha256_INTERNAL.csv"; do
  if [[ ! -f "$file" ]]; then
    printf 'Missing panel-review input: %s\n' "$file" >&2
    exit 1
  fi
done

POTASSIUM_DIR="${OPERA_POTASSIUM_DIR:-$REPO_ROOT/data/results/potassium_numeric_v1_$STAMP}"
RELEASE_DIR="$REPO_ROOT/data/results/approved_outcome_release_v2_$STAMP"
if [[ -e "$RELEASE_DIR" ]]; then
  printf 'Versioned release output path already exists; wait one second and rerun.\n' >&2
  exit 1
fi

printf 'Panel review: %s\nLab candidates: %s\nTLS snapshot: %s\n' \
  "$PANEL_REVIEW_DIR" "$LAB_CANDIDATES_DIR" "$TLS_SNAPSHOT_DIR"

if [[ -n "${OPERA_POTASSIUM_DIR:-}" ]]; then
  for file in \
    "$POTASSIUM_DIR/manifest_INTERNAL.json" \
    "$POTASSIUM_DIR/outcomes/potassium_below_3p0_mmol_l.parquet"; do
    if [[ ! -f "$file" ]]; then
      printf 'Requested potassium candidate is incomplete: %s\n' "$file" >&2
      exit 1
    fi
  done
  printf 'Reusing versioned potassium candidate: %s\n' "$POTASSIUM_DIR"
else
  if [[ -e "$POTASSIUM_DIR" ]]; then
    printf 'Versioned potassium output path already exists; wait one second and rerun.\n' >&2
    exit 1
  fi
  python -m opera.run.export_potassium_below_3_v1 \
    --tls-labs "$TLS_SNAPSHOT_DIR/candidate_tls_labs_v1_INTERNAL.parquet" \
    --base-outcomes-dir "$BASE_DIR" \
    --output-dir "$POTASSIUM_DIR"
fi

python -m opera.run.build_approved_outcome_release_v2 \
  --panel-dir "$PANEL_DIR" \
  --base-dir "$BASE_DIR" \
  --panel-review-dir "$PANEL_REVIEW_DIR" \
  --lab-candidates-dir "$LAB_CANDIDATES_DIR" \
  --potassium-candidates-dir "$POTASSIUM_DIR" \
  --output-dir "$RELEASE_DIR"

python -m opera.run.audit_outcome_exports \
  --outcomes-dir "$RELEASE_DIR/outcomes" \
  --registry "$RELEASE_DIR/experiment_registry_primary_v2.yaml" \
  --output "$RELEASE_DIR/structure_audit_INTERNAL.csv"

export BONSAI_OUTCOMES_DIR="$RELEASE_DIR/outcomes"
python -m opera.run.generate_sweep_configs \
  --registry "$RELEASE_DIR/experiment_registry_primary_v2.yaml" \
  --output-dir "$RELEASE_DIR/generated_configs"

python -m opera.run.verify_approved_outcome_training_v2 \
  --release-dir "$RELEASE_DIR" \
  --config "$RELEASE_DIR/generated_configs/direct_cr_bce_lowrank_endpoint_groups_v1.yaml"

printf '\nOutcome release candidate is structurally verified, pending clinical sign-off.\n'
printf 'RELEASE_DIR=%q\n' "$RELEASE_DIR"
printf 'PANEL_REVIEW_DIR=%q\n' "$PANEL_REVIEW_DIR"
printf 'POTASSIUM_DIR=%q\n' "$POTASSIUM_DIR"
printf 'Set BONSAI_OUTCOMES_DIR=%q in the fine-tuning terminal.\n' "$BONSAI_OUTCOMES_DIR"
