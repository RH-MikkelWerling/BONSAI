#!/usr/bin/env bash
set -euo pipefail

# Bundle validation-stage architecture results without opening held-out outputs.
# Usage:
#   bash scripts/bundle_architecture_validation.sh RUN_ROOT [OUTPUT.tar.gz]

RUN_ROOT="${1:?Usage: $0 RUN_ROOT [OUTPUT.tar.gz]}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${2:-$PWD/architecture_validation_metrics_${STAMP}.tar.gz}"

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Run root does not exist: $RUN_ROOT" >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing archive: $OUTPUT" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TMP_ROOT"' EXIT
BUNDLE="$TMP_ROOT/architecture_validation_metrics"
mkdir -p "$BUNDLE/runs" "$BUNDLE/provenance"

shopt -s nullglob
RUNS=(
  "$RUN_ROOT"/direct_cr_architecture_*
  "$RUN_ROOT"/direct_cr_objective_horizon_bce
)

if (( ${#RUNS[@]} == 0 )); then
  echo "No architecture or horizon-BCE run directories found below $RUN_ROOT" >&2
  exit 1
fi

for RUN in "${RUNS[@]}"; do
  [[ -d "$RUN" ]] || continue
  NAME="$(basename "$RUN")"
  DEST="$BUNDLE/runs/$NAME"
  mkdir -p "$DEST"

  while IFS= read -r -d '' FILE; do
    REL="${FILE#"$RUN"/}"
    # Keep held-out/test artifacts sealed during model selection.
    case "/${REL,,}/" in
      */held_out/*|*/heldout/*|*/test/*|*/evaluation/*|*predictions*) continue ;;
    esac
    mkdir -p "$DEST/$(dirname "$REL")"
    cp -p "$FILE" "$DEST/$REL"
  done < <(
    find "$RUN" -type f \
      \( -name 'metrics.csv' \
         -o -name 'hparams.yaml' \
         -o -name 'checkpoint_metadata.json' \
         -o -name 'training_complete.json' \
         -o -name 'competing_risk_interval_support.csv' \
         -o -name 'competing_risk_null_log_hazards.csv' \
         -o -name 'config.yaml' \
         -o -name 'overrides.yaml' \
         -o -name 'hydra.yaml' \
         -o -name '*.log' \) -print0
  )
done

# Snapshot the exact code/config state used on this server checkout. This is
# small, and lets us audit the horizon-BCE implementation difference directly.
for FILE in \
  opera/modules/networks/competing_risk.py \
  opera/modules/networks/contrastive_net.py \
  opera/modules/lightningmodules/OperaContrastiveModule.py \
  opera/run/contrastive_multicohort.py \
  opera/configs/experiment_registry.yaml \
  opera/configs/generated/direct_cr_architecture_*.yaml \
  opera/configs/generated/direct_cr_objective_horizon_bce.yaml
do
  [[ -f "$FILE" ]] || continue
  mkdir -p "$BUNDLE/provenance/$(dirname "$FILE")"
  cp -p "$FILE" "$BUNDLE/provenance/$FILE"
done

git rev-parse HEAD > "$BUNDLE/provenance/git_revision.txt" 2>&1 || true
git status --short > "$BUNDLE/provenance/git_status.txt" 2>&1 || true
git log -20 --date=iso --format='%H %ad %s' \
  > "$BUNDLE/provenance/git_log.txt" 2>&1 || true

find "$BUNDLE" -type f -printf '%P\t%s bytes\n' | sort \
  > "$BUNDLE/CONTENTS.tsv"

tar -C "$TMP_ROOT" -czf "$OUTPUT" "$(basename "$BUNDLE")"

echo "Wrote: $OUTPUT"
echo "Runs discovered:"
printf '  %s\n' "${RUNS[@]##*/}"
echo "Files bundled: $(wc -l < "$BUNDLE/CONTENTS.tsv")"
echo "Archive size: $(du -h "$OUTPUT" | cut -f1)"
