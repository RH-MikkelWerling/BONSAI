# One-pass final outcome triage (INTERNAL)

`opera.run.review_final_outcome_inputs` is a read-only reconciliation of the
assembled 87-outcome panel, its decision audit, the gated source outcomes, and
the isolated laboratory and TLS candidate exports. It verifies file hashes and
same-release provenance, checks the existing clinical/support dispositions,
and lists candidate support and definition issues. It **does not** freeze a
roster, modify parquets, or use held-out results to select outcomes.

All inputs must come from the same sealed first-line/outcome run. The final
panel decision audit must be rerun after syncing any changed registry or
decision rubric; the triage script detects old hashes. Candidate exports must
already exist (see `candidate_outcomes_v1_runbook.md` and
`candidate_tls_v1_runbook.md`). If either is missing, the triage report marks
the review blocked rather than silently omitting it.

From the repository root on the secure server, in the Python environment:

```bash
RUN_ROOT="$PWD/data/results/outcomes_final_medfix_20261005_v1"
PANEL_DIR="/absolute/path/to/assembled/final_panel"
PANEL_REVIEW_DIR="/absolute/path/to/current/panel_review"
LAB_CANDIDATES_DIR="/absolute/path/to/current/lab_candidate_export"
TLS_CANDIDATES_DIR="/absolute/path/to/current/tls_candidate_export"
TRIAGE_DIR="$PWD/data/results/final_outcome_triage_$(date -u +%Y%m%dT%H%M%SZ)"

python -m opera.run.review_final_outcome_inputs \
  --panel-dir "$PANEL_DIR" \
  --panel-review-dir "$PANEL_REVIEW_DIR" \
  --base-dir "$RUN_ROOT/opera_outcomes_smr_gated" \
  --lab-candidates-dir "$LAB_CANDIDATES_DIR" \
  --tls-candidates-dir "$TLS_CANDIDATES_DIR" \
  --output-dir "$TRIAGE_DIR"
```

The useful small handoff is `triage_INTERNAL.md` plus, when permitted,
`candidate_support_INTERNAL.csv` and `issues_INTERNAL.csv`. The Markdown file
lists every candidate with a pass/fail support flag but omits exact small
cell counts. If the decision rules changed after the panel audit, old panel
dispositions are suppressed until that audit is rerun. All outputs remain
INTERNAL and require the
study's disclosure approval before transfer outside the secure environment.
The script never reads or exports patient IDs into its reports. Candidate
`outcomes/*.parquet` and lab snapshots must stay on the secure server.

After review, create a separate versioned *approved* registry and matching
outcome directory, rebuild any affected composites, and hash the frozen
roster before generating training configurations. The current historical
registry and `panel_decisions_INTERNAL.csv` are not the approved roster.
