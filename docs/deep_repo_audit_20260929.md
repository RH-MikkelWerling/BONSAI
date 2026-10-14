# Repository audit — 2026-09-29

Scope: BONSAI tokenization, censoring, finetuning and sampling; OPERA prospective
split selection, competing-risk training/evaluation, outcome-transfer provenance,
experiment generation and launch configuration. Existing uncommitted work was
preserved. This is a local CPU audit, not a certification of the server data.

## Confirmed issues corrected

| Severity | Issue | Correction |
| --- | --- | --- |
| High | Prediction sequences included events exactly at the index timestamp. | Exclusive index censoring is now the default; regression test added. |
| High when sampling is enabled | Per-task sampler weights were formed before physical/patient filtering and could be misaligned with the dataset. | Build sampler after filtering in BONSAI and OPERA finetuning; OPERA order test added. |
| Medium | OPERA finetuning used the first patient's background length for every patient. | Infer background tokens per subject. |
| Medium | Effective-number sampling weights were nearly uniform. | Use inverse effective class size. |
| Medium | Per-task classification metrics received logits rather than probabilities. | Apply sigmoid for metrics, retaining logits for BCE. |
| Medium | Vocabulary IDs depended on unordered set iteration. | Sort newly encountered codes before assigning IDs. |
| Medium | The hospitalisation transfer evidence hash failed on CRLF checkouts. | Verify the archival LF-normalized bytes on every platform. |
| Low | The Fourier encoding lacked its documented thousand-hour input adapter. | Add an explicit input-unit option without altering existing model defaults or checkpoint tensors. |
| Low | Five tests had stale assumptions or exposed the above defects. | Updated or fixed, then rerun. |

## Verification and limitations

- Repository-wide Ruff and Python compilation pass.
- The non-slow, non-integration pytest suite passes locally, including focused
  competing-risk, censoring, model-readout and split tests. A new sampler-order
  test also passes after the full-suite run.
- Real GPU execution, final MEDS files, server environment and PHAIR checkpoint
  compatibility were not available locally. Run server preflight and a bounded
  training/evaluation smoke test on the final versioned dataset before treating
  it as the publication ground truth.
- The exclusive index boundary changes input sequences relative to older runs.
  Do not compare post-fix results directly with older checkpoints without a
  validation rerun or explicit sensitivity analysis.
- The current R-derived OPERA endpoint pipeline is separate from upstream
  BONSAI's outcome-creation overhaul; importing that overhaul blindly would
  change label definitions. Existing source and dataset provenance should be
  verified when the final MEDS release arrives.
- FlashAttention rotary convention changes from upstream are intentionally not
  applied to existing checkpoints without a compatibility/migration test.
