# Fourier versus scaled-Time2Vec: pretraining-only review

Both runs were configured as autoregressive next-code prediction on the same
joined-bin MEDS release, with batch size 4, gradient accumulation 32, 20
epochs, no auxiliary scalar-value loss, the same `[SEP]` target exclusion, and
validation on the tail window. The intended sole intervention is
`model.abspos_encoding`. Confirm this in each *resolved* `hydra/config.yaml`
before interpreting the curves; a YAML in the repository is not proof of the
settings actually used by a run.

On the secure server, after syncing `bonsai/run/compare_pretraining_runs.py`:

```bash
export FOURIER_RUN=/absolute/path/to/fourier/run_id_...  # or version_0
export TIME2VEC_RUN=/absolute/path/to/scaled_time2vec/run_id_...
python -m bonsai.run.compare_pretraining_runs \
  --fourier "$FOURIER_RUN" --time2vec "$TIME2VEC_RUN"
```

This reads only `version_0/metrics.csv` and prints aggregate per-epoch
validation losses and top-1 precision. It does not inspect patients or copy
checkpoints. Share the printed summary, not the run directory. Also verify
both runs have `epoch_10.ckpt`, `best.ckpt` and `last.ckpt` (the milestone's
exact filename can be checked with `ls`), with the same train/tuning/vocabulary
paths and population manifest in resolved configs. Preserve both checkpoint
series even if one pretraining loss is lower.

For a full-length checkpoint audit, run `opera.run.diagnose_vocabulary_learning`
separately on each `best.ckpt`, with the same `subject_data_tuning.pt`,
`vocabulary.pt` and seed. Set `--max-len 8192 --max-loss-batches 0` to evaluate
next-code loss and accuracy across every tuning batch at the trained sequence
length. Keep `--batch-size 1 --logit-chunk-size 32` for memory headroom. The
contextual perturbation stage materializes per-patient and sampled event
representations in memory, so retain `--max-context-batches 32`; this is an
explicitly sampled *representation* audit, not a full-cohort one. Use the same
cap in both runs. The exact-neighbour audit remains frequency-stratified and
capped (`--max-neighbor-tokens`) because its work is quadratic in token count.
The diagnostic `summary.json` records these limits. Its output includes
patient IDs and token-level samples; keep it in protected storage.

Evidence of learning: materially improving validation loss and Precision@1
over early epochs, stable optimization, and non-corrupt checkpoints. Evidence
for choosing a clock: a persistent matched-epoch validation advantage, checked
against the precise target/validation contract. This is still only a
representation-learning signal. The decisive comparison is matched fine-tuning
on identical final cohort, split, outcome masks, horizons, training budget and
selection metric; never choose by held-out results. Differences in checkpoint
size, learned clock parameters, or raw loss alone do not establish clinical
utility. The pretraining log has no explicit random/untrained baseline, so
positive learning claims should be phrased as improvement across epochs, not
as superiority to a measured random model.

## Observed 20-epoch runs (4 October 2026; internal)

Both runs validated through epoch 20 and improved markedly in next-code loss
and top-1 accuracy. The full-length checkpoint diagnostic, run on the same
subject-data path for both checkpoints, favored scaled-Time2Vec for both
same-timestamp and new-timestamp targets. Same-timestamp targets comprised
86.5% of the audited targets, so their aggregate dominance must be kept
separate from the more difficult new-timestamp question. A five-year uniform
calendar shift changed Fourier's last-token representation more than
scaled-Time2Vec's; this is sensitivity, not evidence of superior clinical
prediction or calibration. Both responded strongly to reversing event order.

The checkpoint diagnostic's absolute CE/gap does not reproduce the logged
validation loss/gap. Resolved Hydra configs now confirm that both runs used
the same tuning `.pt`, population CSV, seed 42, observed numeric-value policy,
`[SEP]` target exclusion, same-time-target policy, and an exclusive
1 January 2022 cutoff; only `model.abspos_encoding` differs. The diagnostic
currently loads the tuning `.pt` directly and does not apply the population
CSV filter or cutoff. Its reported CE and target-transition results are
therefore **not** an exact audit of the pre-2022 pretraining objective; do not
use their larger Time2Vec gap to choose a checkpoint. The logged validation
curves are the matched pretraining comparison. A corrected, lightweight
pre-2022 target-transition audit is optional; prioritize data closeout and
matched downstream tuning first.
