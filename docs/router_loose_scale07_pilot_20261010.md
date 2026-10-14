# Exploratory low-constraint router pilot

This is **one** bounded 30-epoch Time2Vec run, not a replacement for the clean
scale-0.3 model or a one-factor ablation. It uses the same approved data,
outcomes, DAPT initialization, seed, optimizer, objective, and checkpoint
selection as the clean candidate. Four router settings change together:

| Setting | Clean 0.3 | Stress pilot | Meaning |
| --- | ---: | ---: | --- |
| Adapter residual scale | 0.3 | 0.7 | Stronger adapter effect on the risk representation |
| Own-source floor | 0.1 each | 0.02 each | Five-fold weaker minimum own-source weight |
| Event-support shrinkage tau | 20 | 0 | No shrinkage for sources with observed training events |
| Shared-route initial logit | 3 | 0 | Neutral, rather than shared-favouring, initialization |

The small home floor is retained because home tasks train their own adapter
parameters; removing it altogether could starve initially near-uniform
adapters of gradient. Sources with zero training events remain disabled. The anchored gradient rule
still protects borrowed adapter parameters; this run does **not** turn that
off. The four changes are inseparable in this stress test, so any gain cannot
be attributed to a specific guardrail without follow-up one-factor runs.
The saved Hydra config records the exact values.

On a GPU allocation, after syncing this code:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
bash scripts/run_router_loose_scale07.sh preflight
bash scripts/run_router_loose_scale07.sh launch
```

Default run path:

```text
data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e
```

The wrapper checks the clean candidate and approved family outputs. It refuses
an existing run directory on a fresh launch. To resume after walltime, use
`bash scripts/run_router_loose_scale07.sh resume`; this requires `last.ckpt`
and refuses a run already marked complete. Supply a different
`OPERA_NEURAL_RUN_ROOT` only when intentionally starting a distinct run.

Run the script inside your usual GPU allocation, just like the other neural
candidates. It does not submit a scheduler job. If the allocation ends after a
valid checkpoint, start a new allocation and use the `resume` command above.

Review after completion: selected tuning epoch, endpoint-group macro CIF-AUC,
family/outcome distribution, Brier/calibration, route entropy and shared
weight, any negative transfer, and whether gates vary meaningfully across
patients/tasks. If performance improves but gates remain flat, the evidence
favours stronger adapters rather than selective routing. Keep held-out closed
until the architecture and comparison plan are frozen.
