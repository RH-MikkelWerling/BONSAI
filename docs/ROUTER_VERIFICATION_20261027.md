# Verify tuning router inference

Sync the current source, then run from the server repo root inside an allocated GPU node, using the same Python environment as the rehearsal:

```bash
python -m opera.run.verify_router_predictions \
  --env-file .env.rehearsal \
  --device cuda \
  --output-dir "data/paper_256_v3/analysis/router_verification_$(date +%Y%m%d_%H%M%S)"
```

This command runs no training and does not overwrite existing predictions, analyses, or figures. It refuses an existing output directory. Keep the node allocation alive until it finishes; nohup does not extend scheduler walltime.

The command checks the selected J_L and R_L tuning artifacts against their frozen bundle configurations, checkpoint selection markers and file hashes. It recomputes native and paper metrics from the full saved tuning predictions, compares joint/router prediction keys, follow-up and absolute CIF differences, then replays two patients per fine disease through the production evaluator with batch size one. The replay must match all saved horizons within atol=1e-5 and rtol=1e-4. Selection uses IDs and disease only, without looking at labels or gains.

For the routed checkpoint, the replay also exports encoder-only and zero-borrowing predictions. Encoder-only means that checkpoint's encoder and head, not the separately trained joint model. Zero-borrowing removes borrowed contributions at unchanged routing weights while preserving own and global corrections. This is a diagnostic intervention, not the primary mean-replacement source-removal analysis or a performance estimate.

Return only `report.json`. Keep `selected_subjects_INTERNAL.json`, the replay parquets and detailed logs on the server. If a subprocess fails, inspect J_L.log or R_L.log there; report.json retains completed checks.

`replay_passed` means both small replays reproduced saved predictions. Metric reconciliation and provenance still need inspection: missing estimates, status mismatches and material numeric errors need explanation. A correct replay cannot establish the scientific value of borrowing, and a median cell gain near zero can coexist with a positive supported, disease-balanced mean. The known missing bootstrap intervals remain a separate analysis issue; these checks do not manufacture confidence intervals or establish their validity.
