"""Print the values the lock closure note needs, from a tuning analysis.

Paired tuning AUC differences pooled over all horizons (every cell weighing
equally), with 95% CI, from <analysis root>/prediction_analysis. Aggregates
only. Anything the analysis did not compute is listed as missing.

    .venv-v100/bin/python scripts/lock_numbers.py data/paper_256_v3/analysis/tuning_1102
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SECTIONS = {
    "Seed spread of J_OA_OWN (s42 minus each other seed)": {
        "OAOWN_NF": "s42 - s43", "OAOWN_NF44": "s42 - s44", "OAOWN_NF45": "s42 - s45",
        "OAOWN4_S42": "s42 - its four-seed mean"},
    "Noise floors": {"NF_AB": "R_AB s43 - s42", "J_NF": "J s42 - s43"},
    "Selection: candidates against R_AB (base rule)": {
        "OAOWN_AB": "J_OA_OWN - R_AB", "OAOWNPH_AB": "J_OA_OWN_PH - R_AB",
        "OALROWN_AB": "J_OA_LR_OWN - R_AB", "FPFPHOWN_AB": "J_FPF_PH_OWN_PH - R_AB",
        "XSW_AB": "J_OA_OWN_X_SW - R_AB", "PEXSW_AB": "J_OA_OWN_PE_X_SW - R_AB",
        "MAX_AB": "J_OA_OWN_MAX - R_AB", "PF5E_AB": "R_PF5E - R_AB"},
    "Parsimony: reading variants against J_OA_OWN": {
        "OAOWNPH_OAOWN": "+ disease-group corrections", "OALROWN_OAOWN": "tied queries",
        "SW20_OAOWN": "shrinkage tau 20", "SW100_OAOWN": "shrinkage tau 100",
        "X_OAOWN": "diagnosis x outcome", "XSW_OAOWN": "cross + shrinkage",
        "PE_OAOWN": "endpoint family", "PHX_OAOWN": "disease group + cross",
        "PEXSW_OAOWN": "family + cross + shrinkage", "MAX_OAOWN": "everything, rank 32",
        "OAOWN_R32": "rank 32 (sensitivity)"},
    "Check: J_OA_OWN beats R_AB on every shared seed": {
        "OAOWN_AB": "seed 42", "OAOWN_J_s43": "(seed 43: OPERA - J)",
        "AB_OFF_s43": "(seed 43: R_AB - own corrections)", "OAOWN_OFF_s43": "(seed 43: OPERA - own corrections)"},
    "Runner-up J_OA_OWN_PH on more seeds (stage 16)": {
        "OAOWNPH_OAOWN": "PH - OA_OWN (s42)", "OAOWNPH_OAOWN_s43": "PH - OA_OWN (s43)",
        "OAOWNPH_OAOWN_s44": "PH - OA_OWN (s44)", "OAOWNPH_AB": "PH - R_AB (s42)",
        "OAOWNPH_AB_s43": "PH - R_AB (s43)", "OAOWNPH_NF": "PH s42 - s43"},
    "Ladder (Table 2), seed 42 and seed 43": {
        "OFF_J": "own corrections - J", "OAOWN_OFF": "+ own reading", "O2": "OPERA - J (alias)",
        "OAOWN_J": "J_OA_OWN - J", "AB_OFF": "routing instead",
        "OFF_J_s43": "own corrections - J (s43)", "OAOWN_OFF_s43": "+ own reading (s43)",
        "AB_OFF_s43": "routing instead (s43)"},
    "Ablations": {"OAOWN_DIS": "disease correction (OPERA - without)",
                  "OAOWN_OUT": "outcome correction (OPERA - without)"},
    "Seed averages": {"OAOWN4_J": "4-seed OPERA - J", "OAOWN4_OFF": "- own corrections",
                      "OAOWN4_AB": "- R_AB", "OAOWN4_X": "- XGBoost"},
    "Encoder-rate rule (national arm at 1.0x only if both > +0.002)": {
        "ELR_J": "J_ELR10 - J", "ELR_AB": "R_AB_ELR10 - R_AB", "ELR3_AB": "R_AB_ELR3 - R_AB",
        "ELR_OAOWN": "J_OA_OWN_ELR10 - J_OA_OWN"},
}


def main(root: Path) -> None:
    summaries = pd.read_csv(root / "prediction_analysis" / "effect_summaries.csv", low_memory=False)
    pooled = summaries.loc[(summaries.split == "tuning") & (summaries.metric == "auc")
                           & (summaries.stratum == "all_cells")
                           & (summaries.horizon_days.astype(str) == "0")].set_index("contrast")
    missing = []
    for title, rows in SECTIONS.items():
        print(f"\n## {title}")
        for code, label in rows.items():
            if code not in pooled.index:
                missing.append(code)
                print(f"  {code:15s} {label:40s} missing")
                continue
            row = pooled.loc[code]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            print(f"  {code:15s} {label:40s} {row.gain:+.4f}  ({row.lower:+.4f} to {row.upper:+.4f})"
                  f"  cells {int(row.n_cells)}")
    present = sorted(c for c in pooled.index if c.startswith("DS_") and c.endswith("__AB"))
    if present:
        print("\n## Design search against R_AB (every design)")
        for code in present:
            row = pooled.loc[code]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            print(f"  {code[3:-4]:24s} {row.gain:+.4f}  ({row.lower:+.4f} to {row.upper:+.4f})")
    if missing:
        print(f"\nMissing ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
