"""Every number the manuscript reports, from the locked aggregate outputs.

Writes ``results_macros.tex`` (one ``\\result`` key per reported number, read by
the manuscript), ``panel_effects_rows.tex`` (the body of Table 2) and
``paper_numbers.json`` (the same values with their sources, for review). Only
aggregate outputs are read, never patient rows, so the files can leave the
secure environment after the usual disclosure review. A number whose source is
missing is simply absent, and the manuscript keeps its placeholder there.

    python -m opera.run.build_paper_numbers --analysis-root <analysis root> \\
        --cell-metadata <cell_metadata.csv> --output-dir <paper numbers dir>

Keys (``<code>`` is a contrast code such as P5, T2 or E3):
  <code>_auc, <code>_ipa   difference pooled over every horizon, with its 95% CI
                           (<code>_auc_1y etc.: the same at one year)
  <code>_better            "k of n" cells in which the first model had the higher AUC
                           (cells pooled over their horizons where the analysis made them)
  <code>_median            median cell difference in AUC with its IQR
  <code>_auc_rare, <code>_auc_noindex
                           the same within subtypes with <500 training patients and
                           within the subtypes without a prognostic index
  cells_evaluable          disease-outcome cells with a one-year joint-model AUC
  evaluable_specialist_tasks  specialist tasks with finite evaluation AUC
  narrow_fits              requires a separate fit inventory; never inferred
                           from placeholder metric rows
  cells_low_support        cells with fewer than 20 training events at one year
  dp90_sens_joint / _xgb, dp90_mcc_joint / _xgb
                           median sensitivity and MCC at 90% tuning specificity
  ipi_extra_per1000        extra events flagged per 1,000 at the IPI's specificity
  ipi_low_extra_per1000    extra patients placed at low risk per 1,000
  ipi_low_risk_model / _index, ipi_high_risk_index, top5_risk_model
                           observed two-year risks with 95% Wilson intervals
  followup_ipi             median follow-up (IQR) of the IPI population, years
  clinical_<m>_atleast, clinical_<m>_better, clinical_<m>_notestimable, clinical_<m>_median
                           every clinical index (``opera.evaluation.clinical_index_summary``),
                           <m> = low_size, high_size (equal group sizes), low_risk,
                           high_risk (equal risk) and both_size (both ends, equal size);
                           "k of n" over every index compared (clinical_indices_total)

Also writes ``clinical_index_rows.tex`` (body of the supplementary table
``paper/tables/clinical_index_summary.tex``) and ``clinical_index_summary.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from opera.evaluation.secondary_analysis import ALL_HORIZONS
from opera.run.analyse_locked_predictions import file_hash, read_table

SPLIT = "held_out"
HORIZON = 365
CODES = ("P5", "P1", "P2", "P3", "P4", "P6", "P0", "P7", "T2", "T3", "T4", "T6", "T7", "T9",
         "T10", "R0", "R1", "E1", "E2", "E3", "E4", "N10", "N25", "N50",
         "LS10", "LS25", "LS50", "LSE10", "LSE25", "LSE50", "FD10", "FD25", "FD50", "RS1",
         "J_vs_best_tuning", "O1", "O2", "O2_MATCHED", "O2_LOCAL", "O_ROUTING_MATCHED", "O_READ_MATCHED", "O_CORRECTIONS_MATCHED", "O3", "T5_router", "T5_shared", "T5E_router", "T5E_shared", "PP0", "PP1", "PP2", "PP3", "PP4", "PT5_AB", "PT3V_AB", "PT5_J", "PT3V_J", "PT5_FZ", "FZ_AB", "PT3V_PT5", "NF_AB", "PF5E_AB", "PF5E_J", "PF5E_FZ", "PF5E_OWN", "PFOFF_RBHO", "PFOFF_AB",
         "PFR32_OWN", "PFOFF32_OFF", "PHG_OFF", "PHGE_OFF", "PHGEP_PHGE", "PFLIVE_OFF",
         "F10_OFF_J", "F10_PF5E_OFF", "F10_PHGE_OFF", "F10_LIVE_OFF", "F10_PF5E_J",
         "PL8_OFF", "PL8_PHGE", "F10_PL8_OFF", "F10_PL8_PHGE",
         "OA_J", "SP_J", "FP_J", "OA_AB", "SP_AB", "FP_AB", "FP_OA", "FPF_J", "FPF_FP", "OAOWN_J", "OFF_J", "OAOWN_OFF", "AB_OFF",
         "OAOWN_AB", "OAOWN_OA", "OAOWNPH_OAOWN", "OAOWNPH_AB", "FPFOWN_AB", "FPFOWN_FPF", "FPFPH_FPF", "FPFPHOWN_AB", "FPFPHOWN_FPFOWN", "OALR_OA", "OALROWN_OAOWN", "OALROWN_AB", "ELR_J", "ELR_AB", "ELR3_AB", "OAOWN_NF", "OAOWN_R32", "OAOWN_DIS", "OAOWN_OUT", "F10_OAOWN_OFF", "F10_OAOWN_PHGE",
         "SW20_OAOWN", "SW100_OAOWN", "X_OAOWN", "XSW_OAOWN", "PE_OAOWN", "PHX_OAOWN", "PEXSW_OAOWN",
         "XSW_AB", "PEXSW_AB", "MAX_OAOWN", "MAX_AB", "OAOWN_NF44", "OAOWN_NF45", "OAOWN_NF4345", "J_NF", "OAOWN_J_s43", "OFF_J_s43", "OAOWN_OFF_s43", "AB_OFF_s43", "OAOWN4_J", "OAOWN4_OFF", "OAOWN4_AB", "OAOWN4_X", "OAOWN4_S42", "E2_AB", "E4_AB", "A2_AB", "T5E_AB", "T5_AB", "I1_AB", "I3_AB", "OAOWN_AB_E", "OAOWN_AB_EA", "OFF_J_E", "OAOWN_OFF_E", "AB_OFF_E", "OFF_J_EA", "OAOWN_OFF_EA", "AB_OFF_EA", "OAOWN_DIS_EA", "OAOWN_OUT_EA", "OFF_GROWTH_EA", "READ_GROWTH_EA", "OAOWNPH_AB_s43", "OAOWNPH_OAOWN_s43", "OAOWNPH_NF", "OAOWNPH_OAOWN_s44", "ELR_FPFPHOWN", "ELR_OAOWN", "ELR_FPFPHOWN_AB", "ELR_OAOWN_AB",
         *(f"{code}_{suffix}" for code in ("PF5", "PF5B", "PF10E", "PF5EP", "PF5EL", "PF5ED", "PF5EO", "PFLR",
                                           "PFW0", "PFR32", "PHG", "PHGE", "PHGEP", "PF5E43", "PFLIVE", "PL8")
           for suffix in ("AB", "E")), "R1_orig", "T1_orig", "R_orig", "IPW_R", "IPW_J",
         *(f"router_{part}_{design}" for part in ("own", "borrow", "both")
           for design in ("O", "AB")))
# Table 2, grouped by question; fixed T contrasts retain joint identities.
# O1 is the explicitly selected OPERA-versus-XGBoost comparison.
TABLE_ROWS = (
    (None, "Borrowing across tasks (each against one model per task)"),
    ("P1", "Patients with other diseases"), ("P2", "Other outcomes in the same patients"),
    ("P5", "Both"),
    (None, "Specialising each prediction (each step added to the one above)"),
    ("OFF_J", "Corrections for the patient's disease and the outcome"),
    ("OAOWN_OFF", "\\quad + each outcome reading the record its own way"),
    ("O2_MATCHED", "\\quad = all specialisation (same encoder)"),
    ("O2", "Final OPERA against the local joint model (total system)"),
    ("O2_LOCAL", "Selected local design against the local joint model"),
    ("O_ROUTING_MATCHED", "Final OPERA against learned routing (same encoder)"),
    ("AB_OFF", "Instead: learning which other tasks to borrow from"),
    ("R1", "Learned routing (R$_{AB}$) against the joint model"),
    (None, "Pretraining on the cohort's records (joint model)"),
    ("P0", "Pretrained encoder, against no pretraining"),
    ("P7", "Fine-tuning the encoder, against training only the readout"),
    (None, "Pretraining on the national population (against the cohort encoder)"),
    ("E1", "Joint model, national encoder"),
    ("E3", "Joint model, national encoder adapted to the cohort"),
    ("E2", "OPERA, national encoder"),
    ("E4", "OPERA, national encoder adapted to the cohort"),
    (None, "Sequence models against tabular models (per-task XGBoost unless stated)"),
    ("T3", "One model per task (specialist)"), ("T7", "XGBoost pooled across diseases"),
    ("O1", "OPERA, cohort encoder"),
    ("T5E_router", "OPERA, national encoder"),
    ("T5_router", "OPERA, national encoder adapted to the cohort"),
    ("T2", "Joint model"),
    ("T5E_shared", "Joint model, national encoder"),
    ("T5_shared", "Joint model, national encoder adapted to the cohort"),
    ("T6", "Joint model, against XGBoost pooled across diseases"),
    ("T9", "Joint model, against XGBoost with registry variables"),
)


def fmt(value, digits=3):
    """Number for LaTeX text, with a proper minus sign."""
    return f"{value:.{digits}f}".replace("-", "$-$")


def with_ci(value, low, high, digits=3, label="95\\% CI"):
    if not np.isfinite([value, low, high]).all():
        return None
    return f"{fmt(value, digits)} ({label} {fmt(low, digits)} to {fmt(high, digits)})"


def percent_ci(events, n):
    """Observed proportion with a 95% Wilson interval, as LaTeX text."""
    if not n:
        return None
    p, z = events / n, 1.959964
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * p:.0f}\\% (95\\% CI {100 * (centre - half):.0f}--{100 * (centre + half):.0f})"


def _read(path, sources):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        frame = read_table(path)
    except pd.errors.EmptyDataError:
        return None
    sources[str(path)] = file_hash(path)
    return frame


def contrast_numbers(summaries, contrasts, *, horizon=ALL_HORIZONS, suffix="",
                     pooled_cells=None):
    """Contrast macros pooled over all horizons (default) or at one horizon.

    ``suffix`` is appended to every key, e.g. ``_1y`` for the one-year values
    that the main figures show.
    """
    numbers = {}
    if summaries is not None:
        part = summaries.loc[(summaries.split == SPLIT) & (summaries.horizon_days == horizon)]
        if "stratum" in part:
            # Prespecified strata: rare subtypes (<500 training patients) and the
            # subtypes without a prognostic index in routine use.
            for stratum, label in (("subtype_size", "rare"), ("no_index_subtype", "noindex")):
                rows = part.loc[part.stratum == stratum]
                if stratum == "subtype_size":
                    rows = rows.loc[rows.stratum_value.astype(str).str.startswith("(-1")]
                else:
                    rows = rows.loc[rows.stratum_value.astype(str) == "no_index"]
                for row in rows.itertuples():
                    if row.contrast in CODES and row.metric in ("auc", "ipa"):
                        text = with_ci(row.gain, row.lower, row.upper)
                        if text:
                            numbers[f"{row.contrast}_{row.metric}_{label}{suffix}"] = text
                        numbers.setdefault(f"{row.contrast}_cells_{label}{suffix}",
                                           f"{int(row.n_cells):,}")
            part = part.loc[part.stratum == "all_cells"]
        for row in part.itertuples():
            if row.contrast in CODES and row.metric in ("auc", "ipa"):
                text = with_ci(row.gain, row.lower, row.upper)
                if text:
                    numbers[f"{row.contrast}_{row.metric}{suffix}"] = text
                numbers.setdefault(f"{row.contrast}_cells{suffix}", f"{int(row.n_cells):,}")
    if horizon == ALL_HORIZONS and pooled_cells is not None:
        contrasts = pooled_cells  # one row per cell, pooled over its horizons
    if contrasts is not None:
        part = contrasts.loc[(contrasts.split == SPLIT) & (contrasts.metric == "auc")]
        if horizon != ALL_HORIZONS:
            part = part.loc[part.horizon_days == horizon]
        for code, rows in part.groupby("contrast"):
            gains = rows.gain.dropna()
            if code not in CODES or not len(gains):
                continue
            numbers[f"{code}_better{suffix}"] = f"{int((gains > 0).sum()):,} of {len(gains):,}"
            q1, median, q3 = gains.quantile([0.25, 0.5, 0.75])
            numbers[f"{code}_median{suffix}"] = with_ci(median, q1, q3, label="IQR")
    return numbers


def coverage_numbers(estimates, cells):
    numbers = {}
    if estimates is not None:
        joint = estimates.loc[(estimates.split == SPLIT) & (estimates.horizon_days == HORIZON)
                              & (estimates.model == "J_L") & (estimates.metric == "auc")]
        numbers["cells_evaluable"] = f"{int(joint.estimate.notna().sum()):,}"
        # Evaluation exports include placeholder rows for untrained tasks.
        # Even finite metrics count evaluable tasks, not the number of fits.
        evaluated = estimates.loc[(estimates.split == SPLIT) & (estimates.metric == "auc")
                                  & estimates.estimate.notna()]
        specialists = evaluated.loc[evaluated.model == "S_L",
                                    ["cohort", "outcome"]].drop_duplicates()
        if len(specialists):
            numbers["evaluable_specialist_tasks"] = f"{len(specialists):,}"

    if cells is not None and "n_events_train" in cells:
        year = cells.loc[cells.horizon_days == HORIZON].drop_duplicates(["cohort", "outcome"])
        numbers["cells_low_support"] = f"{int((year.n_events_train < 20).sum()):,}"
    return numbers


def decision_numbers(points):
    numbers = {}
    if points is None:
        return numbers
    part = points.loc[(points.horizon_days == HORIZON) & (points.status == "estimated")
                      & (points.rule == "tuning_specificity:0.90")]
    for model, name in (("OPERA", "opera"), ("J_L", "joint"), ("X", "xgb")):
        rows = part.loc[part.model == model]
        if len(rows):
            numbers[f"dp90_sens_{name}"] = fmt(rows.sensitivity.median(), 2)
            numbers[f"dp90_mcc_{name}"] = fmt(rows.mcc.median(), 2)
    return numbers


def clinical_numbers(headline, strata, inventory, *, size_matched=None, km_matched=None):
    numbers = {}
    required = {'manuscript', 'score_id', 'threshold_split', 'model'}
    if headline is None or not required <= set(headline):
        return numbers
    main = headline.loc[(headline.manuscript.astype(str).str.lower() == "true")
                        & headline.threshold_split.eq('train')]
    if not len(main):
        return numbers
    if len(main) != 1:
        raise ValueError('Choose exactly one primary clinical score for manuscript numbers')
    row = main.iloc[0]
    numbers["ipi_extra_per1000"] = f"{row.extra_events_flagged_per_1000:.0f}"
    if row.get('grouping_rule') == 'cohort_size_matched':
        # Equal-size strata are not evidence about expansion at matched risk.
        # Use the independent KM-matched report for those legacy prose slots.
        if km_matched is not None and len(km_matched):
            selected = km_matched.loc[km_matched.score_id.eq(row.score_id) & km_matched.status.eq('estimable')]
            if len(selected):
                match = selected.iloc[0]
                numbers['ipi_low_extra_per1000'] = f"{match.additional_low_per_1000:.0f}"
                for system in ('model', 'index'):
                    point, lower, upper = [match.get(system + '_survival' + suffix, np.nan) for suffix in ('', '_lower', '_upper')]
                    if np.isfinite([point, lower, upper]).all():
                        numbers['ipi_low_risk_' + system] = (
                            f"{100 * (1-point):.1f}\\% (95\\% CI {100 * (1-upper):.1f}--{100 * (1-lower):.1f})")
        if size_matched is not None and len(size_matched):
            selected = size_matched.loc[size_matched.score_id.eq(row.score_id)]
            for match in selected.itertuples():
                key = 'clinical_same_size_' + str(match.group).lower().replace('-', '_').replace(' ', '_')
                numbers[key] = with_ci(match.difference, match.lower, match.upper)
        return numbers
    numbers["ipi_low_extra_per1000"] = f"{row.model_low_per_1000 - row.index_low_per_1000:.0f}"
    if strata is not None and inventory is not None:
        ids = inventory.loc[inventory.get("status", pd.Series(dtype=str)) == "rendered", "score_id"]
        candidates = strata.loc[strata.score_id.isin(ids) & strata.score_id.eq(row.score_id)]
        # Match explicit identities, never guess from coincidentally equal shares.
        for score_id, rows in candidates.groupby("score_id"):
            groups = [g for g in rows.group.unique() if not str(g).startswith("top")]
            low, high = groups[0], groups[-1]
            index_low = rows.loc[(rows.system == "index") & (rows.group == low)].iloc[0]
            if not np.isclose(1000 * index_low.share, row.index_low_per_1000):
                continue
            pick = {"ipi_low_risk_index": ("index", low), "ipi_high_risk_index": ("index", high),
                    "ipi_low_risk_model": ("model", low), "top5_risk_model": ("model", "top 5%")}
            for key, (system, group) in pick.items():
                found = rows.loc[(rows.system == system) & (rows.group == group)]
                if len(found):
                    record = found.iloc[0]
                    numbers[key] = percent_ci(round(record.observed_risk * record.n), record.n)
            follow = inventory.loc[inventory.score_id == score_id]
            if len(follow) and "median_followup_days" in follow:
                f = follow.iloc[0]
                if np.isfinite(f.median_followup_days):
                    numbers["followup_ipi"] = (
                        f"{f.median_followup_days / 365.25:.1f} years (IQR "
                        f"{f.followup_q25_days / 365.25:.1f} to {f.followup_q75_days / 365.25:.1f})")
            break
    return numbers


CLINICAL_INDEX_DIGITS = {"low_size": 1, "high_size": 1, "low_risk": 1, "high_risk": 0}


def _tex(text):
    """Plain registry text for a LaTeX cell."""
    return (str(text).replace("\\", "").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("≥", "$\\geq$")
            .replace("–", "--").replace("…", "\\dots{}"))


def clinical_index_cell(row, measure):
    """Estimate (95% CI) for one comparison: bold when the interval lies above
    zero (favouring the model), in the warning colour when below, as Table 2."""
    digits = CLINICAL_INDEX_DIGITS[measure]
    point, low, high = (row.get(measure), row.get(f"{measure}_lower"),
                        row.get(f"{measure}_upper"))
    if point is None or not np.isfinite(point):
        return "n.e."
    if low is None or high is None or not np.isfinite([low, high]).all():
        return fmt(point, digits)
    text = f"{fmt(point, digits)} ({fmt(low, digits)} to {fmt(high, digits)})"
    if low > 0:
        return f"\\textbf{{{text}}}"
    if high < 0:
        return f"\\textcolor{{OperaSienna}}{{{text}}}"
    return text


def clinical_index_rows(summary):
    """Body rows of the supplementary table of every clinical index (a–d)."""
    if summary is None or not len(summary):
        return None
    lines, previous = [], None
    for row in summary.to_dict("records"):
        disease = _tex(row["disease"]) if row["disease"] != previous else ""
        if previous is not None and row["disease"] != previous:
            lines.append("\\addlinespace[2pt]")
        previous = row["disease"]
        horizon = row.get("horizon_days")
        endpoint = _tex(row.get("endpoint") or "--")
        if horizon is not None and np.isfinite(horizon):
            years = horizon / 365
            endpoint += f", {years:g} y" if horizon % 365 == 0 else f", {horizon:g} d"
        n = row.get("n_patients")
        patients = f"{int(n):,}" if n is not None and np.isfinite(n) else "--"
        lead = [disease, _tex(row["index"]), endpoint, patients]
        if row["status"] != "rendered":
            reason = _tex(row.get("reason") or "not estimable")
            lines.append(" & ".join(lead) + f" & \\multicolumn{{5}}{{l}}{{\\textit{{Not estimable: {reason}}}}} \\\\")
            continue
        count = row.get("low_risk_count")
        count = (f"{int(count):+,}".replace("-", "$-$") if count is not None and np.isfinite(count)
                 else "--")
        cells = [clinical_index_cell(row, "low_size"), clinical_index_cell(row, "high_size"),
                 clinical_index_cell(row, "low_risk"), count, clinical_index_cell(row, "high_risk")]
        lines.append(" & ".join(lead + cells) + " \\\\")
    # The closing rule travels with the rows: a table cannot end with \IfFileExists.
    return "\n".join(lines) + "\n\\bottomrule\n"


def clinical_index_numbers(summary):
    """Across-index macros: how many indices are at least as good at each end.

    Denominators are every index compared, including those not estimable (the
    lock reports every index); ``*_notestimable`` gives those separately.
    """
    from opera.evaluation.clinical_index_summary import MEASURES, across_index_counts
    numbers = {}
    if summary is None or not len(summary):
        return numbers
    counts = across_index_counts(summary)
    total = len(summary)
    numbers["clinical_indices_total"] = f"{total:,}"
    numbers["clinical_indices_not_estimable"] = f"{int(summary.status.ne('rendered').sum()):,}"
    keys = {"low_size": "low_size", "high_size": "high_size", "low_risk": "low_risk",
            "high_risk": "high_risk", "both_ends_size": "both_size"}
    for measure, key in keys.items():
        tally = counts[measure]
        numbers[f"clinical_{key}_atleast"] = f"{tally['at_least_as_good']:,} of {total:,}"
        numbers[f"clinical_{key}_better"] = f"{tally['better']:,} of {total:,}"
        numbers[f"clinical_{key}_notestimable"] = f"{tally['not_estimable']:,}"
    for measure in MEASURES:
        values = summary[measure].dropna()
        if len(values):
            digits = CLINICAL_INDEX_DIGITS[measure]
            numbers[f"clinical_{measure}_median"] = (
                f"{fmt(values.median(), digits)} (range {fmt(values.min(), digits)} "
                f"to {fmt(values.max(), digits)})")
    return numbers


def attribution_numbers(path, sources):
    """Integrated-gradients scalars from ``attribution_numbers.json`` (optional)."""
    path = Path(path)
    if not path.is_file():
        return {}
    sources[str(path)] = "attribution summary"
    raw = json.loads(path.read_text())["numbers"]
    out = {}
    for key, value in raw.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if key.startswith(("ig_share_", "ig_var_")):
            out[key] = f"{100 * value:.0f}\\%"
        elif key.startswith(("ig_rho_", "ig_similarity_")):
            out[key] = f"{value:.2f}"
        elif key.startswith("ig_gap_") or key == "ig_completeness_p95":
            out[key] = f"{value * (100 if key == 'ig_completeness_p95' else 1):.1f}"
        else:
            out[key] = f"{int(value):,}"
    return out


RESTRICTION_KEYS = {"Only the timing of care": "restrict_timeline",
                    "Without laboratory results": "restrict_no_labs",
                    "Without medication": "restrict_no_medication",
                    "Without diagnoses": "restrict_no_diagnoses",
                    "Without region and specialty": "restrict_no_region",
                    "Only the last year": "restrict_last_year",
                    **{f"Last {d} days hidden": f"restrict_gap_{d}d" for d in (7, 14, 21, 28)}}


def restriction_numbers(path, sources):
    """AUC change (all outcomes) per input restriction, with its interval (optional)."""
    path = Path(path)
    if not path.is_file():
        return {}
    sources[str(path)] = "input restriction summary"
    table = pd.read_csv(path)
    out = {}
    for row in table.loc[table.group == "All outcomes"].itertuples():
        key = RESTRICTION_KEYS.get(row.restriction)
        if key:
            out[key] = with_ci(row.mean_delta, row.lower, row.upper)
    return out


def table_rows(panel):
    """Body rows of Table 2 from the plot-table ``panel_effects.csv``."""
    if panel is None:
        return None

    def cell(row, metric):
        """Difference with its interval: bold when the interval favours the first
        model, in the warning colour when it favours the second, plain otherwise."""
        gain, interval = row.get(f"{metric}_gain"), row.get(f"{metric}_interval")
        if gain is None or pd.isna(gain):
            return "--"
        if isinstance(interval, str) and " to " in interval:
            low, high = (float(x) for x in interval.split(" to "))
            text = f"{fmt(gain)} ({fmt(low)} to {fmt(high)})"
            if low > 0:
                return f"\\textbf{{{text}}}"
            if high < 0:
                return f"\\textcolor{{OperaSienna}}{{{text}}}"
            return text
        return fmt(gain)

    lines = []
    for code, label in TABLE_ROWS:
        if code is None:
            # \rlap, not \multicolumn: the rows are read inside \IfFileExists.
            lines.append(f"\\rlap{{\\textit{{{label}}}}} & & & & & & \\\\")
            continue
        for variant in ("NA", "N", "L"):
            label = label.replace("OPERA-" + variant, r"\opera{} ("
                                  + {"L": "local", "N": "national", "NA": "national + adaptation"}[variant] + ")")
        label = label.replace('OPERA', r'\opera{}')
        label = "\\quad " + label
        found = panel.loc[panel.contrast == code]
        if not len(found):
            lines.append(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        row = found.iloc[0]
        better = row.get("cells_better")
        verdict = row.get("label")
        lines.append(" & ".join([
            label, f"{int(row.cells):,}", cell(row, "auc"), cell(row, "ipa"), cell(row, "rare_auc"),
            str(better).replace("%", "\\%") if isinstance(better, str) else "--",
            verdict if isinstance(verdict, str) else "--"]) + " \\\\")
    # The closing rule travels with the rows: a table cannot end with \IfFileExists.
    return "\n".join(lines) + "\n\\bottomrule\n"


def build(analysis_root, cell_metadata, output_dir, *, plot_tables=None, clinical_scores=None,
          attribution_summary=None):
    if attribution_summary is not None and not (Path(attribution_summary) / "attribution_numbers.json").is_file():
        raise ValueError('Explicit attribution summary lacks attribution_numbers.json')
    analysis_root = Path(analysis_root)
    analysis = analysis_root / "prediction_analysis"
    plot_tables = Path(plot_tables) if plot_tables else analysis_root / "figure_inputs" / "plot_tables"
    clinical = Path(clinical_scores) if clinical_scores else analysis_root / "clinical_scores"
    sources = {}
    numbers = {}
    effects = _read(analysis / "effect_summaries.csv", sources)
    contrasts = _read(analysis / "contrasts.csv", sources)
    pooled_cells = analysis / "contrasts_all_horizons.csv"
    pooled_cells = _read(pooled_cells, sources) if pooled_cells.exists() else None
    # Headline numbers pool all six horizons (the primary metric is AUC at each
    # horizon); the one-year values shown in the main figures carry ``_1y``.
    numbers.update(contrast_numbers(effects, contrasts, pooled_cells=pooled_cells))
    numbers.update(contrast_numbers(effects, contrasts, horizon=HORIZON, suffix="_1y"))
    balanced = _read(analysis / "effect_summaries_disease_balanced.csv", sources)
    numbers.update(contrast_numbers(balanced, None, suffix="_disease_balanced"))
    numbers.update(contrast_numbers(balanced, None, horizon=HORIZON,
                                    suffix="_1y_disease_balanced"))
    numbers.update(coverage_numbers(_read(analysis / "estimates.csv", sources),
                                    _read(cell_metadata, sources) if cell_metadata else None))
    numbers.update(decision_numbers(_read(analysis / "operating_points.csv", sources)))
    numbers.update(clinical_numbers(_read(clinical / "headline_scores.csv", sources),
                                    _read(clinical / "stratification_diagnostics.csv", sources),
                                    _read(clinical / "score_inventory.csv", sources),
                                    size_matched=_read(clinical / 'size_matched_survival_comparisons.csv', sources),
                                    km_matched=_read(clinical / 'km_matched_low_groups.csv', sources)))
    evaluation_km = _read(clinical / "evaluation_km_matched_groups.csv", sources)
    if evaluation_km is not None:
        for row in evaluation_km.to_dict("records"):
            if row.get("split", SPLIT) != SPLIT:
                continue
            key = "clinical_evaluation_km_" + str(row["score_id"]) + "_" + row["end"]
            numbers[key + "_index_n"] = fmt(row["index_n"], 0)
            if row["status"] == "estimable":
                numbers[key + "_model_n"] = fmt(row["model_n"], 0)
                numbers[key + "_additional_per_1000"] = with_ci(
                    row["additional_per_1000"], row["lower"], row["upper"], digits=1)
    # Every clinical index at both ends (supplementary table and its macros).
    clinical_summary = None
    inventory = _read(clinical / "score_inventory.csv", sources)
    if inventory is not None:
        from opera.evaluation.clinical_index_summary import summarise_clinical_indices
        clinical_summary = summarise_clinical_indices(
            inventory, _read(clinical / "size_matched_survival_comparisons.csv", sources),
            _read(clinical / "km_matched_low_groups.csv", sources),
            _read(clinical / "clinical_operating_point_counts.csv", sources),
            _read(clinical / "headline_scores.csv", sources))
        numbers.update(clinical_index_numbers(clinical_summary))
    numbers.update(restriction_numbers(
        analysis_root / "input_restrictions" / "input_restriction_summary.csv", sources))
    numbers.update(attribution_numbers(
        (Path(attribution_summary) if attribution_summary else
         analysis_root / "attributions" / "summary_365d") / "attribution_numbers.json", sources))
    numbers = {key: value for key, value in numbers.items() if value}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    macros = ["% Generated by opera.run.build_paper_numbers; do not edit.", r"\makeatletter"]
    macros += [rf"\@namedef{{opera@res@{key}}}{{{value}}}" for key, value in sorted(numbers.items())]
    macros.append(r"\makeatother")
    (output_dir / "results_macros.tex").write_text("\n".join(macros) + "\n")
    rows = table_rows(_read(plot_tables / "panel_effects.csv", sources))
    if rows:
        (output_dir / "panel_effects_rows.tex").write_text(rows)
    if evaluation_km is not None:
        evaluation_km.to_csv(output_dir / "clinical_evaluation_km_matching.csv", index=False)
    clinical_rows = clinical_index_rows(clinical_summary)
    if clinical_rows:
        (output_dir / "clinical_index_rows.tex").write_text(clinical_rows)
        clinical_summary.to_csv(output_dir / "clinical_index_summary.csv", index=False)
    (output_dir / "paper_numbers.json").write_text(json.dumps(
        {"split": SPLIT, "horizon_days": HORIZON, "numbers": numbers, "sources": sources},
        indent=2, ensure_ascii=False) + "\n")
    return numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--cell-metadata", type=Path)
    parser.add_argument("--plot-tables", type=Path)
    parser.add_argument("--clinical-scores", type=Path)
    parser.add_argument("--attribution-summary", type=Path,
                        help="Explicit fresh summary directory, e.g. corrected IG summary_365d")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("held_out", "tuning"), default="held_out",
                        help="tuning: preview numbers before the test set is opened")
    args = parser.parse_args()
    global SPLIT
    SPLIT = args.split
    numbers = build(args.analysis_root, args.cell_metadata, args.output_dir,
                    plot_tables=args.plot_tables, clinical_scores=args.clinical_scores,
                    attribution_summary=args.attribution_summary)
    print(json.dumps({"numbers": len(numbers), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
