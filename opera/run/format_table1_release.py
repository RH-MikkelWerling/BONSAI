"""Reformat exact aggregate Table 1 exports without rereading patient histories."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from opera.functional.table1_release import release_table


SPLIT_COLUMNS = (("Overall", "Overall"), ("train", "Training"), ("tuning", "Tuning"),
                 ("held_out", "Test"))
SPLIT_PERIODS = ("All years", "Through 2021", "2022", "From 2023")


def _tex(value):
    text = "" if pd.isna(value) else str(value)
    for raw, escaped in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                         ("_", r"\_"), ("#", r"\#"), ("<", r"$<$"), (">", r"$>$"),
                         ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("±", r"$\pm$")):
        text = text.replace(raw, escaped)
    return text


SPECIALTY_PREFIX = "contact_specialty__"
# Manuscript layout: (section, [(variable, label, decimals)]). decimals applies to
# continuous rows (None keeps the exporter's display). Variables the export does
# not contain, or with no value anywhere, are left out.
LAYOUT = [
    ("Demographics", [
        ("age", "Age at treatment, years", 0),
        ("age_ge70", "Age $\\geq$70 years", None),
        ("sex", "Sex", None),
        ("index_year", "Year of treatment", "year"),
    ]),
    ("Disease", [
        ("cohort_grouped", "Disease group and subtype", None),
        ("cohort_source", "Subtype assignment", None),
        ("performance_status", "Performance status (ECOG)", None),
        ("risk_category", "Risk group by disease-specific index$^{\ddagger}$", None),
        ("nccn_ipi_risk", "NCCN-IPI risk group, DLBCL$^{\S}$", None),
        ("prognostic_score_by_system", "Prognostic index recorded at diagnosis", None),
        ("nccn_ipi_group", "NCCN-IPI risk group (DLBCL)", None),
        *[(name, "First-line treatment", None) for name in (
            "arm_primary", "treatment_group", "treatment_grouped", "treatment_category",
            "regimen_group", "regimen_category", "first_line_treatment_group")],
    ]),
    ("Pre-index history", [
        ("observable_history_years", "Recorded history, years", 1),
        ("recorded_events", "Recorded events", 0),
        ("distinct_diagnoses", "Distinct diagnoses", 0),
        ("distinct_medications", "Distinct medications", 0),
        ("laboratory_measurements", "Laboratory results", 0),
        ("hospital_encounters", "Days with hospital contact", 0),
        ("n_unique_specialties", "Distinct contact specialties", 0),
        ("days_since_last_preindex_event", "Days from last record to treatment", 0),
        ("predominant_region", "Region (most contacts)", None),
        (SPECIALTY_PREFIX, "Contact with specialty", None),
    ]),
    ("Follow-up", [
        ("followup_days", "Follow-up, years", 1),
        ("followup_reverse_km", "Reverse Kaplan--Meier follow-up, years", 1),
        ("death_event", "Died during follow-up", None),
    ]),
]
# Shown elsewhere or internal: never listed under "Other characteristics".
NOT_IN_TABLE = {"patients", "cohort_fine", "index", "ipi", "IPI", "nccn_ipi_score",
                "predominant_region_status", "female", "PS", "ps"}
# Shown only in the supplementary table by disease.
# Also kept out of Table 1: the subtype-assignment pass (an internal pipeline
# field) and the detailed first-line regimens (too many small categories for
# one page; a treatment-class summary belongs in a supplementary table).
TREATMENT_VARIABLES = ("arm_primary", "treatment_group", "treatment_grouped",
                       "treatment_category", "regimen_group", "regimen_category",
                       "first_line_treatment_group")
MAIN_TABLE_EXCLUDE = frozenset({SPECIALTY_PREFIX, "prognostic_score_by_system",
                                "nccn_ipi_group", "cohort_source", *TREATMENT_VARIABLES})
# Two-level characteristics shown as one row (the other level is the complement).
# The patient table codes sex as F/M; older exports wrote Female/Male.
SINGLE_LEVEL = {"sex": (("Female", "F"), "Female")}
CATEGORY_ORDER = ["low", "low-intermediate", "medium", "intermediate",
                  "high-intermediate", "high", "very high", "0–1", "2–4", "not recorded"]


# Danish region codes (Statistics Denmark) as recorded with each contact.
REGION_NAMES = {"1081": "North Denmark", "1082": "Central Denmark",
                "1083": "Southern Denmark", "1084": "Capital Region", "1085": "Zealand"}


def _number(value, decimals):
    """Half-up rounding (64.5 -> 65), thousands separators except for years."""
    from decimal import ROUND_HALF_UP, Decimal

    places = 0 if decimals == "year" else decimals
    rounded = float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP))
    return f"{rounded:.0f}" if decimals == "year" else f"{rounded:,.{places}f}"


def _display(row, decimals):
    if (decimals is not None and row.get("kind") == "continuous"
            and pd.notna(row.get("median")) and pd.notna(row.get("q1"))):
        return (f"{_number(row['median'], decimals)} "
                f"({_number(row['q1'], decimals)}–{_number(row['q3'], decimals)})")
    return row.get("display", "")


def _category_key(category):
    text = str(category)
    name = text.split(": ", 1)[-1].strip().lower()
    system = text.split(": ", 1)[0] if ": " in text else ""
    rank = CATEGORY_ORDER.index(name) if name in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (system, rank, text)


def _category_label(variable, category):
    text = str(category)
    if variable == "cohort_grouped":
        from opera.visualization.paper_taxonomy import disease_group_name
        return disease_group_name(text)
    if variable == "predominant_region":
        code = text.split("//", 1)[-1]
        return REGION_NAMES.get(code, code)
    if variable == "prognostic_score_by_system":
        system, _, level = text.partition(": ")
        return f"{system}, {level}"
    return text


def _disease_rows(result, cells, informative, width, label):
    """Disease groups in clinical order (bold), subtypes indented beneath them.

    Subtypes are listed by size and named in full with their abbreviation; a
    group with one subtype is a single row under the subtype's name. Subtypes
    without a prognostic index in routine use carry a dagger.
    """
    from opera.visualization.paper_taxonomy import (
        GROUP_ORDER, NO_ROUTINE_INDEX, disease_group_long_name, disease_in_group_name,
        disease_long_name, fine_cohort_groups)

    groups = result.loc[result.variable == "cohort_grouped"]
    fines = result.loc[result.variable == "cohort_fine"]
    if groups.empty and fines.empty:
        return []
    mapping = fine_cohort_groups()
    names = sorted(set(groups.category.dropna().astype(str))
                   | {mapping.get(str(f), "Other") for f in fines.category.dropna()},
                   key=lambda g: (GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER),
                                  disease_group_long_name(g)))
    overall = fines.loc[fines.split == "Overall"] if "split" in fines else fines
    size = (pd.to_numeric(overall.get("count"), errors="coerce")
            .groupby(overall.category.astype(str)).max() if "count" in overall else pd.Series(dtype=float))

    def named(fine, standalone=False):
        dagger = r"$^{\dagger}$" if fine in NO_ROUTINE_INDEX else ""
        name = disease_long_name(fine) if standalone else disease_in_group_name(fine)
        return _tex(name) + dagger

    lines = [r"\multicolumn{%d}{@{}l}{%s}\\" % (width, label)]
    for group in names:
        members = sorted((str(f) for f in fines.category.dropna().unique()
                          if mapping.get(str(f), "Other") == group),
                         key=lambda f: (-size.get(f, 0) if pd.notna(size.get(f, 0)) else 0,
                                        disease_long_name(f)))
        title = (named(members[0], standalone=True) if len(members) == 1
                 else _tex(disease_group_long_name(group)))
        values = cells(groups.loc[groups.category.astype(str) == group], None)
        lines.append(rf"\quad \textbf{{{title}}} & " + " & ".join(map(_tex, values)) + r" \\")
        if len(members) <= 1:
            continue  # a group with one subtype is already its own row
        for fine in members:
            values = cells(fines.loc[fines.category.astype(str) == fine], None)
            if informative(values):
                lines.append(rf"\qquad {named(fine)} & "
                             + " & ".join(map(_tex, values)) + r" \\")
    return lines


def _body(result, columns, key, exclude=()):
    """Sectioned table rows; ``key(rows, column, decimals)`` gives one cell.

    ``exclude`` drops layout entries (by variable) from this table."""
    width = len(columns) + 1
    lines = []

    def cells(rows, decimals):
        return [key(rows, column, decimals) for column in columns]

    def informative(values):
        return any(str(v) not in {"", "nan", "Not available"} for v in values)

    listed = {variable for _, entries in LAYOUT for variable, _, _ in entries}
    leftover = [v for v in dict.fromkeys(result.variable)
                if v not in listed and v not in NOT_IN_TABLE
                and not str(v).startswith(SPECIALTY_PREFIX)]
    layout = LAYOUT + ([("Other characteristics",
                         [(v, _tex(result.loc[result.variable == v, "characteristic"].iloc[0]), None)
                          for v in leftover])] if leftover else [])
    for section, entries in layout:
        block = []
        for variable, label, decimals in entries:
            if variable in exclude:
                continue
            if variable == "cohort_grouped":
                block += _disease_rows(result, cells, informative, width, label)
                continue
            if variable == SPECIALTY_PREFIX:
                names = sorted(v for v in result.variable.unique()
                               if str(v).startswith(SPECIALTY_PREFIX))
                rows_out = []
                for name in names:
                    values = cells(result.loc[result.variable == name], None)
                    if informative(values):
                        specialty = _tex(name[len(SPECIALTY_PREFIX):].replace("_", " "))
                        rows_out.append(rf"\quad {specialty} & " + " & ".join(map(_tex, values)) + r" \\")
                if rows_out:
                    block.append(r"\multicolumn{%d}{@{}l}{%s}\\" % (width, label))
                    block += rows_out
                continue
            rows = result.loc[result.variable == variable]
            if rows.empty:
                continue
            kind = rows.kind.iloc[0]
            suffix = ""  # "n (%) or median (IQR)" is stated once, in the caption
            if kind in {"continuous", "binary", "count"}:
                values = cells(rows, decimals)
                if informative(values):
                    block.append(f"{label}{suffix} & " + " & ".join(map(_tex, values)) + r" \\")
                continue
            if variable in SINGLE_LEVEL:
                levels, name = SINGLE_LEVEL[variable]
                values = cells(rows.loc[rows.category.astype(str).isin(levels)], None)
                if informative(values):
                    block.append(f"{name} & " + " & ".join(map(_tex, values)) + r" \\")
                    continue
            category_rows = []
            for category in sorted(rows.category.dropna().unique(), key=_category_key):
                values = cells(rows.loc[rows.category == category], None)
                if informative(values):
                    category_rows.append(rf"\quad {_tex(_category_label(variable, category))} & "
                                         + " & ".join(map(_tex, values)) + r" \\")
            if category_rows:
                block.append(r"\multicolumn{%d}{@{}l}{%s%s}\\" % (width, label, suffix))
                block += category_rows
        if block:
            lines += [r"\addlinespace", r"\multicolumn{%d}{@{}l}{\sffamily\bfseries %s}\\"
                      % (width, section), *block]
    return lines


def _note(policy):
    return (r"\par\vspace{1mm}{\scriptsize Counts of 1--4 are shown as $<$5. "
            r"Percentages use patients with a recorded value; history summaries use "
            r"records before the index date.}") if policy == "primary-only" else ""


def _disease_note():
    return (r"\par{\scriptsize $^{\dagger}$No dedicated, externally validated prognostic "
            r"index; the prespecified stratum of subtypes without an index "
            r"(Supplementary \cref{tab:index-availability}). "
            r"$^{\ddagger}$The low, intermediate or high category recorded at diagnosis "
            r"with the index of each disease's quality registry (e.g.\ IPI, FLIPI, "
            r"CLL-IPI, ISS), collapsed to three groups (e.g.\ CLL-IPI very high as high); "
            r"not recorded includes conflicting records. "
            r"$^{\S}$Computed from registry components at diagnosis (low 0--1, "
            r"intermediate 2--5, high $\geq$6); percentages of patients with DLBCL.}")


def pooled_table_latex(result, *, source_sha256, status="review", policy="targeted"):
    """Render the pooled Overall/train/tuning/held-out display as Table 1.

    Rows keep the exporter's order and its disclosure-processed display
    strings; nothing is recomputed. Multi-category characteristics get a
    heading with indented category rows.
    """
    pooled = result.loc[result.cohort_level == "all"].copy()
    columns = [split for split, _ in SPLIT_COLUMNS]

    def key(rows, split, decimals):
        match = rows.loc[rows.split == split]
        return _display(match.iloc[0], decimals) if len(match) else ""

    patients = pooled.loc[pooled.variable == "patients"].set_index("split").display

    lines = [
        f"% Generated by opera.run.format_table1_release from export sha256 {source_sha256}"
        f" (disclosure policy: {policy}).",
        "% Do not edit values by hand; regenerate from the reviewed export.",
        r"\begin{table*}[p]",
        r"\centering",
        # Written as the final table; review status lives in provenance, not the caption.
        r"\caption{\textbf{Baseline characteristics at first-line treatment.} "
        r"One record per patient at the shared first-line index, by temporal split. "
        r"Values are $n$ (\%) or median (IQR).}",
        r"\label{tab:characteristics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabularx}{\textwidth}{@{}Xrrrr@{}}",  # noqa
        r"\toprule",
        "Characteristic & " + " & ".join(label for _, label in SPLIT_COLUMNS) + r" \\",
        " & " + " & ".join(
            rf"{{\footnotesize\itshape {period}}}" for period in SPLIT_PERIODS) + r" \\",
        r"& " + " & ".join(rf"{{\footnotesize $n$ = {_tex(patients.get(c, ''))}}}"
                           for c in columns) + r" \\",
        r"\midrule",
        # Contact specialties and the categories of every recorded index are
        # listed in the by-disease supplementary table, so Table 1 fits a page.
        *_body(pooled, columns, key, exclude=MAIN_TABLE_EXCLUDE),
        r"\bottomrule",
        r"\end{tabularx}",
        _note(policy),
        _disease_note(),
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def disease_table_latex(result, *, source_sha256, policy="targeted"):
    """Supplementary Table 1 by disease group (all splits together)."""
    grouped = result.loc[(result.cohort_level == "cohort_grouped")
                         & (result.split == "Overall")].copy()
    try:
        from opera.visualization.paper_taxonomy import disease_group_name
    except ImportError:  # pragma: no cover
        disease_group_name = str
    groups = sorted(grouped.cohort.unique(), key=str)
    columns = ["All", *groups]
    overall = result.loc[(result.cohort_level == "all") & (result.split == "Overall")]
    frame = pd.concat([overall.assign(cohort="All"), grouped])
    frame = frame.loc[frame.variable != "cohort_grouped"]

    def key(rows, cohort, decimals):
        match = rows.loc[rows.cohort == cohort]
        return _display(match.iloc[0], decimals) if len(match) else ""

    patients = frame.loc[frame.variable == "patients"].set_index("cohort").display

    header = " & ".join(_tex(disease_group_name(g)) if g != "All" else "All" for g in columns)
    return "\n".join([
        f"% Generated by opera.run.format_table1_release from export sha256 {source_sha256}"
        f" (disclosure policy: {policy}).",
        r"\begin{sidewaystable}",
        r"\centering",
        r"\caption{\textbf{Baseline characteristics by disease group.} All splits together.}",
        r"\label{tab:characteristics-disease}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textheight}{!}{%",
        r"\begin{tabular}{@{}l" + "r" * len(columns) + "@{}}",
        r"\toprule",
        f"Characteristic & {header} \\\\",
        r"\midrule",
        r"Patients, $n$ & " + " & ".join(_tex(patients.get(c, "")) for c in columns) + r" \\",
        *_body(frame, columns, key),
        r"\bottomrule",
        r"\end{tabular}}",
        _note(policy),
        r"\end{sidewaystable}",
        "",
    ])


def layout_table_latex():
    """Table 1 as it will look, with every value pending: the layout preview.

    Built from the same layout and disease taxonomy as the real table, so the
    manuscript shows the final structure (all 24 subtypes under their groups)
    until the reviewed export replaces it.
    """
    from opera.visualization.paper_taxonomy import fine_cohort_groups

    rows = []
    splits = [split for split, _ in SPLIT_COLUMNS]

    def add(variable, characteristic, kind, categories=(None,)):
        for category in categories:
            for split in splits:
                rows.append({"variable": variable, "characteristic": characteristic,
                             "category": category, "kind": kind, "split": split,
                             "cohort_level": "all", "display": "·"})

    add("patients", "Patients", "count")
    add("age", "Age", "continuous")
    add("age_ge70", "Age >=70", "binary")
    add("sex", "Sex", "categorical", ["Female", "Male"])
    add("index_year", "Year of treatment", "continuous")
    mapping = fine_cohort_groups()
    add("cohort_grouped", "Disease group", "categorical", sorted(set(mapping.values())))
    add("cohort_fine", "Disease subtype", "categorical", sorted(mapping))
    add("prognostic_score_by_system", "Prognostic index", "categorical",
        ["IPI: low", "IPI: intermediate", "IPI: high"])
    for variable in ("observable_history_years", "recorded_events", "distinct_diagnoses",
                     "distinct_medications", "laboratory_measurements", "hospital_encounters"):
        add(variable, variable, "continuous")
    add("predominant_region", "Region", "categorical",
        [f"REGION//{r}" for r in ("Hovedstaden", "Sjælland", "Syddanmark", "Midtjylland",
                                   "Nordjylland")])
    add("followup_days", "Follow-up", "continuous")
    add("death_event", "Died", "binary")
    result = pd.DataFrame(rows)
    result["median"] = float("nan")
    tex = pooled_table_latex(result, source_sha256="layout (no data)", status="layout")
    return tex.replace(
        r"\caption{\textbf{Baseline characteristics at first-line treatment.} ",
        r"\caption{\textbf{Baseline characteristics at first-line treatment.} "
        r"Layout preview: every value (·) awaits the reviewed server export. ", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--disclosure-policy",
        choices=["primary-only", "targeted", "conservative"],
    )
    parser.add_argument("--suppression-threshold", type=int, default=5)
    parser.add_argument(
        "--latex",
        type=Path,
        help="Also write the manuscript table, e.g. paper/tables/table1_current_review.tex",
    )
    parser.add_argument(
        "--layout", type=Path,
        help="Write only the layout preview (no data) to this path and exit")
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Caption without the review notice; record reviewer/date separately",
    )
    args = parser.parse_args()
    if args.layout:
        args.layout.write_text(layout_table_latex(), encoding="utf-8")
        print(f"Wrote layout preview {args.layout}")
        return
    if not (args.input and args.output_dir and args.disclosure_policy):
        parser.error("--input, --output-dir and --disclosure-policy are required")
    table = pd.read_csv(args.input)
    # Older aggregate files may still contain the now-excluded treatment field.
    table = table.loc[table.variable != "arm_day1_primary"].copy()
    required = {"n_total", "n_observed", "n_missing", "count", "variable", "kind"}
    if not required <= set(table):
        raise ValueError(
            "Use the exact LOCAL_ONLY_internal/table1_values_long.csv, not an already suppressed export"
        )
    result, policy = release_table(
        table, args.suppression_threshold, args.disclosure_policy
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "table1_values_long.csv", index=False)
    pooled = result.loc[result.cohort_level == "all"].copy()
    pooled["row"] = pooled.characteristic + pooled.category.map(
        lambda value: f": {value}" if pd.notna(value) else ""
    )
    pooled.pivot(index=["variable", "row"], columns="split", values="display").reindex(
        columns=["Overall", "train", "tuning", "held_out"]
    ).to_csv(args.output_dir / "table1_pooled_display.csv")
    (args.output_dir / "disclosure_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    if args.latex:
        source_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(
            pooled_table_latex(
                result,
                source_sha256=source_sha256,
                status="approved" if args.approved else "review",
                policy=args.disclosure_policy,
            ),
            encoding="utf-8",
        )
        args.latex.with_name(args.latex.stem + "_by_disease.tex").write_text(
            disease_table_latex(result, source_sha256=source_sha256,
                                policy=args.disclosure_policy),
            encoding="utf-8",
        )
        print(f"Wrote manuscript table {args.latex} (export sha256 {source_sha256})")
    print(f"Saved {args.disclosure_policy} review tables to {args.output_dir}")
    if args.disclosure_policy == "primary-only":
        print(
            "primary-only: counts of 1-4 shown as <5, no complementary suppression. "
            "Totals and other cells may allow reconstruction; this needs your "
            "supervisor's / institution's explicit approval before leaving the server."
        )


if __name__ == "__main__":
    main()
