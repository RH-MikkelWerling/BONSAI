"""Pre-index characteristics of every patient, for subgroup and score analyses.

One row per patient with only information available at the start of
first-line treatment: disease, age, sex, calendar year, the amount of recorded
history, the share of that history the national encoder's vocabulary covers,
and the prognostic index recorded for the patient (``index``) with its risk
category (``IPI``). History is counted from the same tokenised records the
models read, truncated strictly before the index date exactly as in
fine-tuning, so the subgroups describe what the models saw.

    python -m opera.run.build_paper_patient_metadata --population <membership> \
        --subject-data-dir <sequence dir> --vocabulary <vocabulary.pt> \
        --national-vocabulary <external vocabulary.pt> --output <metadata.parquet>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.functional.tabular_features import _age_at_index, _read_population, _sex

SPECIAL_PREFIXES = ("[", "BG_")
REGION_PREFIX = "REGION//"


def history_summary(subjects, index_abspos, vocabulary, national_vocabulary=None):
    """Pre-index clinical token count, national-vocabulary coverage and region.

    ``predominant_region`` is the region of most pre-index contacts (missing
    when there are none or the most frequent regions tie).

    ``subjects`` iterates BONSAI subject dicts (``subject_id``, ``code``,
    ``abspos``); ``index_abspos`` maps subject to the index time on the same
    scale. Special and background tokens are not history.
    """
    names = {int(i): str(name) for name, i in vocabulary.items()}
    clinical = np.array([not names.get(i, "").startswith(SPECIAL_PREFIXES)
                         for i in range(max(names) + 1)])
    labs = np.array([int(i) for name, i in vocabulary.items()
                     if str(name).startswith("LABTEST//")], dtype=int)
    regions = {int(i): str(name)[len(REGION_PREFIX):] for name, i in vocabulary.items()
               if str(name).startswith(REGION_PREFIX)}
    # Vectorised lookups: one array index per token instead of Python loops.
    is_lab = np.zeros(max(names) + 1, bool)
    is_lab[labs] = True
    region_names = sorted(set(regions.values()))
    region_of = np.full(max(names) + 1, -1, int)
    for code, name in regions.items():
        region_of[code] = region_names.index(name)
    covered = None
    if national_vocabulary is not None:
        known = set(map(str, national_vocabulary))
        covered = np.array([names.get(i, "") in known for i in range(max(names) + 1)])
    rows = []
    for subject in subjects:
        subject_id = subject["subject_id"]
        if subject_id not in index_abspos:
            continue
        abspos = np.asarray(subject["abspos"], float)
        codes = np.asarray(subject["code"], int)
        before = codes[: np.searchsorted(abspos, index_abspos[subject_id], side="left")]
        history = before[clinical[before]]
        row = {"subject_id": subject_id, "history_event_count": int(len(history)),
               # Laboratory results before treatment (coverage differs by region
               # and period, which may explain regional differences).
               "laboratory_event_count": int(is_lab[history].sum())}
        if regions:
            found = region_of[before]
            counts = np.bincount(found[found >= 0], minlength=len(region_names))
            order = np.argsort(-counts, kind="stable")
            top = counts[order[0]] if len(counts) else 0
            # The most frequent region; missing when there is none or a tie.
            row["predominant_region"] = (
                region_names[order[0]]
                if top > 0 and (len(counts) == 1 or top > counts[order[1]]) else None)
        if covered is not None:
            row["national_vocabulary_coverage"] = (
                float(covered[history].mean()) if len(history) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_patient_metadata(population, history, *, score_columns=("index", "IPI"),
                           index_date_column="index_date", identity_date_column=None):
    index_date = pd.to_datetime(population[index_date_column], errors="coerce")
    age, _ = _age_at_index(population)
    sex, _ = _sex(population)
    metadata = pd.DataFrame({
        "subject_id": population.subject_id,
        "index_date": index_date,
        "index_year": index_date.dt.year,
        "age_at_index": age,
        "sex": sex,
    })
    for column in ("cohort_fine", "cohort_grouped", "split", "cohort_source",
                   *score_columns):
        if column in population:
            metadata[column] = population[column].to_numpy()
    if "cohort_source" in metadata:
        # Pass 1 assigns the subtype from diagnoses within 14 days of treatment
        # start; the 90-day reassessment and treatment-arm rescue rely on later
        # or indirect evidence and form the late-assignment sensitivity group.
        metadata["cohort_identity_post_index"] = (
            metadata.cohort_source.astype(str) != "pass1").to_numpy()
    if identity_date_column:
        # Date of the diagnosis that assigned the fine subtype; the analysis
        # repeats every contrast without patients whose evidence is post-index.
        metadata["cohort_identity_evidence_date"] = pd.to_datetime(
            population[identity_date_column], errors="coerce").to_numpy()
    metadata = metadata.merge(history, on="subject_id", how="left", validate="one_to_one")
    if metadata.subject_id.duplicated().any():
        raise ValueError("Patient metadata must have one row per patient")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True,
                        help="Cohort membership with subject_id, index date, age/birth "
                             "date, sex, cohort_fine and the index/IPI columns.")
    parser.add_argument("--demographics", type=Path,
                        help="subject_id, sex and birth_date, when the membership file "
                             "lacks them (the Table 1 demographics export).")
    parser.add_argument("--clinical-metadata", type=Path,
                        help="Optional separate table with subject_id, index and IPI.")
    parser.add_argument("--subject-data-dir", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--national-vocabulary", type=Path)
    parser.add_argument("--index-date-column", default="index_date")
    parser.add_argument("--identity-date-column",
                        help="Column with the date of the diagnosis that assigned the subtype.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from opera.functional.outcomes import _pandas_abspos
    from opera.modules.datamodules.OutcomeFinetuneDataModule import load_subject_pool

    population = _read_population(args.population, args.demographics)
    if args.clinical_metadata:
        clinical = _read_population(args.clinical_metadata)
        # The prognostic index and, when the membership file lacks them,
        # demographics may come from this table.
        # Every column of the clinical table is carried (registry IPI,
        # NCCN-IPI score and group, demographics, cohort_source, ...).
        keep = list(clinical.columns)
        population = population.drop(columns=[c for c in keep[1:] if c in population])
        population = population.merge(clinical[keep], on="subject_id", how="left",
                                      validate="one_to_one")
    index_abspos = dict(zip(
        population.subject_id,
        _pandas_abspos(pd.to_datetime(population[args.index_date_column]))))
    shards = sorted(args.subject_data_dir.glob("subject_data_*.pt"))
    vocabulary = torch.load(args.vocabulary, map_location="cpu", weights_only=False)
    national = (torch.load(args.national_vocabulary, map_location="cpu", weights_only=False)
                if args.national_vocabulary else None)
    history = history_summary(load_subject_pool(map(str, shards)), index_abspos,
                              vocabulary, national)
    metadata = build_patient_metadata(population, history,
                                      index_date_column=args.index_date_column,
                                      identity_date_column=args.identity_date_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(args.output, index=False)
    print(json.dumps({
        "output": str(args.output), "patients": len(metadata),
        "with_history": int(metadata.history_event_count.notna().sum()),
        "with_index": int(metadata["index"].notna().sum()) if "index" in metadata else 0,
        "national_coverage": bool(args.national_vocabulary),
    }))


if __name__ == "__main__":
    main()
