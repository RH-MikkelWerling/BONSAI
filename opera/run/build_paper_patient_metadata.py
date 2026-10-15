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


def history_summary(subjects, index_abspos, vocabulary, national_vocabulary=None):
    """Pre-index clinical token count and national-vocabulary coverage.

    ``subjects`` iterates BONSAI subject dicts (``subject_id``, ``code``,
    ``abspos``); ``index_abspos`` maps subject to the index time on the same
    scale. Special and background tokens are not history.
    """
    names = {int(i): str(name) for name, i in vocabulary.items()}
    clinical = np.array([not names.get(i, "").startswith(SPECIAL_PREFIXES)
                         for i in range(max(names) + 1)])
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
        row = {"subject_id": subject_id, "history_event_count": int(len(history))}
        if covered is not None:
            row["national_vocabulary_coverage"] = (
                float(covered[history].mean()) if len(history) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_patient_metadata(population, history, *, score_columns=("index", "IPI"),
                           index_date_column="index_date"):
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
    for column in ("cohort_fine", "cohort_grouped", "split", *score_columns):
        if column in population:
            metadata[column] = population[column].to_numpy()
    metadata = metadata.merge(history, on="subject_id", how="left", validate="one_to_one")
    if metadata.subject_id.duplicated().any():
        raise ValueError("Patient metadata must have one row per patient")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True,
                        help="Cohort membership with subject_id, index date, age/birth "
                             "date, sex, cohort_fine and the index/IPI columns.")
    parser.add_argument("--clinical-metadata", type=Path,
                        help="Optional separate table with subject_id, index and IPI.")
    parser.add_argument("--subject-data-dir", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--national-vocabulary", type=Path)
    parser.add_argument("--index-date-column", default="index_date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from opera.functional.outcomes import _pandas_abspos
    from opera.modules.datamodules.OutcomeFinetuneDataModule import load_subject_pool

    population = _read_population(args.population)
    if args.clinical_metadata:
        clinical = _read_population(args.clinical_metadata)
        keep = ["subject_id", *[c for c in ("index", "IPI") if c in clinical]]
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
                                      index_date_column=args.index_date_column)
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
