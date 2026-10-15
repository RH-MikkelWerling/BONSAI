"""Streaming raw MEDS summaries strictly before each patient's treatment index."""

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def summarize_meds_batches(population, batches, prefixes, lookback_days=None,
                           top_specialties=15):
    """Pre-index history per patient.

    Besides counts, ``hospital_encounters`` (optional prefix key) counts the
    distinct days with a contact, and, when specialties are configured, the
    ``top_specialties`` most common contact specialties become indicator
    columns ``contact_specialty__<name>`` (any pre-index contact).
    """
    required = {"distinct_medications", "distinct_diagnoses", "laboratory_measurements"}
    optional = {"n_unique_specialties", "predominant_region", "hospital_encounters"}
    if not required <= set(prefixes) or not set(prefixes) <= required | optional or any(
        not isinstance(v, list)
        or not v
        or any(not isinstance(p, str) or not p for p in v)
        for v in prefixes.values()
    ):
        raise ValueError(
            f"Provide explicit nonempty code-prefix lists for {sorted(required)}"
        )
    if lookback_days is not None and lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    index_dates = pd.to_datetime(
        population.set_index("subject_id").index_date, errors="raise"
    )
    if index_dates.isna().any() or not index_dates.index.is_unique:
        raise ValueError("MEDS summaries require unique IDs and nonmissing index dates")
    counts, labs, first, last = Counter(), Counter(), {}, {}
    distinct = {
        key: defaultdict(set) for key in required if key != "laboratory_measurements"
    }
    specialties = defaultdict(set)
    regions = defaultdict(Counter)
    contact_days = defaultdict(set)
    matched_context = Counter()
    for batch in batches:
        if not {"subject_id", "time", "code"} <= set(batch):
            raise ValueError("Raw MEDS shards require subject_id, time, code")
        batch = batch.loc[batch.subject_id.isin(index_dates.index)].copy()
        batch["time"] = pd.to_datetime(batch.time, errors="raise")
        batch["index_date"] = batch.subject_id.map(index_dates)
        valid = (
            batch.time.notna() & batch.code.notna() & batch.time.lt(batch.index_date)
        )
        codes = batch.code.astype("string")
        # DOB is timed at birth; it is not evidence of lifelong clinical coverage.
        demographic = codes.isin(["DOB", "DOD"]) | codes.str.startswith(
            ("SEX//", "BACKGROUND//"), na=False
        )
        valid &= ~demographic
        if lookback_days is not None:
            valid &= batch.time.ge(batch.index_date - pd.Timedelta(days=lookback_days))
        for row in batch.loc[valid].itertuples(index=False):
            sid, code, time = row.subject_id, str(row.code), row.time
            counts[sid] += 1
            first[sid] = min(first.get(sid, time), time)
            last[sid] = max(last.get(sid, time), time)
            for key, codes in prefixes.items():
                matched_prefix = next((prefix for prefix in codes if code.startswith(prefix)), None)
                if matched_prefix is not None:
                    if key == "laboratory_measurements":
                        labs[sid] += 1
                    elif key == "n_unique_specialties":
                        if code != matched_prefix:
                            specialties[sid].add(code)
                            matched_context[key] += 1
                    elif key == "hospital_encounters":
                        contact_days[sid].add(time.normalize())
                        matched_context[key] += 1
                    elif key == "predominant_region":
                        if code != matched_prefix:
                            regions[sid][code] += 1
                            matched_context[key] += 1
                    else:
                        distinct[key][sid].add(code)
    for key in optional & prefixes.keys():
        if matched_context[key] == 0:
            raise ValueError(f"No pre-index MEDS events matched configured {key} prefixes")
    specialty_patients = Counter(code for codes in specialties.values() for code in codes)
    common = [code for code, _ in specialty_patients.most_common(top_specialties)]
    rows = []
    for sid in index_dates.index:
        values = {"subject_id": sid}
        if sid in counts:
            values.update(
                recorded_events=counts[sid],
                observable_history_years=(index_dates[sid] - first[sid]).total_seconds()
                / (365.25 * 86400),
                days_since_last_preindex_event=(index_dates[sid] - last[sid]).total_seconds()
                / 86400,
                laboratory_measurements=labs[sid],
            )
            values.update({key: len(mapping[sid]) for key, mapping in distinct.items()})
        if "n_unique_specialties" in prefixes:
            values["n_unique_specialties"] = (
                len(specialties[sid]) if specialties[sid] else pd.NA
            )
        if "hospital_encounters" in prefixes and sid in counts:
            values["hospital_encounters"] = len(contact_days[sid])
        for code in common:
            values[f"contact_specialty__{code.split('//', 1)[-1]}"] = int(code in specialties[sid])
        if "predominant_region" in prefixes:
            if regions[sid]:
                top = max(regions[sid].values())
                modes = [code for code, n in regions[sid].items() if n == top]
                values["predominant_region"] = modes[0] if len(modes) == 1 else pd.NA
                values["predominant_region_status"] = (
                    "unique_mode" if len(modes) == 1 else "tied_mode"
                )
            else:
                values["predominant_region"] = pd.NA
                values["predominant_region_status"] = (
                    "region_missing" if sid in counts else "no_preindex_history"
                )
        rows.append(values)
    columns = ["subject_id", "recorded_events", "observable_history_years",
               "days_since_last_preindex_event", *sorted(required)]
    if "hospital_encounters" in prefixes:
        columns.append("hospital_encounters")
    if "n_unique_specialties" in prefixes:
        columns.append("n_unique_specialties")
        columns.extend(f"contact_specialty__{code.split('//', 1)[-1]}" for code in common)
    if "predominant_region" in prefixes:
        columns.extend(["predominant_region", "predominant_region_status"])
    return pd.DataFrame(rows).reindex(columns=columns)


def raw_meds_batches(root):
    import pyarrow.parquet as pq

    root = Path(root)
    data_root = root / "data" if (root / "data").is_dir() else root
    paths = sorted(
        path
        for split in ("train", "tuning", "held_out")
        for path in (data_root / split).rglob("*.parquet")
    )
    if not paths:
        raise ValueError(f"No raw MEDS split shards found under {data_root}")
    for path in paths:
        print(f"Summarizing raw MEDS: {path}", flush=True)
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=100000, columns=["subject_id", "time", "code"]
        ):
            yield batch.to_pandas()
