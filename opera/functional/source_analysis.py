"""
Data source analysis for OPERA.

Workflow context:
  The base model is pretrained by collaborators on a broad population
  using LPR3 + shared registries (labs, etc.).  You then receive this
  model and continue pretraining (DAPT) on your 75k hematology cohort.

  Your cohort's token composition:
    - LPR3 tokens: SHARED with base model (LPR data mapped to LPR3 prefix)
    - Cancer registry: NOT in base vocab (small, specialized)
    - Pathology: NOT in base vocab (small, specialized)
    - RKKP: NOT in base vocab (optional expansion)
    - Labs: SHARED if collaborators included them

  When vocabulary expansion is used, patients have a mix of pretrained
  tokens (LPR3, labs) and newly-initialised tokens (cancer, pathology,
  RKKP).  This module diagnoses whether the model relies on the source
  composition of a patient's tokens vs their clinical content.

The source of each token is inferred from namespace prefixes:
  - "LPR3//D501" → source "LPR3"
  - "cancer//tumor_stage_III" → source "cancer"
  - "pathology//M9680" → source "pathology"
  - "RKKP//ann_arbor_III" → source "RKKP"
  - "BACKGROUND//sex_male" → ignored (present for all patients)
  - "[CLS]", "[MASK]" → ignored (special tokens)
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
import numpy as np
import torch
import logging


def infer_patient_sources(
    subject: Dict[str, torch.Tensor],
    vocabulary_inv: Dict[int, str],
    background_prefixes: Set[str] = None,
    special_token_prefixes: Set[str] = None,
) -> Set[str]:
    """
    Determine which data sources contributed tokens to this patient.

    Parameters
    ----------
    subject : dict
        Standard BONSAI subject dict with "code" key.
    vocabulary_inv : dict
        Inverted vocabulary: token_id → token_string.
    background_prefixes : set
        Prefixes to ignore (present in all sources). Default: {"BACKGROUND"}.
    special_token_prefixes : set
        Prefixes for special tokens. Default: tokens starting with "[".

    Returns
    -------
    set of str
        Source prefixes found in this patient's tokens (e.g. {"LPR", "RKKP"}).
    """
    if background_prefixes is None:
        background_prefixes = {"BACKGROUND"}
    if special_token_prefixes is None:
        special_token_prefixes = {"["}

    sources = set()
    for token_id in subject["code"].tolist():
        token_str = vocabulary_inv.get(token_id, "")

        # Skip special tokens
        if any(token_str.startswith(sp) for sp in special_token_prefixes):
            continue

        # Extract prefix before "//"
        if "//" in token_str:
            prefix = token_str.split("//")[0]
            if prefix not in background_prefixes:
                sources.add(prefix)

    return sources


def invert_vocabulary(vocabulary: Dict[str, int]) -> Dict[int, str]:
    """Invert a vocabulary dict: token_string → token_id  →  token_id → token_string."""
    return {v: k for k, v in vocabulary.items()}


def assign_source_labels(
    subjects: list,
    vocabulary: Dict[str, int],
    background_prefixes: Optional[Set[str]] = None,
) -> np.ndarray:
    """
    Assign a source label string to each subject.

    Returns
    -------
    source_labels : np.ndarray of str, shape (N,)
        One of: a single prefix (e.g. "LPR"), "multi_source", or "unknown".
    """
    vocab_inv = invert_vocabulary(vocabulary)
    labels = []

    for subject in subjects:
        sources = infer_patient_sources(
            subject, vocab_inv, background_prefixes=background_prefixes
        )
        if len(sources) == 0:
            labels.append("unknown")
        elif len(sources) == 1:
            labels.append(sources.pop())
        else:
            labels.append("multi_source")

    return np.array(labels)


def compute_source_statistics(
    source_labels: np.ndarray,
    outcome_labels: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """
    Compute summary statistics about data source distribution.

    Parameters
    ----------
    source_labels : (N,) str array
    outcome_labels : dict, optional
        outcome_name → (N,) int array.  If provided, computes prevalence
        per source × outcome.

    Returns
    -------
    dict with keys:
        "source_counts": dict[source, count]
        "source_fractions": dict[source, fraction]
        "source_outcome_prevalence": dict[source, dict[outcome, prevalence]]  (if outcomes provided)
    """
    counter = Counter(source_labels)
    n = len(source_labels)

    stats = {
        "source_counts": dict(counter),
        "source_fractions": {k: v / n for k, v in counter.items()},
    }

    if outcome_labels is not None:
        prevalences = {}
        for source in counter:
            mask = source_labels == source
            prevalences[source] = {}
            for outcome_name, labs in outcome_labels.items():
                valid = (labs >= 0) & mask
                if valid.sum() > 0:
                    prevalences[source][outcome_name] = float(labs[valid].mean())
                else:
                    prevalences[source][outcome_name] = None
        stats["source_outcome_prevalence"] = prevalences

    return stats


def compute_source_separation_score(
    embeddings: np.ndarray,
    source_labels: np.ndarray,
    max_samples: int = 5000,
    seed: int = 42,
) -> float:
    """
    Compute silhouette score of embeddings grouped by data source.

    A high score means the embedding space separates by source — this is
    a WARNING sign that the model might be encoding "which registry did
    this patient come from" rather than clinically meaningful features.

    Returns
    -------
    float
        Silhouette score in [-1, 1].  Values near 0 are good (no source
        separation).  Values near 1 are bad (strong source separation).
        Returns 0.0 if fewer than 2 sources are present.
    """
    from sklearn.metrics import silhouette_score

    unique_sources = np.unique(source_labels)
    if len(unique_sources) < 2:
        return 0.0

    # Subsample if too large
    n = len(embeddings)
    if n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_samples, replace=False)
        embeddings = embeddings[idx]
        source_labels = source_labels[idx]

    # Encode sources as integers
    source_to_int = {s: i for i, s in enumerate(sorted(set(source_labels)))}
    int_labels = np.array([source_to_int[s] for s in source_labels])

    return silhouette_score(embeddings, int_labels, metric="cosine")


def format_source_report(
    source_stats: Dict,
    source_separation: Optional[float] = None,
) -> str:
    """Pretty-print source analysis."""
    lines = []
    lines.append("=" * 60)
    lines.append("DATA SOURCE ANALYSIS")
    lines.append("=" * 60)

    lines.append("\n── Source distribution ──")
    for source, count in sorted(source_stats["source_counts"].items()):
        frac = source_stats["source_fractions"][source]
        lines.append(f"  {source:20s}  n={count:6d}  ({frac:5.1%})")

    if "source_outcome_prevalence" in source_stats:
        lines.append("\n── Outcome prevalence by source ──")
        prev = source_stats["source_outcome_prevalence"]
        # Header
        outcomes = sorted(set(
            o for source_prev in prev.values() for o in source_prev
        ))
        header = f"  {'Source':20s}" + "".join(f"  {o:>15s}" for o in outcomes)
        lines.append(header)
        for source in sorted(prev):
            row = f"  {source:20s}"
            for o in outcomes:
                val = prev[source].get(o)
                if val is not None:
                    row += f"  {val:15.3f}"
                else:
                    row += f"  {'N/A':>15s}"
            lines.append(row)

    if source_separation is not None:
        lines.append(f"\n── Source separation in embedding space ──")
        lines.append(f"  Silhouette score: {source_separation:.4f}")
        if source_separation > 0.3:
            lines.append(
                "  ⚠ WARNING: High source separation. The model may be encoding "
                "data source identity rather than clinical features."
            )
        elif source_separation > 0.1:
            lines.append(
                "  ⚡ Moderate source separation. Worth investigating with "
                "embedding plots colored by source."
            )
        else:
            lines.append(
                "  ✓ Low source separation. Embedding space is not strongly "
                "partitioned by data source."
            )

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Per-source token profiling
# ═════════════════════════════════════════════════════════════════════

def compute_token_source_profile(
    subjects: list,
    vocabulary: Dict[str, int],
    source_prefixes: Optional[List[str]] = None,
    ignore_prefixes: Optional[Set[str]] = None,
) -> Dict:
    """
    For each patient, count how many tokens come from each data source.

    This is the key diagnostic for the OPERA data landscape:
      - Your 75k cohort patients: tokens from LPR, LPR3, cancer, pathology, labs, (RKKP)
      - Collaborator patients: tokens from LPR3, (labs), nothing else

    Parameters
    ----------
    subjects : list of dicts
    vocabulary : dict (token_str → token_id)
    source_prefixes : list of str, optional
        Prefixes to count. Default: common BONSAI/OPERA prefixes.
        Examples: ["LPR", "LPR3", "cancer", "pathology", "LAB", "RKKP"]
    ignore_prefixes : set of str, optional
        Prefixes to skip. Default: {"BACKGROUND", "["} (background + special tokens).

    Returns
    -------
    dict with:
        "per_patient": DataFrame with columns [subject_id, <prefix>_count, ..., total_tokens]
        "source_presence": dict[prefix → (N,) bool array] — whether each patient has tokens from this source
        "summary": dict with aggregate stats per source
    """
    import pandas as pd

    if source_prefixes is None:
        # Auto-detect from vocabulary
        prefixes_found = set()
        for token in vocabulary:
            if "//" in token:
                prefix = token.split("//")[0]
                prefixes_found.add(prefix)
        # Remove background/special
        prefixes_found -= {"BACKGROUND"}
        source_prefixes = sorted(prefixes_found)

    if ignore_prefixes is None:
        ignore_prefixes = {"BACKGROUND", "["}

    vocab_inv = invert_vocabulary(vocabulary)

    records = []
    for subject in subjects:
        counts = {p: 0 for p in source_prefixes}
        total = 0
        for tid in subject["code"].tolist():
            token = vocab_inv.get(tid, "")
            if any(token.startswith(ip) for ip in ignore_prefixes):
                continue
            total += 1
            if "//" in token:
                prefix = token.split("//")[0]
                if prefix in counts:
                    counts[prefix] += 1
        counts["total_tokens"] = total
        counts["subject_id"] = subject["subject_id"]
        records.append(counts)

    df = pd.DataFrame(records)

    # Source presence arrays
    source_presence = {}
    for prefix in source_prefixes:
        col = prefix
        source_presence[prefix] = (df[col] > 0).values

    # Summary
    summary = {}
    n = len(df)
    for prefix in source_prefixes:
        has = df[prefix] > 0
        summary[prefix] = {
            "n_patients_with": int(has.sum()),
            "n_patients_without": int((~has).sum()),
            "fraction_with": float(has.mean()),
            "mean_tokens_when_present": float(df.loc[has, prefix].mean()) if has.any() else 0,
            "median_tokens_when_present": float(df.loc[has, prefix].median()) if has.any() else 0,
        }

    return {
        "per_patient": df,
        "source_presence": source_presence,
        "source_prefixes": source_prefixes,
        "summary": summary,
    }


def classify_patient_data_tier(
    source_presence: Dict[str, np.ndarray],
    pretrained_sources: Optional[List[str]] = None,
    expanded_sources: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Classify each patient by which token types they have: pretrained
    (from the collaborator base model) vs expanded (added by you).

    This is the key diagnostic: patients whose clinical signal lives
    mostly in expanded tokens are relying on randomly-initialised
    embeddings.  Patients whose signal is mostly in pretrained tokens
    are leveraging the base model's knowledge.

    Tiers:
      "pretrained_only" — only tokens from the base model vocab
      "mostly_pretrained" — has some expanded tokens, but majority pretrained
      "mixed" — substantial tokens from both
      "mostly_expanded" — majority of tokens from expanded sources
      "expanded_only" — only expanded tokens (unlikely in practice)

    Parameters
    ----------
    source_presence : dict[prefix → (N,) bool array]
    pretrained_sources : list of str
        Prefixes that are in the base model vocabulary.
        Default: ["LPR3", "LAB"]
    expanded_sources : list of str
        Prefixes added during vocabulary expansion.
        Default: ["cancer", "pathology", "RKKP"]
    """
    if pretrained_sources is None:
        pretrained_sources = ["LPR3", "LAB"]
    if expanded_sources is None:
        expanded_sources = ["cancer", "pathology", "RKKP"]

    n = len(next(iter(source_presence.values())))

    has_pretrained = np.zeros(n, dtype=bool)
    for src in pretrained_sources:
        if src in source_presence:
            has_pretrained |= source_presence[src]

    has_expanded = np.zeros(n, dtype=bool)
    for src in expanded_sources:
        if src in source_presence:
            has_expanded |= source_presence[src]

    tiers = np.full(n, "unknown", dtype=object)
    tiers[has_pretrained & ~has_expanded] = "pretrained_only"
    tiers[~has_pretrained & has_expanded] = "expanded_only"
    tiers[has_pretrained & has_expanded] = "mixed"

    return tiers


def format_token_profile_report(profile: Dict) -> str:
    """Pretty-print the token source profile."""
    lines = []
    lines.append("=" * 65)
    lines.append("TOKEN SOURCE PROFILE")
    lines.append("=" * 65)

    summary = profile["summary"]
    n = len(profile["per_patient"])

    for prefix in profile["source_prefixes"]:
        s = summary[prefix]
        lines.append(
            f"\n  {prefix:15s}  "
            f"present in {s['n_patients_with']:6d}/{n} patients ({s['fraction_with']:.1%})"
        )
        if s["n_patients_with"] > 0:
            lines.append(
                f"  {'':15s}  "
                f"mean={s['mean_tokens_when_present']:.1f} tokens, "
                f"median={s['median_tokens_when_present']:.0f} tokens"
            )

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


