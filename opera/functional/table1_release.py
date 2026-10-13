"""Conservative bundle-wide suppression for Table 1 (not a compliance certificate)."""


def release_table(table, threshold=5, policy="targeted"):
    if threshold < 2:
        raise ValueError("Disclosure threshold must be at least 2")
    result = table.copy()
    if policy == "primary-only":
        discrete = table.kind.isin(["categorical", "binary", "count"])
        primary = discrete & table["count"].gt(0) & table["count"].lt(threshold)
        tiny_summary = (
            table.kind.eq("continuous")
            & table.n_observed.gt(0)
            & table.n_observed.lt(threshold)
        )
        suppressed = primary | tiny_summary
        result["suppressed"] = suppressed
        result.loc[primary, "display"] = f"<{threshold}"
        result.loc[tiny_summary, "display"] = "Suppressed"
        result.loc[suppressed, ["count", "percent", "median", "q1", "q3"]] = float(
            "nan"
        )
        result = result.drop(columns=["n_total", "n_observed", "n_missing"])
        return result, {
            "threshold": threshold,
            "policy": "primary-only masking for internal publication review",
            "review_required": True,
            "complementary_suppression": False,
            "disclosure_clearance": False,
            "limitations": "Totals, percentages of other cells and overlapping tables may reconstruct masked counts. Not an approved release bundle.",
        }

    def small(values):
        return values.gt(0) & values.lt(threshold)

    unsafe = small(table.n_observed) | small(table.n_missing)
    discrete = table.kind.isin(["categorical", "binary", "count"])
    unsafe |= discrete & (
        small(table["count"]) | small(table.n_observed - table["count"])
    )
    if policy == "targeted":
        return targeted_release(table, threshold)
    if policy != "conservative":
        raise ValueError("Unknown disclosure policy")
    blocked = set(table.loc[unsafe, "variable"])
    # System/value marginals and paired cells are linked releases.
    score_variables = {
        v
        for v in table.variable.unique()
        if "ipi" in str(v).lower() or v in {"index", "prognostic_score_by_system"}
    }
    if blocked & score_variables:
        blocked |= score_variables
    if blocked & {"age", "age_ge70"}:
        blocked |= {"age", "age_ge70"}
    if small(table.n_total).any():
        blocked |= {"patients", "cohort_grouped", "cohort_fine", "Disease"}
    suppressed = table.variable.isin(blocked) | small(table.n_total)
    result["suppressed"] = suppressed
    result.loc[suppressed, "display"] = "Suppressed"
    # Only primary SMALL counts get <threshold; secondary cells never reveal counts.
    primary = suppressed & discrete & small(table["count"])
    result.loc[primary, "display"] = f"<{threshold}"
    numeric = [c for c in ("count", "percent", "median", "q1", "q3") if c in result]
    result.loc[suppressed, numeric] = float("nan")
    # Denominators can reconstruct hidden cells; do not release them in this bundle.
    result = result.drop(columns=["n_total", "n_observed", "n_missing"])
    return result, {
        "threshold": threshold,
        "policy": "variable-wide suppression across all cohorts/splits; linked scoring systems suppressed together; denominators withheld",
        "review_required": True,
    }


def targeted_release(table, threshold):
    """Primary cells plus complementary suppression over explicit additive margins.

    External releases and arbitrary joint inference remain institutional review tasks.
    """
    table = table.reset_index(drop=True)
    counts = table["count"]

    def small(values):
        return values.gt(0) & values.lt(threshold)

    discrete = table.kind.isin(["categorical", "binary", "count"])
    primary = discrete & small(counts)
    suppressed = primary | small(table.n_total) | small(table.n_observed)
    # Hide all categories locally if published percentages could reveal small missingness.
    local_keys = ["cohort_level", "cohort", "split", "variable"]
    for _, rows in table.groupby(local_keys, dropna=False):
        if small(rows.n_missing).any():
            suppressed.loc[rows.index] = True
    binary_complement = table.kind.eq("binary") & small(table.n_observed - counts)
    suppressed |= binary_complement
    equations = []
    count_rows = table.loc[discrete & counts.notna()]
    # Overall = train + tuning + held_out for each displayed category.
    for _, rows in count_rows.groupby(
        ["cohort_level", "cohort", "variable", "category"], dropna=False
    ):
        parent = rows.loc[rows.split == "Overall"]
        children = rows.loc[rows.split != "Overall"]
        if (
            len(parent) == 1
            and len(children)
            and parent["count"].iloc[0] == children["count"].sum()
        ):
            equations.append(list(rows.index))
    # Pooled = sum of grouped cohorts; pooled = sum of fine cohorts.
    for _, rows in count_rows.groupby(["split", "variable", "category"], dropna=False):
        parent = rows.loc[rows.cohort_level == "all"]
        for level in ("cohort_grouped", "cohort_fine"):
            children = rows.loc[rows.cohort_level == level]
            if (
                len(parent) == 1
                and len(children)
                and parent["count"].iloc[0] == children["count"].sum()
            ):
                equations.append(list(parent.index) + list(children.index))
    # Identify nested grouped/fine partitions from the existing fine-cohort summaries.
    memberships = count_rows.loc[
        (count_rows.cohort_level == "cohort_fine")
        & (count_rows.split == "Overall")
        & (count_rows.variable == "cohort_grouped")
    ]
    parents = {}
    for fine, rows in memberships.groupby("cohort"):
        if len(rows) == 1:
            parents[fine] = str(rows.category.iloc[0])
    for _, rows in count_rows.groupby(["split", "variable", "category"], dropna=False):
        for _, parent in rows.loc[rows.cohort_level == "cohort_grouped"].iterrows():
            children = rows.loc[
                (rows.cohort_level == "cohort_fine")
                & rows.cohort.map(parents).eq(parent.cohort)
            ]
            if len(children) and parent["count"] == children["count"].sum():
                equations.append([parent.name, *children.index])
    # A complete categorical partition adds up to the patient total when no values are missing.
    for _, rows in count_rows.loc[count_rows.kind == "categorical"].groupby(
        local_keys, dropna=False
    ):
        scope = rows.iloc[0]
        patient = count_rows.loc[
            (count_rows.variable == "patients")
            & (count_rows.cohort_level == scope.cohort_level)
            & (count_rows.cohort == scope.cohort)
            & (count_rows.split == scope.split)
        ]
        if len(patient) == 1 and rows.n_missing.iloc[0] == 0:
            equations.append([patient.index[0], *rows.index])
    # Hide duplicate marginals and paired score outputs together locally, not globally.
    linked = [
        {"age", "age_ge70"},
        {
            v
            for v in table.variable.unique()
            if "ipi" in str(v).lower() or v in {"index", "prognostic_score_by_system"}
        },
    ]
    while True:
        old = int(suppressed.sum())
        for _, rows in table.groupby(["cohort_level", "cohort", "split"], dropna=False):
            for variables in linked:
                selected = rows.loc[rows.variable.isin(variables)]
                if suppressed.loc[selected.index].any():
                    suppressed.loc[selected.index] = True
        for indices in equations:
            hidden = suppressed.loc[indices]
            if hidden.any() and (
                hidden.sum() == 1
                or counts.loc[hidden.index[hidden]].lt(threshold).all()
            ):
                visible = hidden.index[~hidden]
                if len(visible):
                    large = visible[counts.loc[visible].ge(threshold)]
                    # Prefer hiding an aggregate margin rather than spreading
                    # suppression into otherwise unaffected splits/cohorts.
                    if (
                        table.loc[indices, "variable"].nunique() == 1
                        and indices[0] in visible
                    ):
                        choice = indices[0]
                    else:
                        choice = counts.loc[large if len(large) else visible].idxmin()
                    suppressed.loc[choice] = True
        if int(suppressed.sum()) == old:
            break
    result = table.copy()
    result["suppressed"] = suppressed
    result.loc[suppressed, "display"] = "Suppressed"
    result.loc[primary, "display"] = f"<{threshold}"
    result.loc[suppressed, ["count", "percent", "median", "q1", "q3"]] = float("nan")
    result = result.drop(columns=["n_total", "n_observed", "n_missing"])
    return result, {
        "threshold": threshold,
        "policy": "targeted primary and complementary suppression across explicit category, split and cohort additive margins",
        "review_required": True,
        "limitations": "No guarantee against arbitrary combined equations, external releases or background knowledge; review complete publication bundle.",
    }
