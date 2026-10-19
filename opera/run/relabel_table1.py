"""Bring an already reviewed Table 1 .tex up to the current row labels.

The formatter's labels were shortened after a disclosure-reviewed table was
rendered (group headings, subtype names under their group, "n (%) or median
(IQR)" stated once in the caption). This rewrites only the label column, the
caption and the column spacing of that file, and removes rows no longer in
Table 1 (the subtype-assignment pass, the detailed regimens, the Male row).
Every remaining value cell is checked to be byte-identical to the input and in
the same order, so the reviewed numbers cannot change and no new review is
needed.

    python -m opera.run.relabel_table1 paper/tables/table1_current_review.tex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from opera.run.format_table1_release import REGION_NAMES
from opera.visualization.paper_taxonomy import FINE_IN_GROUP, FINE_LONG, GROUP_LONG

# Labels of the formatter before 2026-10-19 (the reviewed export used these).
OLD_GROUP_LONG = {
    "DLBCL_like": "DLBCL and related aggressive B-cell lymphomas",
    "BL_LBL": "Burkitt and lymphoblastic lymphoma",
    "Indolent_B_NHL": "Indolent B-cell non-Hodgkin lymphoma",
    "CLL_SLL": "CLL and small lymphocytic lymphoma",
    "T_NHL": "T-cell non-Hodgkin lymphoma",
    "MM": "Myeloma and other plasma-cell neoplasms",
}
SUFFIXES = (", median (IQR)", r", $n$ (\%)")
# Blocks no longer in Table 1: the internal subtype-assignment pass and the
# detailed first-line regimens (format_table1_release.MAIN_TABLE_EXCLUDE).
DROPPED_BLOCKS = ("Subtype assignment", "First-line treatment")
HEADER = re.compile(r"^\\multicolumn\{\d+\}\{@\{\}l\}\{(?P<name>[^{}]*)\}\\\\\s*$")
OLD_LABELS = {"Median follow-up (reverse Kaplan-Meier), years":
              "Reverse Kaplan--Meier follow-up, years"}
CAPTION_END = "by temporal split.}"
REVIEW_SENTENCE = ("Review version pending supervisor sign-off; values are the disclosure-"
                   "processed aggregate export. ")
CAPTION_NEW = r"by temporal split. Values are $n$ (\%) or median (IQR).}"


def _tex(text):
    return text.replace("_", r"\_").replace("%", r"\%")


def relabel_label(label):
    """New text for one label cell (everything before the first ``&``)."""
    for key, old in OLD_GROUP_LONG.items():
        label = label.replace(rf"\textbf{{{_tex(old)}}}", rf"\textbf{{{_tex(GROUP_LONG[key])}}}")
    if label.lstrip().startswith(r"\qquad "):
        for fine, short in FINE_IN_GROUP.items():
            old = rf"\qquad {_tex(FINE_LONG[fine])}"
            if label.lstrip().startswith(old):
                label = label.replace(old, rf"\qquad {_tex(short)}", 1)
                break
    for suffix in SUFFIXES:
        label = re.sub(re.escape(suffix) + r"(\s*)$", r"\1", label)
    for old, new in OLD_LABELS.items():
        if label.strip() == old:
            label = label.replace(old, new)
    region = re.fullmatch(r"(\s*\\quad )(10\d\d)(\s*)", label)
    if region and region[2] in REGION_NAMES:
        label = region[1] + REGION_NAMES[region[2]] + region[3]
    return label


def drop_rows(lines):
    """Remove the dropped blocks; show sex as one row (Female), as the formatter now does."""
    out, skipping, pending_sex = [], False, False
    for line in lines:
        header = HEADER.match(line)
        name = header["name"] if header else None
        for suffix in SUFFIXES:  # older files: "Sex, $n$ (\%)"
            if name and name.endswith(suffix):
                name = name[: -len(suffix)]
        if skipping and line.startswith(r"\quad "):
            continue
        skipping = False
        if name in DROPPED_BLOCKS:
            skipping = True
            continue
        if name == "Sex":
            pending_sex = True
            continue
        if pending_sex and line.startswith(r"\quad "):
            label, rest = line.split("&", 1)
            if label.strip() in {r"\quad F", r"\quad Female"}:
                out.append("Female &" + rest)
            continue  # the other level is the complement
        pending_sex = False
        if name == r"\sffamily\bfseries Other characteristics":
            # Its one row (reverse Kaplan--Meier follow-up) joins Follow-up.
            if out and out[-1].strip() == r"\addlinespace":
                out.pop()
            continue
        out.append(line)
    return out


def relabel(tex):
    out = []
    for line in drop_rows(tex.splitlines()):
        if line.startswith(r"\multicolumn{") and line.rstrip().endswith(r"\\"):
            for suffix in SUFFIXES:
                line = line.replace(suffix + r"}\\", r"}\\")
        elif "&" in line and not line.startswith("%"):
            label, rest = line.split("&", 1)
            line = relabel_label(label) + "&" + rest
        elif line.startswith(r"\caption{") and line.endswith(CAPTION_END):
            line = line[: -len(CAPTION_END)] + CAPTION_NEW
        if line.startswith(r"\caption{"):
            line = line.replace(REVIEW_SENTENCE, "")
        elif line.strip() == r"\setlength{\tabcolsep}{5pt}":
            line = r"\setlength{\tabcolsep}{4pt}"
        out.append(line)
    return "\n".join(out) + ("\n" if tex.endswith("\n") else "")


def value_cells(tex):
    return [line.split("&", 1)[1] for line in tex.splitlines()
            if "&" in line and not line.startswith("%")]


def is_subsequence(kept, original):
    """Every kept value row appears, unchanged and in order, in the original."""
    remaining = iter(original)
    return all(any(cell == candidate for candidate in remaining) for cell in kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path)
    parser.add_argument("--output", type=Path, help="Default: overwrite in place")
    args = parser.parse_args()
    before = args.table.read_text(encoding="utf-8")
    after = relabel(before)
    kept, original = value_cells(after), value_cells(before)
    if not is_subsequence(kept, original):
        raise SystemExit("Refusing: a value cell would change")
    (args.output or args.table).write_text(after, encoding="utf-8")
    print(f"{len(original) - len(kept)} value rows removed (dropped blocks, Male), "
          f"{len(kept)} kept unchanged -> {args.output or args.table}")


if __name__ == "__main__":
    main()
