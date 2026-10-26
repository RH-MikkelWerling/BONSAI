"""Review sheet of the candidate borrowing pairs, for the study haematologists;
generated from opera/configs/borrowing_hypotheses.yaml so the sheet and the
analysis cannot drift apart.

The sheet is deliberately neutral: hypotheses and negative controls appear in
one list, in a fixed random order, without our rationale or expected strength,
and the reviewer rates each pair. The key that maps the sheet's numbers back to
the pairs (and to what we proposed them as) is written next to it and is not
sent."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import yaml

from opera.visualization.paper_taxonomy import disease_long_name

REPO = Path(__file__).resolve().parents[2]


def tex(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), (">=", r"$\ge$"), ("~", r"$\sim$")):
        text = text.replace(a, b)
    return text


# The 24 diseases by group, and the 69 outcomes by clinical domain, as in the
# paper (definitions supplement; outcome_display_groups_paper_v1.yaml).
DISEASES = [
    ("Large B-cell lymphomas", ["Diffuse large B-cell lymphoma (DLBCL)",
                                "Richter transformation (RT)",
                                "Transformed follicular lymphoma (tFL)"]),
    ("Burkitt and lymphoblastic lymphomas", ["Burkitt lymphoma (BL)",
                                             "Lymphoblastic lymphoma (LBL)"]),
    ("Mantle-cell lymphoma", ["Mantle-cell lymphoma (MCL)"]),
    ("Indolent B-cell lymphomas", ["Follicular lymphoma (FL)", "Lymphoplasmacytic lymphoma (LPL)",
                                   "Extranodal marginal-zone lymphoma (EMZL)",
                                   "Splenic marginal-zone lymphoma (SMZL)",
                                   "Nodal marginal-zone lymphoma (NMZL)",
                                   "B-cell lymphoma, unspecified (BCL)"]),
    ("CLL and SLL", ["Chronic lymphocytic leukaemia (CLL)", "Small lymphocytic lymphoma (SLL)"]),
    ("Hairy-cell leukaemia", ["Hairy-cell leukaemia (HCL)"]),
    ("Hodgkin lymphoma", ["Hodgkin lymphoma (HL)"]),
    ("T-cell lymphomas", ["Peripheral T-cell lymphoma, NOS (PTCL)",
                          "Angioimmunoblastic T-cell lymphoma (AITL)",
                          "Anaplastic large-cell lymphoma (ALCL)",
                          "T-cell lymphoma, unspecified (TCL)"]),
    ("Plasma-cell neoplasms", ["Multiple myeloma (MM)", "Plasma-cell leukaemia (PCL)",
                               "Solitary plasmacytoma"]),
    ("Amyloidosis", ["AL amyloidosis"]),
]
OUTCOMES = [
    ("Disease course and later malignancy", "overall survival; second-line treatment; treatment "
     "failure; treatment failure or transformation; new haematological malignancy; new malignancy (any)"),
    ("Infection and antimicrobial use", "sepsis; serious infection (including invasive fungal "
     "infection and pneumocystosis); antibiotic treatment; intravenous antimicrobial use; "
     "prescription antimicrobial use; infection diagnosis or antimicrobial use"),
    ("Erythroid findings and transfusion", "anaemia grade 2 and 3; red-cell transfusion; "
     "any transfusion; haptoglobin decreased"),
    ("Neutrophil, white-cell and platelet findings", "neutropenia grade 2 and 3; leukopenia "
     "grade 2 and 3; thrombocytopenia grade 2 and 3; platelet transfusion; severe cytopenia"),
    ("Lymphocyte and eosinophil counts", "lymphocytosis grade 2 and 3; eosinophilia"),
    ("Renal injury and support", "acute kidney injury; creatinine increased grade 2 and 3; "
     "creatinine rise above 1.5 times the pre-treatment value; dialysis"),
    ("Liver injury and coagulation", "ALT increased grade 2 and 3; alkaline phosphatase "
     "elevation, two levels; bilirubin increased grade 2 and 3; hepatotoxicity; INR increased "
     "grade 2 and 3"),
    ("Sodium and potassium", "hyponatraemia grade 2; sodium below 125 mmol/L; potassium below "
     "3.0 mmol/L; hypernatraemia grade 2 and 3; hyperkalaemia grade 2 and 3"),
    ("Calcium and magnesium", "hypercalcaemia grade 2 and 3; hypocalcaemia grade 2 and 3; "
     "hypomagnesaemia grade 2"),
    ("Metabolic and cell-turnover markers", "hypertriglyceridaemia grade 2 and 3; "
     "hypoalbuminaemia grade 2 and 3; hyperphosphataemia; hyperuricaemia; LDH increased"),
    ("Cardiovascular and thrombotic events", "atrial fibrillation; heart failure; stroke or "
     "transient ischaemic attack; venous thromboembolism"),
    ("Acute care and other events", "acute hospitalisation; emergency department stay; "
     "respiratory support; bone fracture"),
]


RATINGS = ("strong", "moderate", "weak", "none", "don't know")


def candidates(plan: dict) -> list[dict]:
    """Every pair the reviewers rate, numbered P1, P2, ... in a fixed random order."""
    pairs = ([{**e, "proposed_as": "hypothesis"} for e in plan.get("hypotheses", [])]
             + [{**e, "proposed_as": "negative control"}
                for e in plan.get("negative_controls", [])])
    random.Random(2026).shuffle(pairs)
    return [{**pair, "id": f"P{i}"} for i, pair in enumerate(pairs, 1)]


def write_key(pairs: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "target", "domain", "proposed_as"])
        for pair in pairs:
            writer.writerow([pair["id"], pair["source"], pair["target"], pair["domain"],
                             pair["proposed_as"]])


def outcome_words(domain: str) -> str:
    return "all outcomes" if domain == "All outcomes" else domain[0].lower() + domain[1:]


def render(plan: dict) -> str:
    rows = "\n".join(
        f"{e['id']} & Learn from \\textbf{{{tex(disease_long_name(e['source']))}}}"
        f"\\newline to predict for \\textbf{{{tex(disease_long_name(e['target']))}}} & "
        f"{tex(outcome_words(e['domain']))} & "
        f"\\viewfield{{{e['id']}}} & \\commentfield{{{e['id']}}} \\\\ \\midrule"
        for e in candidates(plan))
    choices = ",".join(f"{r}={r}" for r in RATINGS).replace("don't know=don't know",
                                                             "don't know=dont_know")

    diseases = "\n".join(
        f"\\item[] \\textbf{{{tex(group)}}}: {tex('; '.join(members))}"
        for group, members in DISEASES)
    outcomes = "\n".join(
        f"\\item[] \\textbf{{{tex(domain)}}}: {tex(items)}" for domain, items in OUTCOMES)

    return r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amssymb}
\usepackage{booktabs,longtable,array,xcolor,enumitem}
\usepackage[hidelinks]{hyperref}
% Fillable form: one exclusive choice and one comment box per pair.
\newcommand{\viewfield}[1]{\ChoiceMenu[popdown,name=view_#1,width=2.1cm,height=1.2em,
  bordercolor={0.6 0.6 0.6},charsize=8pt,default= ]{}{ = ,""" + choices + r"""}}
\newcommand{\commentfield}[1]{\TextField[name=comment_#1,multiline=true,width=5.2cm,
  height=1.35cm,bordercolor={0.7 0.7 0.7},charsize=8pt]{}}
\renewcommand{\familydefault}{\sfdefault}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\pagestyle{plain}
\begin{document}
\begin{Form}
{\Large\bfseries Which lymphoid cancers should learn from each other?}\par
{\large Request for review of a list fixed before testing}\par
{\footnotesize\color{black!55}Draft for the study haematologists. Fill in this PDF directly
(Adobe Reader or Preview), save it and send it to Mikkel Werling.}\par
Your name: \TextField[name=reviewer,width=7cm,height=1.1em,bordercolor={0.7 0.7 0.7}]{}

\textbf{Background.} We have built one prediction model for patients starting first-line treatment
for any of 24 lymphoid cancers (37,531 Danish patients). For each patient it predicts the risk of
69 outcomes over the following two years, from relapse and second-line treatment to infection,
cytopenia, organ toxicity and hospital admission. Because the model learns from all diseases at
once, what it learns in one disease can be used for another.

\textbf{What we will test.} After the model is trained, we can switch off what it learned from one
disease and see how much worse its predictions become for another. We want to compare this with
clinical knowledge, and that comparison is only fair if the clinical expectations are written down
\textbf{before} we look at the results on the test patients. Your ratings are those expectations.

\textbf{What we ask of you.}
\begin{enumerate}[leftmargin=1.4em,itemsep=1pt,topsep=0pt]
  \item For each pair below: how much would you expect what is learned in the first disease to
  help predict the stated outcomes in the second? Choose \emph{strong}, \emph{moderate},
  \emph{weak}, \emph{none} or \emph{don't know}, and add a comment where useful. The pairs are in
  random order, and some may well deserve \emph{none}.
  \item Suggest pairs we have missed. Any disease can be paired with any other, for any group of
  outcomes; the diseases and outcomes are listed at the end.
\end{enumerate}

{\small\setlength{\tabcolsep}{4pt}
\begin{longtable}{L{0.55cm}L{4.6cm}L{2.8cm}L{2.3cm}L{5.4cm}}
\toprule
& Pair & For predicting & Expected help & Comment\\ \midrule\endfirsthead
\toprule
& Pair & For predicting & Expected help & Comment\\ \midrule\endhead
""" + rows + r"""
\end{longtable}}

\textbf{Pairs we have missed} (learn from which disease, to predict which outcomes in which
disease, and how much you would expect it to help):\par
\TextField[name=missed_pairs,multiline=true,width=\linewidth,height=3.2cm,
  bordercolor={0.7 0.7 0.7},charsize=9pt]{}\par\medskip
\textbf{Other comments:}\par
\TextField[name=other_comments,multiline=true,width=\linewidth,height=2.4cm,
  bordercolor={0.7 0.7 0.7},charsize=9pt]{}

\newpage
{\large\bfseries The 24 diseases}\par
\begin{itemize}[leftmargin=0pt,itemsep=2pt,topsep=2pt]
""" + diseases + r"""
\end{itemize}

{\large\bfseries The 69 outcomes, by clinical domain}\par
Each outcome is predicted at 30, 60 and 90 days and at 6, 12 and 24 months after the start of
first-line treatment. Laboratory grades are based on the CTCAE laboratory thresholds; pairs can name a
whole domain or ``all outcomes''.
\begin{itemize}[leftmargin=0pt,itemsep=2pt,topsep=2pt]
""" + outcomes + r"""
\end{itemize}
\end{Form}
\end{document}
"""


def paper_table(plan: dict) -> str:
    """Supplementary table of the prespecified pairs with rationale and evidence."""
    def rows(entries, kind):
        out = []
        for i, e in enumerate(entries, 1):
            evidence = "; ".join(e.get("evidence", [])) or "--"
            out.append(
                f"{kind}{i} & {tex(disease_long_name(e['source']))} & "
                f"{tex(disease_long_name(e['target']))} & {tex(outcome_words(e['domain']))} & "
                f"{tex(e.get('clinician_rating', '--'))} & "
                f"{tex(' '.join(e.get('rationale', '').split()))} ({tex(evidence)})\\\\")
        return "\n".join(out)

    def excluded(entries):
        return "\n".join(
            f"X{i} & {tex(disease_long_name(e['source']))} & "
            f"{tex(disease_long_name(e['target']))} & {tex(outcome_words(e['domain']))} & "
            f"-- & {tex(e['reason'])}\\\\"
            for i, e in enumerate(entries, 1))

    margin = plan.get("margin_points")
    note = (f" Pairs were labelled against a practical margin of {margin} AUC points."
            if margin is not None else "")
    return r"""% Generated by opera/run/render_borrowing_pairs_review.py from
% opera/configs/borrowing_hypotheses.yaml; do not edit by hand.
\begingroup\scriptsize\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.03\linewidth}p{0.15\linewidth}p{0.15\linewidth}p{0.13\linewidth}p{0.07\linewidth}p{0.41\linewidth}@{}}
\caption{\textbf{Prespecified borrowing pairs and negative controls.} Fixed before the
test set was opened and reviewed by the study haematologists. Source: the disease
switched on alone; target: the disease predicted; outcomes: the clinical domain
tested. Rating: the study haematologists' expected strength, given blind to the
rationale and to whether the pair was proposed as a hypothesis or a control.""" + tex(note) + r"""}\label{tab:borrowing-pairs}\\
\toprule & Source & Target & Outcomes & Rating & Rationale (evidence)\\\midrule\endfirsthead
\toprule & Source & Target & Outcomes & Rating & Rationale (evidence)\\\midrule\endhead
\bottomrule\endfoot
\multicolumn{6}{@{}l}{\textit{Expected to share signal}}\\
""" + rows(plan.get("hypotheses", []), "H") + r"""
\multicolumn{6}{@{}l}{\textit{Negative controls}}\\
""" + rows(plan.get("negative_controls", []), "C") + r"""
\multicolumn{6}{@{}l}{\textit{Candidates excluded before rating (estimate not informative)}}\\
""" + excluded(plan.get("excluded_candidates", [])) + r"""
\end{longtable}
\endgroup
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        default=REPO / "opera/configs/borrowing_hypotheses.yaml")
    parser.add_argument("--output", type=Path,
                        default=REPO / "paper/review/borrowing_pairs_review.tex")
    parser.add_argument("--paper-table", type=Path,
                        default=REPO / "paper/tables/borrowing_pairs.tex",
                        help="Supplementary table of the pairs, with rationale and evidence")
    args = parser.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(plan))
    write_key(candidates(plan), args.output.with_name("borrowing_pairs_key.csv"))
    args.paper_table.write_text(paper_table(plan))
    print(f"Written: {args.output} and {args.paper_table}")


if __name__ == "__main__":
    main()
