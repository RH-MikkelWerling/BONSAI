# OPERA Poster

This directory contains the reproducible scientific-poster workflow for OPERA.
The analysis repository remains the source of truth for experiments and results;
Illustrator remains the visual master for final composition.

## Core principle

- **Code owns facts:** result extraction, statistics, plots, labels, tables, and figure exports.
- **Illustrator owns composition:** hierarchy, typography, masks, artwork, texture, and final print preparation.
- **Artwork supports the story:** it must never obscure axes, labels, uncertainty, or the main scientific result.

## Scientific story

The working poster is organised around one central claim:

> Shared, outcome-informed representation learning is most useful when labelled clinical data are scarce.

Supporting components:

1. Longitudinal EHR histories are encoded into patient representations.
2. OPERA adapts shared representations using outcomes across hematologic diseases.
3. Controlled label-scarcity experiments compare joint OPERA, grouped OPERA,
   fine-diagnosis OPERA, and task-specific XGBoost.
4. Transfer and atypicality analyses identify which diagnoses and patients benefit.
5. Evaluation uses the prespecified temporal split: train through 2021, tune in
   2022, and test in 2023 onward.

## Directory layout

```text
poster/
├── AGENTS.md                  Codex-specific rules
├── README.md                  workflow documentation
├── poster.yaml                poster content and design contract
├── data/
│   ├── derived/               generated compact tables used by figures
│   └── mock/                  explicitly provisional development data
├── figures/
│   ├── generated/             direct script outputs
│   ├── final/                 stable linked SVG/PDF assets for Illustrator
│   └── previews/              raster previews
├── illustrator/
│   ├── scripts/               JSX automation scripts
│   └── templates/             small versioned Illustrator templates
├── artwork/
│   ├── references/            low-resolution visual references only
│   ├── textures/              approved derived textures
│   └── masks/                 reusable vector/raster masks
├── scripts/                   data collection, plotting, and preflight
└── build/                     local generated output; not committed
```

## What belongs in Git

Commit:

- plotting and extraction code
- YAML/JSON content specifications
- Illustrator JSX scripts
- SVG/PDF figures when reasonably sized
- low-resolution artwork references and approved derived textures
- small Illustrator templates when useful

Do not commit normally:

- private patient-level data
- raw result trees
- full-resolution original artwork
- Illustrator recovery/autosave files
- generated previews and print exports
- large `.ai` binaries unless Git LFS is deliberately enabled

The working `.ai` file should initially remain local or in an approved shared
storage location. Once the layout stabilises, a release copy can be stored with
Git LFS or attached to a tagged release.

## Planned commands

```bash
# Create compact poster result tables from OPERA evaluation outputs
python -m poster.scripts.collect_results --config poster/poster.yaml

# Build all scientific figures
python -m poster.scripts.build_figures --config poster/poster.yaml

# Validate numbers, labels, linked assets, and provisional content
python -m poster.scripts.preflight --config poster/poster.yaml
```

Illustrator scripts will create the initial A0 scaffold, update linked figure
assets, check the document, and export print/screen versions.

## Quantitative integrity

Mock values may be used only in `poster/data/mock/` and must be visibly marked
`PROVISIONAL` in every generated figure. Final poster assets must be traceable to
an OPERA result file and must pass the poster preflight checks.
