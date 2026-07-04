# Codex instructions for `poster/`

These instructions apply to every file under `poster/`.

## Purpose

Build a reproducible OPERA scientific poster in which quantitative content is
generated from repository results and final visual composition is completed in
Adobe Illustrator.

## Hard rules

1. Never invent, infer, or silently substitute quantitative results.
2. Development-only values must come from `poster/data/mock/` and every figure
   using them must contain a clearly visible `PROVISIONAL MOCK DATA` label.
3. Final figures must record their source files, filters, model definitions,
   outcome definitions, split, metric, and generation timestamp in a companion
   metadata JSON file.
4. Preserve stable output filenames and artboard dimensions so Illustrator links
   can be refreshed without moving placed artwork.
5. Export scientific figures as SVG and PDF. PNG is preview-only.
6. Keep text editable in SVG whenever practical. Do not convert all text to
   outlines during normal development.
7. Do not reproduce the layout or decorative motifs of the Riemannian Generative
   Decoder poster. OPERA must have its own visual system.
8. Use the user's artwork as a source of compositional principles and approved
   derived textures, not as a full-page visual effect. Information clarity wins.
9. Never place decorative artwork behind small text, axes, legends, uncertainty
   intervals, or dense data marks.
10. Do not add an acronym expansion for OPERA. OPERA is the project/model name.

## Scientific hierarchy

The main poster claim is:

> Shared, outcome-informed representation learning is most useful when labelled
> clinical data are scarce.

The hero result is the controlled label-scarcity analysis. The other main
figures support it:

- outcome-informed representation atlas
- joint-versus-specific transfer matrix
- atypicality-versus-patient-benefit analysis
- concise longitudinal-history and adaptation diagram

The prospective split must be stated consistently as:

- train through 2021
- tune in 2022
- test in 2023+

## Plot conventions

- Primary cross-cohort comparison: delta AUROC versus the named baseline, with
  uncertainty.
- Show individual task trajectories or distributions when aggregation could hide
  heterogeneity.
- Use direct labels where they improve reading.
- Avoid unnecessary panel borders and default plotting-library styling.
- Use a single shared colour mapping for model families across every figure.
- Maintain legibility at expected A0 viewing distance.
- Do not use decorative gradients in data encodings unless the scale is explicit
  and perceptually ordered.

## Visual system

The intended visual balance is approximately 75% analytical clarity and 25%
artistic intensity.

Approved characteristics:

- warm paper-like neutral ground
- dark navy/charcoal typography
- coral, cyan/teal, orange, magenta, and violet accents
- selected translucent collage fragments
- controlled brush textures at margins or transitions
- contrast between quiet regions and one dense focal composition
- geometric structure interacting with fluid painted forms

Avoid:

- science-fiction interfaces
- black full-page backgrounds
- neon HUD styling
- cityscape decoration
- watercolor travel-poster imitation
- many equal-sized infographic boxes
- fake handwritten body text

## Code quality

- Prefer small typed Python modules with clear CLI entry points.
- Use deterministic seeds for mock/example output.
- Fail loudly on missing result columns or ambiguous duplicate rows.
- Add tests for transformations that affect reported values.
- Keep plotting code independent of private data paths through configuration and
  environment variables.
- Use repository-native result schemas rather than manually transcribed values.

## Illustrator automation

JSX scripts should:

- create named layers and swatches
- use linked SVG/PDF figures rather than embedding them by default
- identify placed items by stable names
- never reposition manually arranged linked figures during refresh
- report missing or duplicated links
- check text overflow and objects outside the artboard
- provide separate print and screen export routines

The `.ai` file is the visual master. Codex may generate its initial scaffold and
supporting objects, but should not repeatedly overwrite a manually art-directed
poster.
