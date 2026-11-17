#!/usr/bin/env bash
# Build the current OPERA manuscript. Since 17 Nov 2026 the default is the story version
# (paper/opera_manuscript_story.tex -> opera_body_story.tex). The consolidated manuscript is kept
# as the previous version and is built only with --consolidated.
set -euo pipefail
cd "$(dirname "$0")/../paper"
mkdir -p build render/story
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=render/story opera_manuscript_story.tex
cp render/story/opera_manuscript_story.pdf opera_manuscript_current.pdf
if [[ "${1:-}" == "--consolidated" ]]; then
  for entry in opera_manuscript_consolidated opera_manuscript; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$entry.tex"
    cp "build/$entry.pdf" "$entry.pdf"
  done
fi
echo 'Current manuscript: paper/opera_manuscript_current.pdf (story version)'
