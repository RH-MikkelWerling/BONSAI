"""Identify clinical renders and reject stale figures during review assembly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RENDER_SOURCES = (
    'opera/run/render_clinical_score_panels.py',
    'opera/visualization/paper_figures.py',
    'opera/visualization/style.py',
)
NATIVE_FOUR = ['Low', 'Low-intermediate', 'High-intermediate', 'High']


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def renderer_sources():
    root = Path(__file__).resolve().parents[2]
    return {name: file_sha(root / name) for name in RENDER_SOURCES}


def render_receipt(directory, specs, split):
    return dict(
        schema=2, evaluation_split=split, renderer_source_sha256=renderer_sources(),
        score_definitions=[{**{key: spec.get(key) for key in (
            'id', 'model', 'score_label', 'score_column', 'group_order',
            'manuscript_figures')}, 'threshold_split': spec.get('threshold_split', 'train')}
            for spec in specs],
        figure_sha256={str(path.relative_to(directory)): file_sha(path)
                       for path in sorted(directory.rglob('*'))
                       if path.suffix.lower() in {'.pdf', '.png'} and path.is_file()},
    )


def verify_clinical_render(directory, *, expected_model=None, split='tuning'):
    """Validate render identity and bytes, without reading patient predictions.

    This verifies freshness and the declared threshold/primary-score contract.
    It is not a clinical-applicability or upstream data-provenance audit.
    """
    directory = Path(directory)
    errors = []
    manifest = directory / 'report_manifest.json'
    if not manifest.is_file():
        return dict(valid=False, errors=['Missing versioned clinical report_manifest.json'], files=0)
    try:
        receipt = json.loads(manifest.read_text())
    except (ValueError, OSError) as exc:
        return dict(valid=False, errors=[f'Unreadable clinical manifest: {exc}'], files=0)
    if receipt.get('schema') != 2:
        errors.append('Legacy clinical render: rebuild with the current clinical renderer')
    if receipt.get('renderer_source_sha256') != renderer_sources():
        errors.append('Clinical renderer source changed since these figures were generated')
    if receipt.get('evaluation_split') != split:
        errors.append(f'Clinical evaluation split must be {split}')
    definitions = receipt.get('score_definitions', [])
    if not definitions:
        errors.append('No score definitions in clinical render manifest')
    for spec in definitions:
        if spec.get('threshold_split', 'train') != 'train':
            errors.append(f"{spec.get('id')}: clinical thresholds were not developed on training data")
        if expected_model and spec.get('model') != expected_model:
            errors.append(f"{spec.get('id')}: model {spec.get('model')} differs from selected {expected_model}")
        label = str(spec.get('score_label', '')).upper().replace('-', '').replace(' ', '')
        if (label == 'NCCNIPI' and (spec.get('score_column') or spec.get('manuscript_figures'))
                and spec.get('group_order') != NATIVE_FOUR):
            errors.append(f"{spec.get('id')}: NCCN-IPI comparison must retain its four native groups")
    expected = receipt.get('figure_sha256', {})
    actual = {str(path.relative_to(directory)): path
              for path in directory.rglob('*') if path.is_file()
              and path.suffix.lower() in {'.pdf', '.png'}}
    if not expected:
        errors.append('No clinical figures recorded in the render manifest')
    if set(actual) != set(expected):
        errors.append('Clinical figure files differ from the recorded output list')
    changed = [name for name, path in actual.items()
               if name in expected and file_sha(path) != expected[name]]
    if changed:
        errors.append(f'Clinical figure content changed: {changed}')
    return dict(valid=not errors, errors=errors, files=len(actual),
                evaluation_split=receipt.get('evaluation_split'), score_definitions=definitions)
