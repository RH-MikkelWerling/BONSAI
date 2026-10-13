from pathlib import Path

import yaml

from opera.run.prepare_independent_specialist_grid import planned_cells


def test_grid_respects_fine_cohort_and_parent_exclusions():
    base = yaml.safe_load(Path('opera/configs/generated/direct_cr_bce_lowrank_endpoint_groups_v1.yaml').read_text())
    base['competing_risk']['fine_names'] = ['CLL', 'DLBCL']
    base['outcomes'] = {'overall_survival': {}, 'sepsis': {}}
    base['cohorts']['CLL_SLL']['exclude_outcomes'] = ['sepsis']
    cells = planned_cells(base)
    assert {(c['fine_cohort'], c['outcome']) for c in cells} == {
        ('CLL', 'overall_survival'), ('DLBCL', 'overall_survival'), ('DLBCL', 'sepsis')
    }
    assert len(planned_cells(base, exclude_fine={'CLL'})) == 2


def test_round_robin_shards_are_disjoint_and_complete():
    cells = list(range(23))
    shards = [cells[index::4] for index in range(4)]
    assert sorted(item for shard in shards for item in shard) == cells
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_plain_shared_template_uses_canonical_fine_inventory():
    base = yaml.safe_load(Path('opera/configs/generated/direct_cr_bce_lowrank_endpoint_groups_v1.yaml').read_text())
    base['competing_risk'].pop('fine_names', None)
    base['outcomes'] = {'overall_survival': {}}
    cells = planned_cells(base, exclude_fine={'CLL'})
    assert len({cell['fine_cohort'] for cell in cells}) == 23
    assert all(cell['fine_cohort'] != 'CLL' for cell in cells)
