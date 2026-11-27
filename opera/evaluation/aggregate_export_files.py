"""Select named aggregate reports, including dynamically named subgroup tables."""
import re

SUBGROUP_TABLES = {'patient_subgroups.csv','patient_subgroup_panels.csv',
    'identity_sensitivity_counts.csv','late_diagnosis_exclusion_sensitivity.csv',
    'negative_transfer_summary.csv'}


def aggregate_export_paths(directory, known_names):
    names=set(known_names)|SUBGROUP_TABLES
    names.update(p.name for p in directory.glob('stratified_*.csv')
                 if re.fullmatch(r'stratified_[A-Za-z0-9_]+_(effects|models)\.csv',p.name))
    return [directory/name for name in sorted(names) if (directory/name).is_file()]
