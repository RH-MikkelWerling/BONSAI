"""Compare clinical render provenance with refreshed metadata; aggregate paths/hashes only."""
import hashlib
import json
from pathlib import Path
import yaml

root=Path('data/clinical_registry_refresh/20261111/prepared_clinical')
analysis=yaml.safe_load((root/'analysis.yaml').read_text())
manifest_path=root/'clinical_results/report_manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
checks={'metadata_sha':Path(analysis['patient_metadata']),
        'analysis_config_sha':root/'analysis.yaml','score_config_sha':root/'clinical_scores.yaml'}
passed=bool(manifest)
for key,path in checks.items():
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    matches=manifest.get(key)==digest
    passed=passed and matches
    print(key,'MATCH' if matches else 'MISMATCH',str(path),flush=True)
expected=(root/'patient_metadata.parquet').resolve()
points_to_refresh=Path(analysis['patient_metadata']).resolve()==expected
print('Analysis points to refreshed metadata:',points_to_refresh)
print('Rendered output matches prepared configs and metadata:',passed and points_to_refresh)
print('Source extract refresh/version still needs confirmation; matching hashes certify consumption, not data recency.')
