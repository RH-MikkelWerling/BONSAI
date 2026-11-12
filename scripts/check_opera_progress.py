"""Read-only server progress snapshot; never loads patient-level data or checkpoints."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path('data/paper_256_v3/analysis/local_test_20261109_four_allocations')
    national=Path('data/paper_256_v3_national_20261111')
    print('TEST ANALYSIS',flush=True)
    for name in ('pipeline_actions_complete','cpu_numerical_done','cpu_clinical_done'):
        path=root/name if name=='pipeline_actions_complete' else Path('data/paper_256_v3/predictions/.overnight/local_test_20261109_four_allocations')/name
        print(name,path.exists(),flush=True)
    subprocess.run([sys.executable,'-m','opera.run.analysis_cpu_worker','--checkpoint-dir',str(root/'_cpu_checkpoints_INTERNAL'),'--status'],check=True)
    print('\nDAPT',flush=True)
    dapt=national/'external_dapt/version_0'
    for name in ('best.ckpt','last.ckpt','checkpoint_metadata.json'):
        path=dapt/name
        print(name,path.exists(),flush=True)
        if name.endswith('.json') and path.exists():
            print(path.read_text(),flush=True)
    print('\nCLINICAL REFRESH',flush=True)
    clinical=Path('data/clinical_registry_refresh/20261111/prepared_clinical/clinical_results')
    print('report_manifest.json:',(clinical/'report_manifest.json').exists(),flush=True)
    for name in ('clinical_refresh_20261111.log','phair_dapt_20261111_restart.log'):
        path=Path('logs')/name
        if path.exists():
            # Progress bars can make training logs large: read only the final 12KB.
            with path.open('rb') as handle:
                handle.seek(max(0,path.stat().st_size-12000))
                tail=handle.read().decode(errors='replace').replace('\r','\n').splitlines()
            print('\n'+name+' (last lines):\n'+'\n'.join(tail[-8:]),flush=True)


if __name__=='__main__':
    main()
