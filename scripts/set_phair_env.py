"""Wire audited PHAIR server artifacts without retargeting the local study run."""
from datetime import datetime
from pathlib import Path
import shutil
from dotenv import dotenv_values, set_key


def main():
    repo = Path.cwd().resolve()
    env = repo / '.env'
    bundle = Path('/ngc/people/mikwer_r/PHAIR_TO_DALY_MODEL')
    sequences = repo / 'data/processed_data/daly_phair_20261111'
    direct = repo / 'data/models/phair_direct_export_20261111'
    root = repo / 'data/paper_256_v3_national_20261111'
    updates = {
        'OPERA_PAPER_EXTERNAL_SOURCE_CKPT': str(bundle / 'model/model.ckpt'),
        'OPERA_PAPER_EXTERNAL_SOURCE_VOCAB': str(bundle / 'model/vocabulary.pt'),
        'OPERA_PAPER_EXTERNAL_SOURCE_ROPE_LAYOUT': 'interleaved',
        'OPERA_PAPER_EXTERNAL_DATA_CONFIG': 'daly_care_joined_bins_data',
        'OPERA_PAPER_EXTERNAL_SEQUENCE_DIR': str(sequences),
        'OPERA_PAPER_EXTERNAL_WIDTH': '256',
    }
    required = [env, bundle/'model/model.ckpt', bundle/'model/vocabulary.pt',
                direct/'encoder.ckpt', direct/'vocabulary.pt',
                sequences/'vocabulary.pt', sequences/'population_full.csv',
                sequences/'subject_data_train.pt', sequences/'subject_data_tuning.pt']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing required artifacts: ' + ', '.join(missing))
    link = root / 'external_direct'
    if link.exists() or link.is_symlink():
        if link.resolve() != direct.resolve():
            raise ValueError(f'{link} already points elsewhere')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    shutil.copy2(env, env.with_name('.env.before_phair_' + stamp))
    for key, value in updates.items():
        set_key(str(env), key, value)
    root.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(direct, target_is_directory=True)
    # The national workspace gets its own roots; local study pointers stay fixed.
    national_env = root / 'national.env'
    if national_env.exists():
        shutil.copy2(national_env, root / ('national.env.backup_' + stamp))
    shutil.copy2(env, national_env)
    for key, value in {
        'OPERA_PAPER_ROOT': str(root),
        'OPERA_PAPER_ANALYSIS_ROOT': str(root / 'analysis'),
    }.items():
        set_key(str(national_env), key, value)
    settings = dotenv_values(national_env)
    print('National environment:', national_env)
    print('DAPT epochs:', settings.get('OPERA_PAPER_MIN_DAPT_EPOCHS') or '10 (workflow default)')
    print('Existing direct export reused:', link)
    print('Local sequence/MEDS/root pointers and saved overnight.env were not changed.')


if __name__ == '__main__':
    main()
