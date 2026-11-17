import json
import pytest
from opera.evaluation import distributed_analysis as da
from opera.run.migrate_cpu_memory_checkpoints import migrate, previous_signature, OLD_HASH, IDENTITY_PREVIOUS_HASHES, BOUNDED_PREVIOUS_HASHES, RARITY_PREVIOUS_HASHES, IDENTITY_COLLECTION_PREVIOUS_HASHES


@pytest.mark.parametrize("fix", ["columnwise", "identity-sensitivity", "bounded-factorization", "rarity-copy", "identity-collection"])
def test_migration_preserves_results_and_rejects_changed_recipe(tmp_path, fix):
    identity=da.worker_identity()
    overrides=(IDENTITY_COLLECTION_PREVIOUS_HASHES if fix=='identity-collection' else RARITY_PREVIOUS_HASHES if fix=='rarity-copy' else BOUNDED_PREVIOUS_HASHES if fix=='bounded-factorization' else IDENTITY_PREVIOUS_HASHES if fix=='identity-sensitivity' else {'opera/evaluation/distributed_analysis.py':OLD_HASH})
    from pathlib import Path
    for path,digest in overrides.items():
        if Path(path).name in identity['implementation']:
            identity['implementation'][Path(path).name]=digest
    config={'artifacts':[],'bootstrap_seed':2026}
    generation=previous_signature(config,identity,overrides)
    base=tmp_path/'checkpoints'
    old=base/generation
    (old/'prepared').mkdir(parents=True)
    da.save_json(old/'prepared/prepared.json',{'parts':[],'rows':0})
    queue=old/'paired/primary'
    queue.mkdir(parents=True)
    da.save_json(queue/'queue.json',{'implementation':identity,'total':1,'options':{}})
    da.save_json(queue/'batch_00000.ready.json',{})
    da.save_pickle(queue/'batch_00000.result_INTERNAL.pkl',{'gain':0.2})
    da.save_json(queue/'collected.json',{})
    empty=old/'tasks/calibration'
    empty.mkdir(parents=True)
    da.save_json(empty/'queue.json',{'implementation':identity,'total':91,'options':{}})
    da.save_json(base/'active.json',{'generation':generation})
    with pytest.raises(ValueError,match='exact recipe'):
        migrate({**config,'bootstrap_seed':7},base,fix=fix)
    if fix == 'identity-collection':
        import shutil
        incomplete_collection = old/'paired/identity'
        shutil.copytree(queue, incomplete_collection)
        (incomplete_collection/'collected.json').unlink()
        da.save_json(incomplete_collection/'published.json', {'total':1})
    target=migrate(config,base,execute=True,fix=fix)
    assert da.load_pickle(target/'paired/primary/batch_00000.result_INTERNAL.pkl')=={'gain':0.2}
    assert not (target/'tasks/calibration').exists()
    assert (old/'tasks/calibration/queue.json').exists()
    assert json.loads((target/'paired/primary/queue.json').read_text())['implementation']==da.worker_identity()
    assert json.loads((old/'paired/primary/queue.json').read_text())['implementation']==identity
