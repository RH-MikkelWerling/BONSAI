import json
from pathlib import Path
from unittest.mock import patch
import pytest

from opera.run.results_worker import pending_queues
from opera.run.prepare_representation_replay_plan import prepare
from opera.diagnostics.correction_replay import verified_replay,digest_file


def test_multi_queue_ignores_finished_batches_and_checks_generation(tmp_path):
    base=tmp_path/'checkpoints';base.mkdir()
    (base/'active.json').write_text(json.dumps({'generation':'generation'}))
    queue=base/'generation/paired/primary';queue.mkdir(parents=True)
    (queue/'queue.json').write_text('{}')
    (queue/'batch_00000.ready.json').write_text('{}')
    assert pending_queues(base)==[queue]
    (queue/'batch_00000.result_INTERNAL.pkl').touch()
    assert pending_queues(base)==[]
    (queue/'batch_00000.failed.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Failed queue'):
        pending_queues(base)
    (base/'active.json').write_text(json.dumps({'generation':'../bad'}))
    with pytest.raises(ValueError,match='Invalid'):
        pending_queues(base)


def test_plan_uses_all_three_verified_checkpoints_without_training(tmp_path):
    root=tmp_path/'national';(root/'direct_fits').mkdir(parents=True)
    local=tmp_path/'local'
    (root/'direct_fits/bundle.json').write_text(json.dumps({'provenance':{'frozen_local_bundle':str(local)}}))
    def bundle_rows(path):
        origin='L' if Path(path)==local else 'E' if Path(path).name=='direct_fits' else 'EA'
        row=dict(name='seed42/'+origin+('/fix_J_OA_OWN' if origin=='L' else '/R'),origin=origin,
                 phase='router_fix' if origin=='L' else 'external',seed=42,
                 run_directory=origin+'/seed42/hash',scientific_hash='frozen')
        return {},[(row,Path(path)/'configs/R.yaml',dict(seed=42,competing_risk={'readout_mode':'own'}))]
    with patch('opera.run.prepare_representation_replay_plan.bundle_runs',side_effect=bundle_rows), \
         patch('opera.run.prepare_representation_replay_plan.verified_training_selection',return_value={'best_checkpoint_sha256':'verified'}):
        plan=prepare(root,tmp_path/'plan.json')
    assert {j['encoder'] for j in plan['jobs']}=={'L','E','EA'}
    assert plan['inference_passes']==15
    assert not plan['training_launched']


def test_replay_reuse_checks_inputs_and_saved_outputs(tmp_path):
    output=tmp_path/'output';output.mkdir()
    prediction=output/'direct_cr_predictions.parquet';prediction.write_bytes(b'saved')
    job={'seed':42}
    (output/'diagnostic_provenance.json').write_text(json.dumps(dict(job=job,mode='all',split='tuning',
        checkpoint_sha256='checkpoint',outputs={prediction.name:digest_file(prediction)})))
    assert verified_replay(output,job,'all','tuning','checkpoint')
    prediction.write_bytes(b'changed')
    with pytest.raises(ValueError,match='output changed'):
        verified_replay(output,job,'all','tuning','checkpoint')


@pytest.mark.parametrize('e,ea,expected',[(.017131,.030203,'R_EA'),(.017,.02,'R_E'),
    (.001,.003,'J_OA_OWN_L'),(.004,.01,'R_EA'),(.02,.01,'R_E'),(.0052,.0052,'J_OA_OWN_L')])
def test_recorded_encoder_rule(e,ea,expected):
    from opera.run.record_encoder_selection import choose_encoder
    assert choose_encoder(e,ea,ea-e)==expected


def test_presentation_figure_exports_saved_aggregate_effects(tmp_path,monkeypatch):
    import pandas as pd
    from scripts.export_encoder_architecture_figure import main
    codes=['EXT_ARCH_L','EXT_ARCH_E','EXT_ARCH_EA','EXT_J_EA_VS_XP']
    path=tmp_path/'effects.csv'
    pd.DataFrame([dict(contrast=code,split='held_out',seed=42,metric='auc',stratum='all_cells',
                       horizon_days=0,gain=.01,lower=-.01,upper=.02,n_cells=100) for code in codes]).to_csv(path,index=False)
    output=tmp_path/'figures'
    monkeypatch.setattr('sys.argv',['figure','--effects',str(path),'--output-dir',str(output),'--label','Held-out provisional'])
    main()
    assert (output/'encoder_architecture.pdf').is_file()
    assert (output/'encoder_architecture.png').is_file()
    assert set(pd.read_csv(output/'figure_data.csv').contrast)==set(codes)


def test_figure_watcher_waits_then_exports_preview_and_final(tmp_path,monkeypatch):
    from scripts.watch_result_figures import export_once
    project=tmp_path/'project';root=tmp_path/'figures'
    assert not export_once(project,root)
    progress=project/'results.progress';progress.mkdir(parents=True)
    (progress/'effect_summaries.csv').write_text('contrast,gain\na,0.1\n')
    calls=[]
    def export(command,check):
        output=Path(command[command.index('--output-dir')+1]);output.mkdir(parents=True)
        calls.append(output)
    monkeypatch.setattr('scripts.watch_result_figures.subprocess.run',export)
    assert not export_once(project,root)
    assert not export_once(project,root)
    assert len(calls)==1
    final=project/'results';final.mkdir()
    (final/'analysis_manifest.json').write_text('{}')
    (final/'effect_summaries.csv').write_text((progress/'effect_summaries.csv').read_text())
    assert export_once(project,root)
    assert len(calls)==2
