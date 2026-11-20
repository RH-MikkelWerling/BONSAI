"""Cost-only vocabulary-control report. Never launches fitting or pretraining."""
import argparse
import json
import math
from pathlib import Path


def cost_report(record, adaptation_hours=None, joint_hours=None):
    for value in [adaptation_hours,joint_hours]:
        if value is not None and (not math.isfinite(value) or value<=0):
            raise ValueError('Measured reference GPU hours must be finite and positive')
    tokens=record.get('training_tokens_seen')
    steps=record.get('global_step')
    return dict(status='estimate_only_no_training',
        scope='J3 approved in principle: report cost before starting; no job launched',
        reference_adaptation=record, nominal_pretraining_epochs=20,
        retained_checkpoints=[10,20], joint_finetunes=2,
        relative_budget='Two original ten-epoch adaptation passes plus two full joint fine-tunes; reference estimate, not measured new-run cost',
        reference_equivalent_token_budget=2*tokens if tokens is not None else None,
        reference_equivalent_step_budget=2*steps if steps is not None else None,
        estimated_gpu_hours=(2*adaptation_hours+2*joint_hours
            if adaptation_hours is not None and joint_hours is not None else None),
        missing_cost_inputs=[k for k,v in [('measured original adaptation GPU hours',adaptation_hours),
            ('measured matched joint fine-tuning GPU hours',joint_hours)] if v is None],
        required_design_checks=['Exactly the same 78,668 cohort pretraining patients and pre-2022 cutoff',
            'National vocabulary AND laboratory binning; no pretrained weights',
            'Match cohort encoder architecture, masking, batching, optimizer and schedule',
            'Both joint fits copy J_L/J_L_epoch20 settings, including encoder LR multiplier; change only dapt_ckpt',
            'Measure actual tokens/steps per epoch, validation masked-token loss and unseen national-vocabulary token share'],
        budget_caveat='Reference token/step budget assumes same eligible records and batching as original adaptation. New measured counts may differ; no equal-compute claim.',
        matched_contrasts=['NV10-J_L','NV20-J_L_epoch20','J_EA-NV20','J_EA-NV10','J_E-NV20'],
        target_package='download_vocab_control.zip')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--national-root',type=Path,default=Path('data/paper_256_v3_national_20261111'))
    p.add_argument('--output',type=Path,default=Path('logs/vocabulary_control_cost20_20261120.json'))
    p.add_argument('--measured-adaptation-hours',type=float)
    p.add_argument('--measured-joint-hours',type=float)
    a=p.parse_args()
    from opera.run.prepare_epoch20_joint_finetune import checkpoint_record
    source=a.national_root/'external_dapt/version_0/last.ckpt'
    record=checkpoint_record(source)
    if record['completed_epochs']!=10:raise ValueError('Reference is not completed10epoch adaptation')
    report=cost_report(record,a.measured_adaptation_hours,a.measured_joint_hours)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
