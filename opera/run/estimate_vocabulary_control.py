"""Cost-only vocabulary-control report. Never launches fitting or pretraining."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--national-root',type=Path,default=Path('data/paper_256_v3_national_20261111'))
    p.add_argument('--output',type=Path,default=Path('logs/vocabulary_control_cost_20261120.json'))
    p.add_argument('--measured-adaptation-hours',type=float)
    p.add_argument('--measured-joint-hours',type=float)
    a=p.parse_args()
    from opera.run.prepare_epoch20_joint_finetune import checkpoint_record
    source=a.national_root/'external_dapt/version_0/last.ckpt'
    record=checkpoint_record(source)
    if record['completed_epochs']!=10:raise ValueError('Reference is not completed10epoch adaptation')
    report=dict(status='estimate_only_no_training',scope='Optional J3; explicit approval still needed',
                reference_adaptation=record,
                relative_budget='Approximately one original ten-epoch adaptation pass plus one full joint fine-tune; convergence/stopping may differ',
                nominal_pretraining_epochs=10,
                estimated_pretraining_token_exposures=record.get('training_tokens_seen'),
                estimated_gpu_hours=(a.measured_adaptation_hours+a.measured_joint_hours
                    if a.measured_adaptation_hours is not None and a.measured_joint_hours is not None else None),
                missing_cost_inputs=([k for k,v in [('measured original adaptation GPU hours',a.measured_adaptation_hours),
                    ('measured J_EA fine-tuning GPU hours',a.measured_joint_hours)] if v is None]),
                required_design_checks=['Same cohort pretraining patients and cutoff as cohort encoder',
                    'National vocabulary and lab binning without population-pretrained initialization',
                    'Matched local encoder architecture, masking, batching and optimizer; list all differences',
                    'Confirm record eligibility and tokenization before any launch'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
