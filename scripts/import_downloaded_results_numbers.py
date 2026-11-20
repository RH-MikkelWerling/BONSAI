"""Generate manuscript numbers from a verified completed aggregate download."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from opera.run.build_paper_numbers import contrast_numbers, with_ci


# Each row names both models. Encoder settings belong to the group or label.
RESULT_ROWS = [
    (None, 'A. Does sharing labels improve prediction? Cohort encoder.', ''),
    ('P1', 'Diseases pooled', 'Specialists'),
    ('P2', 'Outcomes pooled', 'Specialists'),
    ('P5', 'Joint model', 'Specialists'),
    ('P3', 'Joint model', 'Outcomes pooled'),
    ('P4', 'Joint model', 'Diseases pooled'),
    (None, 'B. What does specialisation add? Cohort encoder.', ''),
    ('OFF_J', 'Corrections only', 'Joint model'),
    ('OAOWN_OFF', 'OPERA', 'Corrections only'),
    ('O2_MATCHED', 'OPERA', 'Joint model'),
    (None, 'C. What does specialisation add? Adapted national encoder.', ''),
    ('OFF_J_EA', 'Corrections only', 'Joint model'),
    ('OAOWN_OFF_EA', 'OPERA', 'Corrections only'),
    ('EXT_ARCH_EA', 'OPERA', 'Joint model'),
    ('AB_OFF_EA', 'Learned routing', 'Corrections only'),
    ('OAOWN_AB_EA', 'OPERA', 'Learned routing'),
    (None, 'D. What do national pretraining and specialty adaptation add?', ''),
    ('E1', 'Joint (national)', 'Joint (cohort)'),
    ('E3', 'Joint (adapted)', 'Joint (cohort)'),
    ('E2', 'OPERA (national)', 'OPERA (cohort)'),
    ('E4', 'OPERA (adapted)', 'OPERA (cohort)'),
    ('EXT_ADAPTATION', 'OPERA (adapted)', 'OPERA (national)'),
    (None, 'E. How do shared models compare with specialist and pooled alternatives?', ''),
    ('T3', 'Neural specialists', 'XGBoost specialists'),
    ('EXT_R_EA_VS_S_L', 'OPERA (adapted)', 'Neural specialists'),
    ('T7', 'Pooled XGBoost', 'XGBoost specialists'),
    ('O1', 'OPERA (cohort)', 'XGBoost specialists'),
    ('T5_router', 'OPERA (adapted)', 'XGBoost specialists'),
    ('T5_shared', 'Joint (adapted)', 'XGBoost specialists'),
    ('EXT_J_EA_VS_XP', 'Joint (adapted)', 'Pooled XGBoost'),
    ('EXT_R_EA_VS_XP', 'OPERA (adapted)', 'Pooled XGBoost'),
    ('EXT_R_AB_EA_VS_XP', 'Routing (adapted)', 'Pooled XGBoost'),
    ('T9', 'Joint (cohort)', 'XGBoost + registry'),
    # Ridge logistic regression per task (19 Nov). FINAL_R_EA_VS_LR appears once the server rerun
    # exports it; a contrast absent from the source is skipped, as for every row.
    ('T4', 'XGBoost specialists', 'Logistic regression'),
    ('O7', 'OPERA (cohort)', 'Logistic regression'),
    ('FINAL_R_EA_VS_LR', 'OPERA (adapted)', 'Logistic regression'),
]


def render_table(panel):
    lines=[]
    def value(row,metric):
        point=row.get(metric+'_gain')
        bounds=row.get(metric+'_interval')
        if point is None or pd.isna(point) or not isinstance(bounds,str):
            return '--'
        low,high=map(float,bounds.split(' to '))
        # Avoid displaying negative zero at the chosen table precision.
        def fmt(v):
            return f'{0. if abs(v)<.0005 else v:.3f}'.replace('-', '$-$')
        return (r'\shortstack[c]{\textbf{'+fmt(point)+r'}\\[-1.8pt]'
                +r'{\fontsize{7}{7.6}\selectfont ('+fmt(low)+' to '+fmt(high)+r')}}')
    first_group=True
    for code,candidate,reference in RESULT_ROWS:
        if code is None:
            if first_group:
                first_group=False
                continue  # first heading is emitted directly by the table preamble
            lines.append(r'\specialrule{0.4pt}{2pt}{1pt}')
            lines.append(r'\noalign{\vskip1.5pt\hbox{\textbf{'+candidate+r'}}\vskip1.5pt}')
            lines.append(r'\specialrule{0.25pt}{0pt}{2pt}')
            continue
        row=panel.get(code)
        if row is None:
            continue
        candidate=candidate.replace('OPERA',r'\opera{}')
        reference=reference.replace('OPERA',r'\opera{}')
        lines.append(' & '.join([r'\rule[-5.2pt]{0pt}{13.6pt}'+candidate,reference,f'{int(row["cells"]):,}',value(row,'auc'),
            value(row,'ipa'),str(row.get('cells_better','--')).replace('%',r'\%')])+r' \\')
    return '\n'.join(lines)+'\n'+r'\bottomrule'+'\n'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.bundle/'bundle_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(a.bundle/name)!=digest:
            raise ValueError(f'Checksum mismatch: {name}')
    source=json.loads((a.bundle/'analysis_manifest.json').read_text())
    if source.get('config',{}).get('opera_model')!='J_OA_OWN_L':
        raise ValueError('This table maps generic O* contrasts to cohort OPERA; '
                         'review model identities before importing another primary alias')
    effects=pd.read_csv(a.bundle/'tables/effect_summaries.csv.gz',low_memory=False)
    balanced=pd.read_csv(a.bundle/'tables/effect_summaries_disease_balanced.csv.gz',low_memory=False)
    pooled=pd.read_csv(a.bundle/'tables/contrasts_all_horizons.csv.gz',
                       usecols=['split','seed','metric','contrast','gain'])
    if set(effects.split)!={'held_out'} or set(effects.seed)!={42} or set(pooled.seed)!={42}:
        raise ValueError('This import expects held-out seed 42 only')
    numbers={}
    for frame,base in [(effects,''),(balanced,'_disease_balanced')]:
        for horizon,suffix in [(0,''),(365,'_1y')]:
            numbers.update(contrast_numbers(frame,None,horizon=horizon,suffix=suffix+base,
                pooled_cells=pooled if horizon==0 and not base else None))
            # Preserve explicit EXT architecture identities, not only legacy CODES.
            selected=frame[(frame.stratum=='all_cells')&(frame.horizon_days==horizon)]
            if selected.duplicated(['contrast','metric']).any():
                raise ValueError('Ambiguous headline rows')
            for r in selected.itertuples():
                if r.metric in ['auc','ipa']:
                    value=with_ci(r.gain,r.lower,r.upper)
                    if value:
                        numbers[f'{r.contrast}_{r.metric}{suffix}{base}']=value
                    numbers[f'{r.contrast}_cells{suffix}{base}']=f'{int(r.n_cells):,}'
        for value,label in [('(-1.001, 19.0]','0_19'),('(19.0, 99.0]','20_99'),
                            ('(99.0, 499.0]','100_499'),('(499.0, inf]','500plus')]:
            row=frame[(frame.contrast=='P5')&(frame.metric=='auc')&(frame.horizon_days==0)&
                      (frame.stratum=='support_bin')&(frame.stratum_value==value)]
            if len(row)!=1:
                raise ValueError(f'Expected unique saved support bin: {value}')
            r=row.iloc[0]
            numbers[f'P5_auc_support_{label}{base}']=with_ci(r.gain,r.lower,r.upper)
    # Count outcome/horizon predictors with actual known-status test predictions;
    # inventory paths alone can also include non-estimable placeholders.
    xp_tasks=set()
    for part in pd.read_csv(a.bundle/'tables/model_coverage.csv.gz',chunksize=100_000,
        usecols=['model','seed','split','outcome','horizon_days','n_known_predictions']):
        part=part[(part.model=='XP')&(part.seed==42)&(part.split=='held_out')&(part.n_known_predictions>0)]
        xp_tasks.update(zip(part.outcome,part.horizon_days))
    numbers['pooled_xgboost_predictors']=f'{len(xp_tasks):,}'
    selected=effects[(effects.stratum=='all_cells')&(effects.horizon_days==0)]
    panel=[]
    for code,_,_ in RESULT_ROWS:
        if code is None or code not in set(selected.contrast):
            continue
        record=dict(contrast=code)
        for metric in ['auc','ipa']:
            rows=selected[(selected.contrast==code)&(selected.metric==metric)]
            if rows.empty:
                continue
            row=rows.iloc[0]
            record[f'{metric}_gain']=row.gain
            record[f'{metric}_interval']=f'{row.lower} to {row.upper}'
            if metric=='auc':
                record['cells']=row.n_cells
        rare=effects[(effects.contrast==code)&(effects.metric=='auc')&
                     (effects.horizon_days==0)&(effects.stratum=='subtype_size')&
                     effects.stratum_value.astype(str).str.startswith('(-1')]
        if len(rare)>1:
            raise ValueError('Ambiguous rare-disease stratum')
        if len(rare):
            row=rare.iloc[0];record['rare_auc_gain']=row.gain
            record['rare_auc_interval']=f'{row.lower} to {row.upper}'
        gains=pooled[(pooled.contrast==code)&(pooled.metric=='auc')].gain.dropna()
        if len(gains):
            record['cells_better']=f'{100*(gains>0).mean():.0f}%'
        panel.append(record)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    macros=['% Held-out aggregate import; seed 42. Source hashes in paper_numbers.json.',r'\makeatletter']
    macros += [rf'\@namedef{{opera@res@{k}}}{{{v}}}' for k,v in sorted(numbers.items())]
    macros += [r'\makeatother']
    (a.output_dir/'results_macros.tex').write_text('\n'.join(macros)+'\n')
    table=render_table({r['contrast']:r for r in panel})
    # Separate filename prevents legacy seven-column tables from being loaded
    # under the new explicit candidate/reference column headings.
    (a.output_dir/'panel_comparisons_rows.tex').write_text(table)
    (a.output_dir/'panel_comparisons_heading.tex').write_text(
        r'\def\panelcomparisonheading{'+RESULT_ROWS[0][1]+'}\n')
    (a.output_dir/'paper_numbers.json').write_text(json.dumps(dict(split='held_out',seed=42,
        source_bundle=str(a.bundle),source_manifest_sha256=manifest['source_manifest_sha256'],
        source_hashes=manifest['files'],numbers=numbers,
        caution='Generic O* aliases identify cohort-encoder OPERA in this source. '
                'Table reports differences and intervals; headline Holm results remain separate.'),indent=2))
    print(f'Generated {len(numbers)} values and {len(panel)} table comparisons in {a.output_dir}')


if __name__=='__main__':
    main()
