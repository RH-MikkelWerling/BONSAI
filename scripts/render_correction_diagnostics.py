"""Existing correction-arrow design, separate sources and null control, all at 1x."""
import argparse,json
from pathlib import Path
import pandas as pd
from opera.run.paper_story_figures import thesis_style,_save
from opera.visualization import opera_design_figures as od
from opera.visualization.outcome_names import outcome_display_name


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();meta=json.loads((a.inputs/'manifest.json').read_text())
    if meta['status']!='complete':raise ValueError('Incomplete diagnostic')
    arrows=pd.read_csv(a.inputs/'correction_arrows.csv');density=pd.read_csv(a.inputs/'correction_arrow_density.csv')
    arrows.outcome=arrows.outcome.map(outcome_display_name)
    meta['outcomes']=[outcome_display_name(x) for x in meta['outcomes']]
    thesis_style()
    od.COMPONENT_STYLE.update({'isolated_global':('#2D3A8C','Global correction alone'),
        'isolated_disease':('#3E8A84','Diagnosis correction alone'),
        'isolated_outcome':('#B18B4D','Outcome correction alone'),
        'projection_null':('#8A93A0','Unchanged shared representation: projection null')})
    for stem,keys in [('supp_correction_arrows_anchored',['total','reading','corrections']),
                      ('supp_correction_sources',['isolated_global','isolated_disease','isolated_outcome']),
                      ('supp_correction_projection_null',['projection_null'])]:
        _save(od.figure_correction_arrows(arrows,density,meta,components=keys,arrow_scale=1),a.output/stem)


if __name__=='__main__':main()
