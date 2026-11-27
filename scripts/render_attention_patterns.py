"""Render aggregate functional-attention matrices with the paper's taxonomy recipe."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from opera.run.paper_story_figures import thesis_style, _save
from opera.visualization import opera_design_figures as od


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    meta=json.loads((a.inputs/'manifest.json').read_text())
    if meta['status'] != 'complete':raise ValueError('Need the completed full-split export')
    summary=pd.read_csv(a.inputs/'outcome_attention_summary.csv')
    groups=summary.set_index('outcome').endpoint_family
    primary=pd.read_csv(a.inputs/'attention_cosine.csv',index_col=0)
    order=od._within_group_order(primary,groups,od._domain_order())
    thesis_style()
    for key,title,centred in [('attention_cosine','Similarity of outcome attention patterns',False),
                              ('centred_attention_cosine','Similarity of departures from uniform attention',True)]:
        matrix=pd.read_csv(a.inputs/(key+'.csv'),index_col=0)
        fig=plt.figure(figsize=(7.2,7.2))
        ax=fig.add_axes([.24,.19,.62,.62]);cax=fig.add_axes([.41,.085,.3,.012])
        image=od._taxonomy_heatmap(ax,matrix,groups,order=order,label_fn=od._outcome_label,
            group_label=od._short_domain,fontsize=4.6,
            cmap=od.SIMILARITY_CMAP if centred else 'YlGnBu')
        if not centred:image.set_norm(Normalize(0,1))
        od._panel_title(ax,'A' if not centred else 'B',title,fontsize=9.5)
        od._stat_line(ax,meta.get('clinical_group_statistics',{}).get(key),'domain')
        cb=fig.colorbar(image,cax=cax,orientation='horizontal')
        cb.set_ticks([-1,0,1] if centred else [0,.5,1]);cb.outline.set_visible(False)
        cb.set_label('Mean within-patient, corresponding-head cosine similarity',fontsize=7)
        fig.text(.24,.025,f"Held-out patients: {meta['patients']:,}; heads compared before averaging; attention is not attribution",
                 fontsize=6.3)
        _save(fig,a.output/('supp_'+key))


if __name__=='__main__':main()
