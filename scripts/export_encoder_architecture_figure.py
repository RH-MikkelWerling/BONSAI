"""Export a presentation-ready paired-effect figure from saved aggregate tables."""
import argparse
from pathlib import Path
import pandas as pd


def headline_rows(path):
    frame=pd.read_csv(path,low_memory=False)
    for column,value in [('metric','auc'),('stratum','all_cells'),('horizon_days',0),('seed',42)]:
        frame=frame.loc[frame[column].eq(value)]
    if frame.contrast.duplicated().any():
        raise ValueError('Expected one row per contrast; do not mix splits or estimands')
    return frame.set_index('contrast')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--effects',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--label',required=True,help='Explicitly name tuning or held-out')
    args=p.parse_args()
    frame=headline_rows(args.effects)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('Encoder gains over the same local architecture',[
        ('Joint: national','EXT_J_E_VS_J_L'),('Joint: adapted','EXT_J_EA_VS_J_L'),
        ('OPERA: national','EXT_R_E_VS_J_OA_OWN_L'),('OPERA: adapted','EXT_R_EA_VS_J_OA_OWN_L')]),
        ('OPERA minus joint, same encoder',[
        ('Local','EXT_ARCH_L'),('National','EXT_ARCH_E'),('Adapted','EXT_ARCH_EA')]),
        ('Adapted models minus pooled XGBoost',[
        ('Joint','EXT_J_EA_VS_XP'),('Own corrections','EXT_R_PF_OFF_EA_VS_XP'),
        ('OPERA','EXT_R_EA_VS_XP'),('Routing','EXT_R_AB_EA_VS_XP')])]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    exported=[]
    for ax,(title,items) in zip(axes,panels):
        labels=[]
        for name,code in items:
            if code not in frame.index:
                continue
            row=frame.loc[code];y=len(labels);labels.append(name)
            ax.hlines(y,row.lower,row.upper,color='#355f8d',linewidth=2)
            ax.plot(row.gain,y,'o',color='#355f8d')
            exported.append(dict(panel=title,label=name,contrast=code,**row.to_dict()))
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
        ax.axvline(0,color='grey',linewidth=.8);ax.set_title(title,fontsize=10)
        ax.set_xlabel('Mean AUC difference (paired 95% interval)')
    fig.suptitle(args.label)
    fig.tight_layout()
    fig.savefig(args.output_dir/'encoder_architecture.pdf',bbox_inches='tight')
    fig.savefig(args.output_dir/'encoder_architecture.png',dpi=200,bbox_inches='tight')
    pd.DataFrame(exported).to_csv(args.output_dir/'figure_data.csv',index=False)
    (args.output_dir/'README.md').write_text(
        'Aggregate paired differences, seed 42, equal-cell all-horizon AUC. '
        'Report tuning separately from held-out; provisional early effects lack later annotations. '
        'Panel A uses each architecture’s own local baseline; it is not absolute AUC. '
        'Intervals spanning zero do not establish equivalence.\n')
    print(args.output_dir)


if __name__=='__main__':
    main()
