"""Post-hoc aggregate sensitivity; no model/data inference or active-job changes."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml
from opera.evaluation.task_similarity_sensitivity import sensitivities, ALIASES, base_endpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path,help='Task-structure root, one model folder, or functional-attention export')
    p.add_argument('output',type=Path)
    p.add_argument('--permutations',type=int,default=2000)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    definitions=repo/'opera/configs/endpoint_dependencies.yaml'
    composites=yaml.safe_load(definitions.read_text())['endpoint_dependencies']['composites']
    a.output.mkdir(parents=True,exist_ok=True)
    folders=sorted(a.inputs.glob('task_structure_*')) or [a.inputs]
    rows=[];audits=[];sources={str(definitions):hashlib.sha256(definitions.read_bytes()).hexdigest()}
    for folder in folders:
        table=folder/'outcome_specialisation.csv'
        if not table.exists():table=folder/'outcome_attention_summary.csv'
        families=pd.read_csv(table).set_index('outcome').endpoint_family.to_dict()
        sources[str(table)]=hashlib.sha256(table.read_bytes()).hexdigest()
        for key in ['outcome_query_similarity','outcome_correction_similarity','attention_cosine','centred_attention_cosine']:
            source=folder/(key+'.csv')
            if not source.exists():continue
            matrix=pd.read_csv(source,index_col=0)
            if set(matrix.index)!=set(matrix.columns):raise ValueError('Non-square outcome roster')
            matrix=matrix.loc[matrix.index,matrix.index];names=list(matrix.index)
            if any(not isinstance(families.get(n),str) for n in names):raise ValueError('Missing clinical family')
            values,overlap,groups=sensitivities(matrix.to_numpy(),names,families,composites,a.permutations)
            rows.extend(dict(model=folder.name,matrix=key,**r) for r in values)
            if key in ['outcome_query_similarity','attention_cosine']:
                for i,name in enumerate(names):
                    for j in range(i+1,len(names)):
                        if overlap[i,j]:audits.append(dict(model=folder.name,outcome_a=name,outcome_b=names[j],
                            reason='Shared underlying endpoint/grade or composite component'))
            sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        audit=pd.DataFrame([dict(outcome=n,underlying_endpoint=base_endpoint(n),
            composite_removed=ALIASES.get(n,n) in composites,clinical_family=families[n]) for n in families])
        audit.to_csv(a.output/(folder.name+'_definition_audit.csv'),index=False)
    pd.DataFrame(rows).to_csv(a.output/'similarity_sensitivity.csv',index=False)
    pd.DataFrame(audits).to_csv(a.output/'excluded_pairs.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps(dict(
        scope='Exploratory outcome-definition sensitivity; aggregate matrices only',
        permutations=a.permutations,source_sha256=sources,
        aliases=ALIASES,
        primary_sensitivity='Exclude shared-primitive pairs from both within and between comparisons; keep mask fixed during permutations',
        strict_sensitivity='Remove composite outcomes; average variant-to-variant similarities between each pair of underlying endpoints; permute endpoint-level family labels',
        interpretation='Persistence addresses definition overlap as the sole explanation, not biological mechanism or causal benefit'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
