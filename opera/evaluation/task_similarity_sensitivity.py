"""Definition-overlap sensitivity for descriptive task-similarity matrices."""
import re
import numpy as np

ALIASES = {
    'infection_diagnosis_or_antimicrobial_use_composite':'infections_all_definitions',
    'iv_antimicrobial_use_proxy':'infections_IV_based',
    'prescription_antimicrobial_use_proxy':'infections_prescription_based',
}


def base_endpoint(name):
    name = ALIASES.get(name, name)
    if name.startswith('alp_elevation_legacy_tier'):
        return 'alp_elevation_legacy'
    if name == 'sodium_below_125_mmol_l':return 'hyponatremia'
    return re.sub(r'_g[234]plus$', '', name)


def overlap_mask(names, composites):
    """Exclude same-endpoint variants and overlapping composite components.

    Masks are fixed by definitions, never by clinical labels or similarities.
    """
    def atoms(name, stack=()):
        name = ALIASES.get(name, name)
        if name in stack:raise ValueError('Cyclic composite definitions')
        if name in composites:
            return set().union(*(atoms(x, (*stack,name)) for x in composites[name]))
        return {base_endpoint(name)}
    sets = [atoms(n) for n in names]
    return np.array([[bool(a & b) for b in sets] for a in sets])


def within_between(matrix, labels, eligible=None, permutations=2000, seed=0):
    matrix = np.asarray(matrix, float);labels = np.asarray(labels)
    if matrix.shape != (len(labels),len(labels)):raise ValueError('Unaligned similarity matrix')
    keep = np.triu(np.ones(matrix.shape,dtype=bool),1) & np.isfinite(matrix)
    if eligible is not None:keep &= eligible
    def means(lab):
        same = lab[:,None] == lab[None,:]
        a,b = matrix[keep & same],matrix[keep & ~same]
        if not len(a) or not len(b):return None
        return float(a.mean()),float(b.mean()),len(a),len(b)
    observed = means(labels)
    if observed is None:return {'status':'insufficient_pairs'}
    within,between,nwithin,nbetween = observed
    gap = within-between;rng=np.random.default_rng(seed);exceed=valid=0
    for _ in range(permutations):
        m=means(rng.permutation(labels))
        if m is not None:valid+=1;exceed+=m[0]-m[1]>=gap
    return dict(status='estimated',within=within,between=between,difference=gap,
        n_within_pairs=nwithin,n_between_pairs=nbetween,
        p_permutation=(exceed+1)/(valid+1) if valid else float('nan'),
        valid_permutations=valid,n_units=len(labels))


def sensitivities(matrix, names, families, composites, permutations=2000):
    """Primary plus two sensitivities; the strict one removes all composites
    and gives each underlying endpoint one unit regardless of grade count.
    """
    if len(set(names)) != len(names):raise ValueError('Repeated outcome name')
    labels=[families[n] for n in names]
    related=overlap_mask(names,composites)
    rows=[]
    for label,eligible in [('original',None),('exclude_definition_overlap',~related)]:
        rows.append({'analysis':label,**within_between(matrix,labels,eligible,permutations)})
    groups={}
    for i,name in enumerate(names):
        if ALIASES.get(name,name) in composites:continue
        groups.setdefault(base_endpoint(name),[]).append(i)
    units=list(groups);collapsed=np.full((len(units),len(units)),np.nan);group_labels=[]
    for i,name in enumerate(units):
        lab={labels[k] for k in groups[name]}
        if len(lab)!=1:raise ValueError('Grade variants cross clinical families')
        group_labels.append(next(iter(lab)))
        for j,other in enumerate(units):
            block=np.asarray(matrix)[np.ix_(groups[name],groups[other])]
            if np.isfinite(block).any():collapsed[i,j]=np.nanmean(block)
    rows.append({'analysis':'no_composites_one_unit_per_endpoint',
                 **within_between(collapsed,group_labels,permutations=permutations)})
    return rows,related,groups
