"""Detect offered-route experiments that cannot distinguish free preferences."""

import math

import numpy as np


DIAGNOSTIC_PROFILES = ((1.,.1,.1),(1.,3.,.1),(1.,.1,3.),(1.,3.,3.))


def audit_choice_design(rows, requested_layouts):
    if not isinstance(requested_layouts,int) or isinstance(requested_layouts,bool) or requested_layouts<1:
        raise ValueError('requested_layouts must be a positive integer')
    available=[r for r in rows if 'feature_difference' in r]
    if len(available)>requested_layouts:
        raise ValueError('more design rows than requested layouts')
    differences=np.array([r['feature_difference'] for r in available],dtype=np.float64).reshape(-1,3)
    if not np.isfinite(differences).all():
        raise ValueError('feature differences must be finite')
    tolerance=1e-10
    free=differences[:,1:]
    scales=np.sqrt((free**2).sum(0)/requested_layouts)
    normalized=free/np.where(scales>tolerance,scales,1.)
    gram=normalized.T@normalized/requested_layouts
    eigenvalues=np.linalg.eigvalsh(gram)
    rank=int(np.linalg.matrix_rank(gram,tol=1e-10))
    condition=float(eigenvalues[-1]/eigenvalues[0]) if rank==2 else None
    profiles=np.asarray(DIAGNOSTIC_PROFILES)
    profile_delta=differences@profiles.T
    predictions=(profile_delta< -1e-12).astype(int)
    for i,row in enumerate(available):
        predictions[i,np.abs(profile_delta[i])<=1e-12]=row.get('shortest_index',0)
    disagree=int(np.sum(np.ptp(predictions,axis=1)>0))
    variation=(np.abs(differences)>tolerance).sum(0)
    opposing=((differences>tolerance).any(1)&(differences< -tolerance).any(1))
    required=math.ceil(.2*requested_layouts)
    gates={'all_layouts_available':len(available)==requested_layouts,
           'all_pairs_trade_off':int(opposing.sum())==requested_layouts,
           **{f'{name}_varies':bool(variation[j]>=required) for j,name in enumerate(('distance','turning','clearance'))},
           'free_directions_identified':rank==2 and eigenvalues[0]>=.01 and condition<=100,
           'diagnostic_profiles_disagree':disagree>=required}
    # Oracle diagnostics do not feed candidate construction, pair ranking or gates.
    delta=differences@np.array([1.,.7,1.2])
    log_p0=-np.logaddexp(0.,-delta)
    log_p1=-np.logaddexp(0.,delta)
    p0,p1=np.exp(log_p0),np.exp(log_p1)
    fisher=(free.T*(p0*p1))@free/requested_layouts
    entropy=-(p0*log_p0+p1*log_p1)
    return {'passed':bool(all(gates.values())),'gates':{k:bool(v) for k,v in gates.items()},
            'requested_layouts':requested_layouts,'available_layouts':len(available),
            'varying_feature_counts':dict(zip(('distance','turning','clearance'),map(int,variation))),
            'opposing_pair_count':int(opposing.sum()),'free_feature_rank':rank,
            'normalized_design_eigenvalues':eigenvalues.tolist(),'normalized_design_condition':condition,
            'diagnostic_profiles':[list(w) for w in DIAGNOSTIC_PROFILES],
            'profile_disagreement_count':disagree,'profile_disagreement_fraction':disagree/requested_layouts,
            'oracle_mean_choice_entropy':float(entropy.mean()) if len(entropy) else None,
            'oracle_free_fisher_per_choice':fisher.tolist(),
            'oracle_free_fisher_eigenvalues':np.linalg.eigvalsh(fisher).tolist()}
