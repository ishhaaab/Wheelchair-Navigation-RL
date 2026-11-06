"""Equal-layout summaries and paired whole-layout uncertainty, not trial pooling."""

import math

import numpy as np

from wnp.env import _integer


LEARNED_MODELS = tuple(f'learned_{seed}' for seed in (11,29,47,83,101))
MODEL_NAMES = ('shortest','fixed','soft_distance','oracle',*LEARNED_MODELS)
METRICS = ('sampled_accuracy','expected_accuracy','nll','brier','expected_regret')


def _metrics_valid(models):
    if not isinstance(models,dict) or set(models) != set(MODEL_NAMES):
        return False
    for name in MODEL_NAMES:
        for metric in METRICS:
            value = models[name].get(metric)
            if value is None and name == 'shortest' and metric in ('nll','brier'):
                continue
            if (isinstance(value,bool) or not isinstance(value,(int,float))
                    or not math.isfinite(value) or value < 0):
                return False
            if metric in ('sampled_accuracy','expected_accuracy','brier') and value > 1:
                return False
    return True


def summarize_layouts(records: list[dict], requested_seeds: tuple[int,...],
                      bootstrap_refits: int = 10000, bootstrap_seed: int = 920001) -> dict:
    if (not requested_seeds or any(not _integer(s) or s < 0 for s in requested_seeds)
            or len(set(requested_seeds)) != len(requested_seeds)):
        raise ValueError('requested seeds must be distinct nonnegative integers')
    if (not _integer(bootstrap_refits) or bootstrap_refits < 1
            or not _integer(bootstrap_seed) or bootstrap_seed < 0):
        raise ValueError('invalid bootstrap refits or seed')
    by_seed = {}
    for row in records:
        seed = row['seed']
        if not _integer(seed) or seed not in requested_seeds or seed in by_seed:
            raise ValueError('duplicate or unrequested layout seed')
        by_seed[seed] = row
    failures = []
    for seed in requested_seeds:
        row = by_seed.get(seed)
        if row is None or row.get('status') != 'passed' or not _metrics_valid(row.get('model_metrics')):
            failures.append({'seed':int(seed), 'reason': 'missing layout' if row is None else
                             row.get('reason','invalid or missing model metrics')})
    result = {'requested_layouts':len(requested_seeds),
              'successful_layouts':len(requested_seeds)-len(failures),
              'failed_layouts':len(failures), 'failures':failures,
              'model_means':None,'training_seed_accuracy':None,
              'comparisons':None,'superior_to_both':False,
              'bootstrap':{'unit':'layout','refits':int(bootstrap_refits),'seed':int(bootstrap_seed),
                           'percentiles':[1.25,98.75],'method':'linear',
                           'training_uncertainty_included':False}}
    if failures:
        return result
    rows = [by_seed[s]['model_metrics'] for s in requested_seeds]
    means = {}
    for name in MODEL_NAMES:
        means[name] = {metric: None if rows[0][name][metric] is None else
                       float(np.mean([r[name][metric] for r in rows])) for metric in METRICS}
    means['learned_mean'] = {metric:float(np.mean([means[name][metric] for name in LEARNED_MODELS]))
                             for metric in METRICS}
    accuracy = np.array([[np.mean([r[name]['sampled_accuracy'] for name in LEARNED_MODELS]),
                          r['shortest']['sampled_accuracy'],r['fixed']['sampled_accuracy']] for r in rows])
    differences = accuracy[:,:1]-accuracy[:,1:]
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0,len(rows),size=(int(bootstrap_refits),len(rows)))
    intervals = np.percentile(differences[indices].mean(1),[1.25,98.75],axis=0,method='linear').T
    comparisons = {name:{'difference':float(differences[:,j].mean()),'interval':intervals[j].tolist()}
                   for j,name in enumerate(('shortest','fixed'))}
    result.update(model_means=means, comparisons=comparisons,
                  training_seed_accuracy={name:means[name]['sampled_accuracy'] for name in LEARNED_MODELS},
                  superior_to_both=bool((intervals[:,0]>0).all()))
    return result
