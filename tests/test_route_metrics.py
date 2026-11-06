import copy

import numpy as np
import pytest

from wnp.route_metrics import summarize_layouts


def _record(seed, learned, shortest, fixed):
    def metrics(accuracy):
        return dict(sampled_accuracy=accuracy, expected_accuracy=accuracy,
                    nll=.5, brier=.2, expected_regret=0.)
    models = {key: metrics(value) for key,value in
              [('shortest',shortest),('fixed',fixed),('oracle',.9),('soft_distance',shortest)]}
    models['shortest'].update(nll=None,brier=None)
    models.update({f'learned_{seed}':metrics(learned) for seed in (11,29,47,83,101)})
    return dict(seed=seed,status='passed',model_metrics=models)


def test_aggregation_uses_equal_layouts_and_paired_bootstrap():
    records = [_record(10000,.75,.25,.5),_record(10001,.5,.5,.25)]
    got = summarize_layouts(records,(10000,10001),bootstrap_refits=100,bootstrap_seed=7)
    assert got['requested_layouts']==2 and got['failed_layouts']==0
    assert got['model_means']['learned_mean']['sampled_accuracy']==.625
    assert len(got['training_seed_accuracy'])==5
    rng=np.random.default_rng(7)
    indices=rng.integers(0,2,size=(100,2))
    paired=np.array([[.5,.25],[0.,.25]])
    intervals=np.percentile(paired[indices].mean(axis=1),[1.25,98.75],axis=0).T
    np.testing.assert_array_equal(got['comparisons']['shortest']['interval'],intervals[0])
    np.testing.assert_array_equal(got['comparisons']['fixed']['interval'],intervals[1])
    assert got['comparisons']['shortest']['difference']==.25
    assert not got['superior_to_both']
    assert got['bootstrap']['unit']=='layout' and got['bootstrap']['refits']==100
    assert got == summarize_layouts(records,(10000,10001),100,7)


def test_single_layout_and_zero_difference_do_not_imply_superiority():
    got=summarize_layouts([_record(1,.5,.5,.5)],(1,),10)
    assert got['comparisons']['fixed']['interval']==[0.,0.]
    assert not got['superior_to_both']


@pytest.mark.parametrize('failure',['missing_layout','missing_model','invalid_metric','failed_layout'])
def test_failed_studies_keep_requested_denominator_and_no_success_only_comparisons(failure):
    records=[_record(1,.8,.2,.3),_record(2,.9,.2,.3)]
    if failure=='missing_layout': records.pop()
    elif failure=='missing_model': del records[1]['model_metrics']['learned_11']
    elif failure=='invalid_metric': records[1]['model_metrics']['fixed']['nll']=float('nan')
    else: records[1]=dict(seed=2,status='failed',reason='no alternative')
    got=summarize_layouts(records,(1,2),10)
    assert got['requested_layouts']==2 and got['failed_layouts']==1
    assert got['comparisons'] is None and got['model_means'] is None
    assert got['failures'][0]['seed']==2


def test_duplicate_unrequested_seeds_and_bad_bootstrap_rejected():
    record=_record(1,.5,.4,.3)
    for rows,seeds in (([record,copy.deepcopy(record)],(1,)),([record],(2,))):
        with pytest.raises(ValueError): summarize_layouts(rows,seeds)
    with pytest.raises(ValueError): summarize_layouts([record],(1,),0)
