from itertools import permutations
from fractions import Fraction

import numpy as np
import pytest
import torch

from wnp.demos import enumerate_trajectories, validate_trajectories
from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.routes import feature_priority_route, tradeoff_candidate_pair, select_tradeoff_pair
from wnp.route_design import audit_choice_design


def test_all_six_priority_routes_match_exhaustive_feature_order():
    env=GridMDP(np.zeros((2,2),dtype=bool),(0,0,0),(1,1),4)
    features=feature_table(env)
    complete=[r for r in enumerate_trajectories(env,env.start_state,7) if r.arrival_step is not None]
    def key(route,order):
        totals=features[list(route.states[:-1]),list(route.actions)].sum(0).tolist()
        return tuple(totals[j] for j in order)+(route.arrival_step,route.actions[:route.arrival_step])
    for order in permutations(range(3)):
        result=feature_priority_route(env,features,order)
        assert key(result,order)==min(key(r,order) for r in complete)
        assert result.arrival_step==len(result.actions)


def test_selection_requires_competing_features_not_a_clearance_only_race():
    # Selection must not return either a dominated route or a feature-equal pair.
    totals=np.array([[10.,2.,8.],[10.,2.,3.],[12.,1.,1.],[11.,4.,9.]])
    i,j=select_tradeoff_pair(totals,[(0,),(1,),(2,),(3,)])
    assert (i,j)==(1,2)
    delta=totals[j]-totals[i]
    assert (delta>1e-10).any() and (delta< -1e-10).any()
    for bad in (np.array([[10.,2.,3.],[10.,2.,8.]]),np.ones((2,3))):
        with pytest.raises(ValueError,match='trade'):
            select_tradeoff_pair(bad,[(0,),(1,)])


def test_priority_search_uses_clearance_and_never_mutates_scoring_graph():
    occupied=np.zeros((4,5),dtype=bool); occupied[1,2]=True
    env=GridMDP(occupied,(1,0,0),(1,4),4)
    features=feature_table(env)
    before=env.next_state.clone(),env.legal.clone(),features.clone()
    pair,metadata=tradeoff_candidate_pair(env,features,40)
    validate_trajectories(env,pair.routes)
    totals=np.array(metadata['selected_feature_totals'])
    delta=totals[1]-totals[0]
    assert (delta>1e-10).any() and (delta< -1e-10).any()
    assert len(metadata['recipes'])==6
    assert metadata['uses_preference_weights'] is False
    assert torch.equal(before[0],env.next_state) and torch.equal(before[1],env.legal)
    assert torch.equal(before[2],features)
    assert tradeoff_candidate_pair(env,features,40)==(pair,metadata)


def test_design_audit_rejects_the_original_clearance_only_experiment():
    rows=[dict(seed=i,feature_difference=[0.,0.,float(i+1)]) for i in range(10)]
    report=audit_choice_design(rows,10)
    assert not report['passed']
    assert report['free_feature_rank']==1
    assert report['profile_disagreement_fraction']==0
    assert not report['gates']['distance_varies'] and not report['gates']['turning_varies']


def test_design_audit_accepts_independent_opposing_costs_without_needing_model_wins():
    rows=[dict(seed=1,feature_difference=[2.,-2.,0.]),
          dict(seed=2,feature_difference=[2.,0.,-2.]),
          dict(seed=3,feature_difference=[0.,2.,-2.])]
    report=audit_choice_design(rows,3)
    assert report['passed'] and report['free_feature_rank']==2
    assert report['profile_disagreement_fraction']==1
    assert report['normalized_design_eigenvalues'][0]>=.01
    assert 'learned_accuracy' not in report
    missing=audit_choice_design(rows[:1],3)
    assert not missing['passed'] and missing['requested_layouts']==3


def test_feature_library_does_not_use_external_truth_or_fitted_weights(monkeypatch):
    from dataclasses import replace
    import wnp.offered_routes as runner
    occupied=np.zeros((4,5),dtype=bool); occupied[1,2]=True
    env=GridMDP(occupied,(1,0,0),(1,4),4)
    monkeypatch.setattr(runner,'generate_recorded_layout',lambda seed,cfg:(env,1))
    cfg=replace(runner.OfferedRouteConfig(),shape=(4,5),start=(1,0,0),goal=(1,4),
                horizon=40,terminal_penalty=80.)
    first,_=runner._preflight(10000,cfg,[{'name':'oracle','weights':[1.,.7,1.2]},
                                     {'name':'learned_11','weights':[1.,.1,.1]}])
    second,_=runner._preflight(10000,cfg,[{'name':'oracle','weights':[1.,3.,.1]},
                                      {'name':'learned_11','weights':[1.,.1,3.]}])
    assert first['candidate_construction']==second['candidate_construction']
    assert first['routes']==second['routes']


def test_feature_priority_secondary_objectives_count_turns_on_clearance_ties():
    env=GridMDP(np.zeros((2,2),dtype=bool),(0,0,0),(1,1),4)
    f=feature_table(env)
    route=feature_priority_route(env,f,(2,1,0))
    assert route.actions==(0,2,0)
    assert len(route.actions)==3
    # Start at goal must terminate rather than explore its zero-cost continuation.
    goal=GridMDP(np.zeros((1,1),dtype=bool),(0,0,0),(0,0),4)
    assert feature_priority_route(goal,feature_table(goal),(2,1,0)).actions==()


def test_tradeoff_pair_rejects_unreachable_and_feature_identical_libraries():
    for occupied,goal in ((np.zeros((2,2),dtype=bool),(1,1)),
                          (np.array([[False,True,False]]),(0,2))):
        env=GridMDP(occupied,(0,0,0),goal,4)
        with pytest.raises(ValueError,match='trade'):
            tradeoff_candidate_pair(env,feature_table(env),8)


def test_exact_clearance_tie_preserves_better_secondary_turning_objective():
    occupied=np.random.default_rng(325).random((8,8))<.2
    occupied[0,0]=occupied[7,7]=False
    env=GridMDP(occupied,(0,0,0),(7,7),4)
    features=feature_table(env)
    route=feature_priority_route(env,features,(2,0,1))
    alternative=(0,2,0,0,0,0,0,1,0,0,0,0,0,2,0,0,1,0)
    def exact_key(actions):
        state=env.start_state; totals=[Fraction(0) for _ in range(3)]
        for action in actions:
            assert env.legal[state,action]
            for j in range(3): totals[j]+=Fraction.from_float(features[state,action,j].item())
            state=int(env.next_state[state,action])
        assert env.is_goal[state]
        return totals[2],totals[0],totals[1],len(actions),actions
    assert exact_key(route.actions)<=exact_key(alternative)
    assert len(route.actions)==18


def test_candidate_failure_preserves_all_recipe_and_library_diagnostics():
    from wnp.routes import CandidateConstructionError
    env=GridMDP(np.zeros((2,2),dtype=bool),(0,0,0),(1,1),4)
    with pytest.raises(CandidateConstructionError) as caught:
        tradeoff_candidate_pair(env,feature_table(env),8)
    evidence=caught.value.metadata
    assert len(evidence['recipes'])==6
    assert all(r['status']=='eligible' and r['actions'] for r in evidence['recipes'])
    assert evidence['library']
    assert evidence['selected_library_indices'] is None
    assert evidence['failure_reason']
