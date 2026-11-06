import numpy as np
import pytest
import torch

from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.routes import candidate_pair
from wnp.soft_dp import trajectory_log_prob
from wnp.route_choice import (
    route_features, choice_log_probs, predict_route, draw_choices, choice_metrics,
)


@pytest.mark.parametrize('weights', [[1,.7,1.2], [1,1,1], [1,0,0]])
def test_conditional_probabilities_match_full_dp_and_padding(weights):
    env = GridMDP(np.zeros((2,2), dtype=bool), (0,0,0), (1,1), 4)
    f = feature_table(env)
    w = torch.tensor(weights, dtype=torch.float64)
    pair = candidate_pair(env, 8)
    totals = route_features(env, f, pair)
    got = choice_log_probs(totals, w)
    full = trajectory_log_prob(env, f, w, pair.routes, 30.)
    torch.testing.assert_close(got, full.log_softmax(0), atol=1e-10, rtol=0)
    assert torch.equal(totals, route_features(env, f, candidate_pair(env, 10)))
    assert candidate_pair(env, 8) == pair


def test_equal_features_extreme_logits_and_prediction_ties():
    totals = torch.ones((2,3), dtype=torch.float64)
    assert choice_log_probs(totals, totals[0]).exp().tolist() == [.5,.5]
    totals[1] = 10000
    lp = choice_log_probs(totals, totals[0])
    assert torch.isfinite(lp).all() and lp[1] < -20000
    for shortest in (0,1):
        assert predict_route(torch.tensor([1.,1.+5e-13], dtype=torch.float64), shortest) == shortest
    assert predict_route(torch.tensor([2.,1.], dtype=torch.float64), 0) == 1


def test_metrics_use_shared_labels_binary_brier_and_cost_regret():
    lp = torch.tensor([.25,.75], dtype=torch.float64).log()
    labels = np.array([0,1,1,1])
    metrics = choice_metrics(lp, 1, labels, lp, torch.tensor([2.,5.],dtype=torch.float64))
    assert metrics['sampled_accuracy'] == .75
    assert metrics['expected_accuracy'] == .75
    assert metrics['brier'] == .1875
    assert metrics['expected_regret'] == 3.
    assert metrics['nll'] == pytest.approx(-(.25*np.log(.25)+.75*np.log(.75)))
    deterministic = choice_metrics(None, 0, labels, lp, torch.tensor([2.,5.],dtype=torch.float64))
    assert deterministic['nll'] is None and deterministic['brier'] is None
    assert deterministic['sampled_accuracy'] == .25
    invalid = choice_metrics(lp, None, labels, lp, torch.tensor([2.,5.],dtype=torch.float64))
    assert invalid['sampled_accuracy'] == 0 and invalid['expected_accuracy'] == 0
    first = draw_choices(lp, 123)
    np.testing.assert_array_equal(first, draw_choices(lp,123))
    expected = np.random.default_rng(np.random.SeedSequence([420001,123])).choice(2,64,p=[.25,.75])
    np.testing.assert_array_equal(first,expected)


@pytest.mark.parametrize('bad', [torch.ones(2), torch.tensor([float('nan'),0.],dtype=torch.float64),
                               torch.tensor([0.,0.],dtype=torch.float64)])
def test_invalid_log_probabilities_rejected_not_normalized(bad):
    with pytest.raises(ValueError):
        draw_choices(bad, 11)


def test_invalid_totals_weights_labels_and_seeds_rejected():
    totals = torch.ones((2,3),dtype=torch.float64)
    for w in (torch.ones(3), torch.ones(2,dtype=torch.float64),
              torch.tensor([1.,float('inf'),1.],dtype=torch.float64)):
        with pytest.raises(ValueError): choice_log_probs(totals,w)
    lp = torch.tensor([.5,.5],dtype=torch.float64).log()
    for labels in (np.array([]), np.array([2]), np.array([.5])):
        with pytest.raises(ValueError): choice_metrics(lp,0,labels,lp,totals[0,:2])
    with pytest.raises(ValueError): draw_choices(lp,True)
