"""Statistical-outcome simulation only; these tests never generate layouts."""

import json
from statistics import NormalDist

import numpy as np
import pytest

from wnp.transfer_power import choose_n, simulate_cell, wilson, write_report


def test_wilson_known_boundaries():
    low, high = wilson(0, 100)
    assert low == pytest.approx(0, abs=1e-15)
    assert high == pytest.approx(0.03699349820698568)
    low, high = wilson(100, 100)
    assert low == pytest.approx(1 - 0.03699349820698568)
    assert high == pytest.approx(1)


def test_simulation_matches_direct_paired_outcome_calculation():
    n, reps, seed = 12, 20, 31
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((reps, n, 1))
    separate = rng.standard_normal((reps, n, 6))
    outcomes = .25 + 1.5 * (common + separate) / np.sqrt(2)
    reject = outcomes.mean(axis=1) / (outcomes.std(axis=1, ddof=1) / np.sqrt(n)) > (
        NormalDist().inv_cdf(1 - .05 / 6))
    result = simulate_cell(n, .25, 1.5, reps, seed)
    assert result["joint_successes"] == int(reject.all(axis=1).sum())
    assert result["any_successes"] == int(reject.any(axis=1).sum())
    assert result["per_contrast_successes"] == reject.sum(axis=0).tolist()
    assert result == simulate_cell(n, .25, 1.5, reps, seed)


def test_selection_requires_wilson_lower_bound_in_every_scenario():
    cells = [dict(n=n, sd=sd, joint_ci=[lower, 1])
             for n, sd, lower in [(64, .5, .95), (64, 1.5, .89),
                                   (96, .5, .95), (96, 1.5, .90)]]
    assert choose_n(cells, (64, 96), (.5, 1.5)) == 96
    assert choose_n(cells, (64,), (.5, 1.5)) is None


def test_report_never_reuses_an_existing_directory(tmp_path):
    target = tmp_path / "fresh"
    write_report(target, {"selected_n": 96})
    assert json.loads((target / "power.json").read_text())["selected_n"] == 96
    before = (target / "power.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_report(target, {"selected_n": 128})
    assert (target / "power.json").read_bytes() == before
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        write_report(empty, {})


@pytest.mark.parametrize("args", [(1, .25, 1., 10, 1), (4, .25, 0., 10, 1),
                                  (4, .25, 1., 0, 1)])
def test_simulation_rejects_invalid_sizes(args):
    with pytest.raises(ValueError):
        simulate_cell(*args)
