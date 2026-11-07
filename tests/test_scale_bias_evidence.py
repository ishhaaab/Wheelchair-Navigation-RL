"""Run-once large regression; excluded from routine suite unless explicitly enabled."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from wnp.demos import sample
from wnp.irl import fit
from wnp.recovery import RecoveryConfig, _fixture
from wnp.soft_dp import plan


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("WNP_RUN_SCALE_REGRESSION") != "1",
                    reason="run-once 65536-demo scale regression")
def test_doubled_scale_recovers_ratios_only_with_three_free():
    torch.set_num_threads(1)
    env, features = _fixture(RecoveryConfig())
    truth = torch.tensor([2., 1.4, 2.4], dtype=torch.float64)
    records = sample(env, plan(env, features, truth, 24, 30),
                     [env.start_state] * 65536, 20261106)
    fixed = fit(env, features, records)
    if os.environ.get("WNP_SCALE_EVIDENCE"):
        output = Path(os.environ["WNP_SCALE_EVIDENCE"])
        output.mkdir(parents=True, exist_ok=False)
        with (output / "fixed_fit.json").open("x") as stream:
            json.dump(fixed, stream, indent=2, allow_nan=False)
    assert fixed["success"], fixed
    assert all(1e-3 <= w <= 10 for w in fixed["weights"][1:])
    assert np.max(np.abs(np.asarray(fixed["weights"])[1:] - [.7, 1.2])) > .20
    recovered = fit(env, features, records, parameterization="three_free")
    if os.environ.get("WNP_SCALE_EVIDENCE"):
        with (output / "three_free_fit.json").open("x") as stream:
            json.dump(recovered, stream, indent=2, allow_nan=False)
    assert recovered["success"], recovered
    ratios = np.asarray(recovered["weights"])[1:] / recovered["weights"][0]
    assert np.all(np.abs(ratios - [.7, 1.2]) <= .12), (recovered, ratios)
    assert abs(recovered["weights"][0] - 2) <= .20
