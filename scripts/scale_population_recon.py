"""Exact population reconnaissance, before freezing the scale-bias regression.

No sampling. Run once into a new output directory, using legacy inference.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from wnp.irl import classify_fit, physical_weights
from wnp.recovery import RecoveryConfig, _fixture
from wnp.soft_dp import expected_features, plan
from wnp.verification import terminal_margin_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    protected = list((root / "results").rglob("*"))
    protected += [root / "src/wnp" / f"{name}.py"
                  for name in ("env", "features", "soft_dp", "verification", "irl")]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in protected if p.is_file() and args.output not in p.parents}
    with (args.output / "baseline_hashes.json").open("x") as stream:
        json.dump(hashes, stream, indent=2, sort_keys=True)
    torch.set_num_threads(1)
    config = RecoveryConfig()
    env, features = _fixture(config)
    rows = []
    for scale in (.5, 1., 2.):
        truth = torch.tensor([1., .7, 1.2], dtype=torch.float64) * scale
        truth_plan = plan(env, features, truth, config.horizon, config.terminal_penalty)
        mean = expected_features(env, features, truth_plan, [env.start_state]).detach()
        mass = torch.zeros(len(env.states), dtype=torch.float64)
        mass[env.start_state] = 1
        for t in range(config.horizon):
            flow = mass[:, None] * truth_plan.log_policy[t].detach().exp()
            mass = torch.zeros_like(mass).scatter_add_(0, env.next_state.flatten(), flow.flatten())
        unfinished = mass[~env.is_goal].sum().item()
        raw = torch.full((2,), math.log(math.expm1(1.)), dtype=torch.float64,
                         requires_grad=True)
        optimizer = torch.optim.LBFGS([raw], max_iter=150, tolerance_grad=1e-9,
            tolerance_change=1e-14, line_search_fn="strong_wolfe")
        evaluations = 0

        def objective(weights):
            return (mean @ weights + config.terminal_penalty * unfinished
                    + plan(env, features, weights, config.horizon,
                           config.terminal_penalty).values[0, env.start_state])

        def closure():
            nonlocal evaluations
            evaluations += 1
            optimizer.zero_grad()
            value = objective(physical_weights(raw))
            value.backward()
            return value

        optimizer.step(closure)
        weights = physical_weights(raw).detach().requires_grad_()
        value = objective(weights)
        gradient = torch.autograd.grad(value, weights)[0]
        raw_gradient = gradient[1:] * raw.detach().sigmoid()
        iterations = optimizer.state[raw]["n_iter"]
        covariance = torch.autograd.functional.hessian(lambda w: plan(env, features,
            w, config.horizon, config.terminal_penalty).values[0, env.start_state], truth)
        eigenvalues = torch.linalg.eigvalsh((covariance + covariance.T) / 2)
        # Delta-method joint coordinates [scale, turn/distance, clearance/distance].
        jacobian = torch.tensor([[1., 0., 0.],
            [-truth[1]/truth[0]**2, 1/truth[0], 0.],
            [-truth[2]/truth[0]**2, 0., 1/truth[0]]], dtype=torch.float64)
        asymptotic = jacobian @ torch.linalg.inv(covariance) @ jacobian.T / 65536
        row = {"truth": truth.tolist(), "scale": scale,
            "fixed_population_weights": weights.detach().tolist(),
            "fixed_population_ratios": (weights[1:]/weights[0]).detach().tolist(),
            "signed_ratio_bias": (weights[1:]/weights[0]-truth[1:]/truth[0]).detach().tolist(),
            "cross_entropy": value.item(), "raw_gradient": raw_gradient.tolist(),
            "physical_gradient": gradient.tolist(), "iterations": iterations,
            "closure_evaluations": evaluations,
            "failure_reason": classify_fit(value.item(), raw_gradient.tolist(),
                                            weights[1:].tolist(), iterations),
            "covariance": covariance.tolist(), "eigenvalues": eigenvalues.tolist(),
            "condition_number": (eigenvalues[-1]/eigenvalues[0]).item(),
            "completion_probability": 1-unfinished,
            "predicted_std_n65536_scale_ratios": asymptotic.diagonal().sqrt().tolist(),
            "terminal_margin": terminal_margin_diagnostic(env, features, truth,
                config.horizon, config.terminal_penalty)}
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    budget_rows = [{"weights": w, "budget": config.terminal_penalty + config.horizon * (
        (features @ torch.tensor(w, dtype=torch.float64)).abs().max().item() + math.log(3)),
        "limit": 1000.} for w in ([1., .7, 1.2], [2., 1.4, 2.4], [10., 10., 10.])]
    with (args.output / "population.json").open("x") as stream:
        json.dump({"method": "Exact expected-features cross-entropy, no sampling",
                   "rows": rows, "budgets": budget_rows}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
