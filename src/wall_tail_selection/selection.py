"""Batched fixed-shape scale-family pilot, CPU or CUDA, always float64.

This finite parameter grid makes the Bayesian baseline an EXACT proper
discrete prior. Grid MLEs and continuous-family resolution penalties are
approximations to a future continuous-parameter selector. Do not interpret
a grid-resolution effect as evidence for a new selection principle.
"""

import csv
import hashlib
import json
import math

import numpy as np
import torch

from .geometry import separated_grid_count


def cdf(values, scale, shape):
    if shape is None:
        return -np.expm1(-np.maximum(values, 0) / scale)
    return 1 - np.clip(1 - np.maximum(values, 0) / scale, 0, 1) ** shape


def log_likelihoods(x, scales, shapes):
    """Return [replicate, model, scale] log likelihoods with true -inf support."""
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError("observations must be finite and nonnegative")
    residual = 1 - x[:, :, None] / scales[None, None, :]
    valid = torch.all(residual > 0, dim=1)
    log_residual = torch.log(torch.clamp(residual, min=1e-300)).sum(dim=1)
    n = x.shape[1]
    result = []
    for a in shapes:
        if a is None:
            ll = -n * torch.log(scales)[None, :] - x.sum(dim=1)[:, None] / scales
        else:
            ll = n * math.log(a) - n * torch.log(scales)[None, :] + (a - 1) * log_residual
            ll = torch.where(valid, ll, -torch.inf)
        result.append(ll)
    return torch.stack(result, dim=1)


def run_selection(config, output, device):
    cfg = config["selection"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    dev = torch.device(device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    shapes = cfg["shapes"]
    names = ["exponential" if a is None else f"endpoint_a{a:g}" for a in shapes]
    lower, upper = cfg["scale_bounds"]
    scales_np = np.geomspace(lower, upper, cfg["scale_grid_size"])
    scales = torch.as_tensor(scales_np, dtype=torch.float64, device=dev)
    # Fixed uniform evaluation weight on [0, evaluation_upper]. This is a
    # bounded risk, not whole-line CRPS or an estimated KL divergence.
    nodes, weights = np.polynomial.legendre.leggauss(cfg["evaluation_nodes"])
    ys = (nodes + 1) * cfg["evaluation_upper"] / 2
    weights = weights / 2
    all_cdf_np = np.stack([cdf(ys[None, :], scales_np[:, None], a) for a in shapes])
    all_cdf = torch.as_tensor(all_cdf_np, dtype=torch.float64, device=dev)
    w = torch.as_tensor(weights, dtype=torch.float64, device=dev)
    methods = ["aic", "bic", "geometry", "bayes_grid"]
    fields = ["truth", "n", "replicate", "method", "predictive", "selected_model",
              "cdf_risk", "false_support_mass", "impossible_candidate_count"]
    rows = []
    data_hash = hashlib.sha256()
    diagnostics = {"datasets": 0, "all_endpoint_models_impossible": 0, "grid_mle_at_lower": 0, "grid_mle_at_upper": 0}
    with (output / "replicates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for truth_index, truth in enumerate(shapes):
            truth_cdf = torch.as_tensor(cdf(ys, 1.0, truth), dtype=torch.float64, device=dev)
            for n in cfg["sample_sizes"]:
                penalty = torch.tensor([
                    math.log(separated_grid_count(a, n, lower, upper, cfg["resolution_c"]))
                    for a in shapes], dtype=torch.float64, device=dev)
                for first in range(0, cfg["replicates"], cfg["batch_size"]):
                    ids = range(first, min(first + cfg["batch_size"], cfg["replicates"]))
                    draws = []
                    for rep in ids:
                        rng = np.random.default_rng(np.random.SeedSequence(
                            [config["seed"], truth_index, n, rep]))
                        u = rng.random(n)
                        draws.append(-np.log1p(-u) if truth is None else -np.expm1(np.log1p(-u) / truth))
                    data_hash.update(np.stack(draws).astype("<f8").tobytes())
                    x = torch.as_tensor(np.stack(draws), dtype=torch.float64, device=dev)
                    ll = log_likelihoods(x, scales, shapes)
                    best_ll, best_scale = ll.max(dim=2)
                    evidence = torch.logsumexp(ll, dim=2) - math.log(len(scales_np))
                    possible = torch.isfinite(evidence)
                    assert not torch.isnan(ll).any() and not torch.isposinf(ll).any()
                    assert possible.any(dim=1).all(), "no possible candidate"
                    diagnostics["datasets"] += len(draws)
                    endpoint_ids = [i for i,a in enumerate(shapes) if a is not None]
                    diagnostics["all_endpoint_models_impossible"] += int((~possible[:,endpoint_ids]).all(dim=1).sum().item())
                    diagnostics["grid_mle_at_lower"] += int(((best_scale == 0) & possible).sum().item())
                    diagnostics["grid_mle_at_upper"] += int(((best_scale == len(scales_np)-1) & possible).sum().item())
                    # Invalid models retain zero posterior model weight. Use a
                    # harmless normalized placeholder only for unselected CDFs.
                    safe_ll = torch.where(possible[:, :, None], ll, torch.zeros_like(ll))
                    posterior = torch.softmax(safe_ll, dim=2)
                    mixture_cdf = torch.einsum("bmg,mgy->bmy", posterior, all_cdf)
                    plug_cdf = all_cdf[torch.arange(len(shapes), device=dev)[None, :], best_scale]
                    risks = {
                        "plugin": ((plug_cdf - truth_cdf) ** 2 * w).sum(dim=2),
                        "within_model_bayes": ((mixture_cdf - truth_cdf) ** 2 * w).sum(dim=2),
                    }
                    # All models have one scale parameter. The aic/bic equality
                    # is a control, not independent replication of a finding.
                    scores = {"aic": best_ll - 1, "bic": best_ll - .5 * math.log(n),
                              "geometry": best_ll - penalty, "bayes_grid": evidence}
                    choices = {method: score.argmax(dim=1).cpu().numpy() for method, score in scores.items()}
                    for chosen in choices.values():
                        assert possible[torch.arange(len(draws), device=dev), torch.as_tensor(chosen,device=dev)].all(), "impossible model selected"
                    scales_hat = scales_np[best_scale.cpu().numpy()]
                    risks_np = {key: value.cpu().numpy() for key, value in risks.items()}
                    bma_weights = torch.softmax(evidence, dim=1)
                    assert (bma_weights[~possible] == 0).all()
                    assert torch.isfinite(bma_weights).all()
                    bma = torch.einsum("bm,bmy->by", bma_weights, mixture_cdf)
                    bma_risk = (((bma - truth_cdf) ** 2) * w).sum(dim=1).cpu().numpy()
                    impossible = (~possible).sum(dim=1).cpu().numpy()
                    for row_index, rep in enumerate(ids):
                        common = {"truth": names[truth_index], "n": n, "replicate": rep,
                                  "impossible_candidate_count": int(impossible[row_index])}
                        for method in methods:
                            model = int(choices[method][row_index])
                            for predictive in risks:
                                endpoint = scales_hat[row_index, model] if predictive == "plugin" else upper
                                false_mass = 0.0 if shapes[model] is None else float(1 - cdf(endpoint, 1.0, truth))
                                row = dict(common, method=method, predictive=predictive,
                                           selected_model=names[model], cdf_risk=float(risks_np[predictive][row_index, model]),
                                           false_support_mass=false_mass)
                                writer.writerow(row)
                                rows.append(row)
                        row = dict(common, method="bayes_model_average", predictive="within_model_bayes",
                                   selected_model="mixture", cdf_risk=float(bma_risk[row_index]), false_support_mass=0.0)
                        writer.writerow(row)
                        rows.append(row)
                handle.flush()
                print(f"selection: truth={names[truth_index]} n={n} reps={cfg['replicates']}", flush=True)
    diagnostics["data_sha256"] = data_hash.hexdigest()
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    summarize(rows, output)
    return {"device": str(dev), "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
            "torch": torch.__version__, "row_count": len(rows), "dtype": "float64"}


def summarize(rows, output):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("truth", "n", "method", "predictive"))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, values in groups.items():
        risk = np.array([r["cdf_risk"] for r in values])
        summary.append(dict(zip(("truth", "n", "method", "predictive"), key),
                            replicates=len(values), mean_cdf_risk=float(risk.mean()),
                            mc_se=float(risk.std(ddof=1) / np.sqrt(len(risk))) if len(risk) > 1 else 0.,
                            mean_false_support_mass=float(np.mean([r["false_support_mass"] for r in values]))))
    with (output / "selection_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    paired = []
    for truth, n, method, predictive in groups:
        if method != "geometry":
            continue
        ours = {r["replicate"]: r["cdf_risk"] for r in groups[truth, n, method, predictive]}
        for baseline in ("bic", "bayes_grid"):
            comparison = groups[truth, n, baseline, predictive]
            differences = np.array([ours[r["replicate"]] - r["cdf_risk"] for r in comparison])
            paired.append({"truth": truth, "n": n, "predictive": predictive, "baseline": baseline,
                           "mean_risk_difference": float(differences.mean()),
                           "paired_mc_se": float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.,
                           "replicates": len(differences)})
    with (output / "paired_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
