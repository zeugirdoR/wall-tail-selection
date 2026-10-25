"""Run immutable configurations and record successful or failed completion."""

import argparse
import csv
import math
import time

from .geometry import endpoint_h2, resolution_log_step
from .provenance import finish_run, save_manifest, start_run


def run_geometry(config, output):
    cfg = config["geometry"]
    rows = []
    for n in cfg["sample_sizes"]:
        cases = [("fixed_shape", a, None) for a in cfg["shapes"]]
        cases += [("boundary_layer", 2 + b / math.log(n), b) for b in cfg["boundary_b"]
                  if 2 + b / math.log(n) >= 1]
        for kind, a, b in cases:
            step = resolution_log_step(a, n)
            h = math.expm1(step)
            normalization = math.sqrt(n) if a > 2 else math.sqrt(n * math.log(n)) if a == 2 else n ** (1 / a)
            rows.append({"kind": kind, "a": a, "b": b, "n": n, "log_step": step,
                         "relative_step": h, "n_h2": n * endpoint_h2(a, step),
                         "scaled_relative_step": h * normalization})
    with (output / "geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"geometry: {len(rows)} radii computed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["geometry", "selection", "pilot"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="A NEW directory; existing runs are never overwritten")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config, output, manifest = start_run(args.config, args.output)
    start = time.monotonic()
    try:
        if args.command in ("geometry", "pilot"):
            run_geometry(config, output)
        if args.command in ("selection", "pilot"):
            from .selection import run_selection
            manifest["selection_runtime"] = run_selection(config, output, args.device)
        manifest["elapsed_seconds"] = time.monotonic() - start
        finish_run(output, manifest)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=time.monotonic() - start)
        save_manifest(output, manifest)
        raise


if __name__ == "__main__":
    main()
