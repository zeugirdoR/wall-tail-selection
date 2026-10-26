# Wall-tail selection

## Latest: boundary-geometric suite (October 26)

The two-family, 42-cell held-out suite tested 4,096 replicates per cell (172,032 datasets), after separate null calibration and a variance-only sizing pilot. **No whole-region result met the prespecified practical selection-improvement rule against the matched strong comparators.** The boundary-Laplace approximation substantially repaired evidence accuracy, while predictive selection did not outperform exact evidence or the strong predictive baselines. The mixture boundary region was worse than LOO by a practically meaningful amount.

Read the [full results](CROSSOVER_REPORT_20261026.md), [protocol](CROSSOVER_PROTOCOL.md), [derivation and baseline definitions](CROSSOVER_DERIVATION.md), and [compact frozen artifacts](results/crossover-v1). The report includes the failed calibration job, numerical repair, power limitation, fallback counts and all-region negative findings.

```bash
python -m pip install -e '.[selection]'
PYTHONPATH=src python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/run_crossover.py --config results/crossover-v1/confirmation.json --output runs/NEW_CONFIRMATION --workers 4
python scripts/validate_crossover.py runs/NEW_CONFIRMATION
PYTHONPATH=src python scripts/analyze_crossover.py runs/NEW_CONFIRMATION results/crossover-v1/frozen.json
```

This reproduces the frozen benchmark; it does not refit thresholds. The calibration and pilot configurations are also published. `freeze_crossover.py` targets the documented historical pilot path; adjust paths, not seeds/methods, when reconstructing that workflow. Outputs must be new directories. Raw compressed records and host logs are excluded from Git. DeltaAI scripts show the actual existing allocation and versioned snapshot paths, not authorization for another user's resources. Computation is CPU-only despite the minimum GH200 allocation.

## Earlier singular-crossover pilot (October 25)

Exploratory model-selection experiments separating geometric/evidence approximations from predictive performance. The earlier experiment studies the singular crossover in a symmetric two-normal mixture, with fixed proper priors and exact numerical evidence, LOO, WAIC, BMA and stacking references.

**Result:** the local shape correction helps approximate evidence near the null but fails outside its local regime. This pilot does not establish superior predictive selection. Read [the full report](SINGULAR_REPORT_20261025.md), [derivation](SINGULAR_DERIVATION.md), and [frozen protocol and amendment](SINGULAR_PROTOCOL.md). Compact results and provenance are in [results/singular-v1](results/singular-v1).

## Reproduce

Python 3.10+ is required. Install with `python -m pip install -e '.[selection]'` to include the PyTorch dependency used by earlier regression tests. The singular experiment itself uses NumPy/SciPy CPU computation.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/run_singular.py --config configs/singular-pilot.json --output runs/NEW_PILOT --workers 4
python scripts/validate_singular.py runs/NEW_PILOT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/run_singular.py --config configs/singular-benchmark.json --output runs/NEW_BENCHMARK --workers 4
python scripts/validate_singular.py runs/NEW_BENCHMARK
python scripts/analyze_singular.py runs/NEW_BENCHMARK
```

Output directories must be new; the runner refuses to overwrite them. `scripts/audit_singular.py` recomputes the fixed numerical pilot checks and writes its diagnostics under `runs/` (create that directory first if using it directly). The report-export script targets the historical run path listed in its source; ordinary replication and analysis accept new output paths as shown above.

The frozen benchmark has 21 cells × 32 replicates, with independent training/test seeds, fixed uniform prior μ∈[0,2], and equal model probabilities. The decision-boundary grid was added transparently before benchmark execution. All comparisons are exploratory; no approximation is retuned using held-out prediction.

The package also retains the earlier wall/endpoint-family geometry and calibration implementation needed by regression tests. Those experiments use different priors and losses; they are not interchangeable with the mixture benchmark. The small historical regression fixture contains simulated data summaries only. Large raw runs, immutable local snapshots, host logs and credentials are excluded from publication.

DeltaAI batch example: `deltaai/singular.sbatch` records the actual existing allocation and remote snapshot path used for job3453668. Adapt paths/allocation only to an environment you are authorized to use. The job requested one GH200 allocation but performed CPU integration; no GPU speedup is claimed. Raw outputs remain at the paths in the report; published manifests record their hashes and executable provenance.
