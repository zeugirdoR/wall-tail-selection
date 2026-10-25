# Wall-tail selection

Exploratory model-selection experiments separating geometric/evidence approximations from predictive performance. The latest experiment studies the singular crossover in a symmetric two-normal mixture, with fixed proper priors and exact numerical evidence, LOO, WAIC, BMA and stacking references.

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
