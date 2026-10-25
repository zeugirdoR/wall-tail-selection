"""Export compact, auditable summaries and a human-readable scientific report."""
from pathlib import Path
import csv,json,hashlib,shutil
p=Path('runs/deltaai-singular-3453668');o=json.loads((p/'analysis_overview.json').read_text());m=json.loads((p/'manifest.json').read_text());out=Path('results/singular-v1');out.mkdir(parents=True,exist_ok=True)
for f in p.glob('analysis_*'):shutil.copy2(f,out/f.name)
for f in ['manifest.json','config.json','replication_validation.json']:shutil.copy2(p/f,out/f)
shutil.copy2('runs/singular-numerical-audit-v1.json',out/'numerical_audit.json');shutil.copy2('snapshots/singular-v1/snapshot-sha256.txt',out/'executed_snapshot_sha256.txt')
ev=list(csv.DictReader((p/'analysis_evidence.csv').open()));pr=list(csv.DictReader((p/'analysis_predictions.csv').open()));co=list(csv.DictReader((p/'analysis_risk_contrasts.csv').open()))
s='''# Singular crossover results — 25 October 2026

The finite-sample shape correction improves evidence approximation in the near-null region, but this pilot does **not** establish improved predictive model selection. It is inaccurate for separated mixtures, and matching exact model choices can conceal very large evidence errors.

## Execution and reproducibility

DeltaAI job **3453668** completed with exit0 in12seconds allocated, from12:26:46 to12:26:58 UTC. Numerical workload1.405seconds; CPU SciPy,8 workers on gh045 (ARM Neoverse/GH200 allocation), no GPU acceleration claim. Local replication took4.563seconds. Existing bhry-dtai-gh allocation,1 node/GPU,16GB,20-minute cap. No second-family campaign or further tuning was performed.

Frozen configuration: configs/singular-benchmark.json, seed2026102501,21 cells ×32 replicates =672 independent training/test pairs, with2048 independent test observations per pair. Both models have fixed unit component variance; M1 has equal weights and symmetric means±mu, uniform proper prior mu∈[0,2], and model prior probabilities1/2. The bounded-h, decision-boundary and separated controls are distinct regimes. A transparent pre-benchmark amendment added the boundary grid after numerical pilot validation; it was not chosen from benchmark outcomes. See SINGULAR_PROTOCOL.md and SINGULAR_DERIVATION.md.

All28 tests passed locally and on the cluster. The24 pilot datasets were checked with256 versus512 quadrature nodes and independent adaptive integration: maximum log-BF refinement error1.981e-13, adaptive error6.972e-14, predictive log-loss change9.326e-15, no changed selections. Tests also cover density normalization, direct leave-one-out refits, the local expansion, I(0), the integral differential equation, its positive-Z asymptotic, and stacking endpoints/interior optimum.

All672 benchmark datasets,3360 evidence records and7392 predictive records validate. Full local/cluster source/config/data agree; maximum log-BF difference2.607e-13, LOO/WAIC score difference1.322e-11, predictive loss difference2.576e-14. Model choices agree exactly. Floating aggregation weights differ by at most7.663e-13. Timings are excluded from equality checks. Original data, code and earlier experiment outputs remain preserved.

Raw local run: runs/deltaai-singular-3453668; independent replication: runs/local-singular-benchmark-v1. Remote: /work/nvme/bhry/crodriguez3/wall-tail-selection/singular-v1/runs/deltaai-singular-3453668. Executed snapshot: snapshots/singular-v1. Compact publishable summaries, manifests, configuration, source hashes and replication audit are in results/singular-v1; raw arrays/rows and host logs are not included in Git. The snapshot hashes, rather than a later publication commit, identify the exact code that executed.

## Evidence approximation

The local expression is logBF≈−(1/4)logn−log2+logI(Z), with I(Z)=integral_0^infinity exp(t²Z/sqrt2−t⁴/4)dt. Its proof applies to bounded true h and compact rescaled t plus tail control, not uniformly to regular separated controls or growing h. The shape ablation retains the local peak but holds its peak-adjusted integral correction at Z=0. RLCT-only is the crude exact-peak−(1/4)logn surrogate with omitted constant set to0; it is **not full sBIC**. BIC is a weak singularity control.

All-cell evidence errors (MAE in log-BF nats; null is h=0):

|Regime|n|h, k or fixed mu|Local shape MAE|Ablated shape MAE|RLCT-only MAE|BIC MAE|Local within0.1nat|
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for r in [x for x in ev if x['method']=='local_shape']:
 vals={x['method']:x for x in ev if x['cell']==r['cell']};s+=f"|{r['regime']}|{r['n']}|{r['separation']}|{float(r['mae']):.4f}|{float(vals['shape_ablation']['mae']):.4f}|{float(vals['rlct_only']['mae']):.4f}|{float(vals['bic']['mae']):.4f}|{100*float(r['within_point1_fraction']):.1f}%|\n"
s+='''
At n512 and h≤1, local-shape MAE is0.0368–0.0405nat. At the null, adding the shape correction reduces mean absolute error versus ablation by0.1552nat (paired MCSE0.0257). This is useful evidence for the mechanism, not a general finite-sample guarantee. For h2, MAE remains0.5528nat at n512, and the n32 h2 cell contains an error of15.857nat. The boundary k1.2 cell at n512 has MAE0.4537nat.

For fixed mu1, MAE grows from3.869 to48.610nat as n grows32→512. Those are failures outside the local validity regime, not quadrature failures. Exact logBF cannot exceed the constrained maximum logLR because the prior is normalized. The local approximation violates that exact upper bound in22 bounded-h cases,5 boundary cases and83 regular cases. It was not clipped or repaired after seeing results.

## Decisions and independent-test prediction

Local-shape and exact-evidence selections agree on664/672 datasets (98.81%), despite the evidence failures above. All regular-control decisions agree because both strongly select M1, so agreement alone is insufficient validation. The predeclared5-percentage-point disagreement margin is exceeded in2/21 cells (each6.25%). Choice rates and descriptive truth-label errors for all selectors are in analysis_predictions.csv.

Under bounded h, Z converges to N(h²/sqrt2,1) and BF decays as n^(-1/4) times a tight random factor. Evidence selection eventually prefers M0 even under local alternatives. We do not call that a numerical failure or require true-label consistency in that regime. The boundary grid targets competition with the logarithmic penalty but is not covered by the bounded-h theorem.

Every selector uses the **same** exact within-M1 posterior predictive and fixed M0 density. Exact LOO is computed through the full-posterior identity and independently checked with refits. WAIC uses posterior log-density variance. BMA uses exact posterior model weights; stacking optimizes the two-model LOO score. Both are aggregation baselines. Always-M0/M1 controls are retained.

For local shape versus exact evidence selection,12 bounded-h cells have0 lower /9 tied /3 higher mean test log losses;6 boundary cells have2 lower /3 tied /1 higher;3 separated cells tie. Neither lower boundary contrast reaches the predeclared0.001nat/observation practical margin. Two cells are worse by more than that margin: n32 h2, +0.004651 (paired MCSE0.003244), and n32 k1.2, +0.001255 (SE0.001255). These are exploratory mean differences, not claims of statistically established harm or benefit.

Against BMA, local shape is lower in2/12 bounded-h cells and higher in10; against stacking, lower in4 and higher in8. None of these local-cell improvements reaches0.001nat/observation. In the6 boundary cells, local shape is worse in all6 against both BMA and stacking. Its few separated-control improvements over aggregation are shared by exact selection and therefore are not attributable to the local approximation.

All21-cell paired risk differences and MCSEs against exact selection, BMA, stacking and shape ablation are in analysis_risk_contrasts.csv. Ordinary LOO/WAIC are implemented, but full sBIC, WBIC, generalized variance/mixture-weight models and a comprehensive current-best comparison are not. No superiority claim over omitted methods is made.

## Computation, interpretation and limits

On the cluster, median local-integral evaluation was67microseconds, versus2.08milliseconds for reference quadrature plus exact MLE; Z computation is outside the local timer. LOO/WAIC/stacking added a median0.84milliseconds, and independent-test posterior prediction11.89milliseconds. These overlapping/shared computations are not an end-to-end speed benchmark. The identical posterior predictive still requires reference integration; fast evidence alone does not make that cost disappear.

The strongest supported direction is a **regime-limited evidence approximation**, not a new superior selector. The data-dependent correction is measurably useful near zero compared with its ablation, but its validity boundary matters and predictive gains have not been demonstrated. A future globally valid approximation could combine local and regular behavior, but designing or tuning such a procedure is outside this run and was not done.

Only32 independent replicates per cell were used. Each paired MCSE includes randomness from training and finite independent test samples; it is not a simultaneous confidence statement. The0.1nat evidence,0.001nat/observation prediction and5-point choice margins were fixed before benchmark results. No multiple-testing significance claim, parameter tuning, optional stopping or second family was introduced. This is a first-stage exploratory pilot and requires stronger independent replication for conclusive claims.

## Publication scope

The initial repository publication contains the runnable package (including earlier modules needed by existing tests), configurations, tests with a37KB simulated historical reference fixture, singular runner/audit/analysis/report scripts, frozen protocol/derivation, and compact result summaries/provenance. Raw run directories, snapshots, host logs, credentials and unrelated local reports remain outside the commit. No existing GitHub history was replaced; origin had no refs when inspected.
'''
Path('SINGULAR_REPORT_20261025.md').write_text(s)
files=[Path('SINGULAR_REPORT_20261025.md'),Path('SINGULAR_PROTOCOL.md'),Path('SINGULAR_DERIVATION.md')]+[x for x in out.iterdir() if x.is_file() and x.name!='summary_sha256.json']
(out/'summary_sha256.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2)+'\n')
print('Compact results and report saved')
