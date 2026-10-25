# Singular crossover results — 25 October 2026

The finite-sample shape correction improves evidence approximation in the near-null region, but this pilot does **not** establish improved predictive model selection. It is inaccurate for separated mixtures, and matching exact model choices can conceal very large evidence errors.

## Execution and reproducibility

DeltaAI job **3453668** completed with exit 0 in 12 seconds allocated, from 12:26:46 to 12:26:58 UTC. Numerical workload 1.405 seconds; CPU SciPy, 8 workers on gh045 (ARM Neoverse/GH200 allocation), no GPU acceleration claim. Local replication took 4.563 seconds. Existing bhry-dtai-gh allocation, 1 node/GPU, 16 GB, 20-minute cap. No second-family campaign or further tuning was performed.

Frozen configuration: configs/singular-benchmark.json, seed 2026102501, 21 cells × 32 replicates = 672 independent training/test pairs, with 2048 independent test observations per pair. Both models have fixed unit component variance; M1 has equal weights and symmetric means ±mu, uniform proper prior mu∈[0,2], and model prior probabilities 1/2. The bounded-h, decision-boundary and separated controls are distinct regimes. A transparent pre-benchmark amendment added the boundary grid after numerical pilot validation; it was not chosen from benchmark outcomes. See SINGULAR_PROTOCOL.md and SINGULAR_DERIVATION.md.

All 28 tests passed locally and on the cluster. The 24 pilot datasets were checked with 256 versus 512 quadrature nodes and independent adaptive integration: maximum log-BF refinement error 1.981e-13, adaptive error 6.972e-14, predictive log-loss change 9.326e-15, no changed selections. Tests also cover density normalization, direct leave-one-out refits, the local expansion, I(0), the integral differential equation, its positive-Z asymptotic, and stacking endpoints/interior optimum.

All 672 benchmark datasets, 3360 evidence records and 7392 predictive records validate. Full local/cluster source/config/data agree; maximum log-BF difference 2.607e-13, LOO/WAIC score difference 1.322e-11, predictive loss difference 2.576e-14. Model choices agree exactly. Floating aggregation weights differ by at most 7.663e-13. Timings are excluded from equality checks. Original data, code and earlier experiment outputs remain preserved.

Raw local run: runs/deltaai-singular-3453668; independent replication: runs/local-singular-benchmark-v1. Remote: /work/nvme/bhry/crodriguez3/wall-tail-selection/singular-v1/runs/deltaai-singular-3453668. Executed snapshot: snapshots/singular-v1. Compact publishable summaries, manifests, configuration, source hashes and replication audit are in results/singular-v1; raw arrays/rows and host logs are not included in Git. The snapshot hashes, rather than a later publication commit, identify the exact code that executed.

## Evidence approximation

The local expression is logBF≈−(1/4)logn−log2+logI(Z), with I(Z)=integral_0^infinity exp(t²Z/sqrt2−t⁴/4)dt. Its proof applies to bounded true h and compact rescaled t plus tail control, not uniformly to regular separated controls or growing h. The shape ablation retains the local peak but holds its peak-adjusted integral correction at Z=0. RLCT-only is the crude exact-peak−(1/4)logn surrogate with omitted constant set to 0; it is **not full sBIC**. BIC is a weak singularity control.

All-cell evidence errors (MAE in log-BF nats; null is h=0):

|Regime|n|h, k or fixed mu|Local shape MAE|Ablated shape MAE|RLCT-only MAE|BIC MAE|Local within 0.1 nat|
|---|---:|---:|---:|---:|---:|---:|---:|
|local|32|0.0|0.1146|0.2286|0.4483|0.4347|56.2%|
|local|32|0.5|0.1984|0.3131|0.4088|0.4608|56.2%|
|local|32|1.0|0.1715|0.2293|0.2513|0.6152|37.5%|
|local|32|2.0|1.5037|1.7772|0.3242|0.5422|21.9%|
|regular|32|1.0|3.8687|4.3654|0.4556|0.4109|3.1%|
|local|128|0.0|0.0471|0.2053|0.4984|0.7146|96.9%|
|local|128|0.5|0.0713|0.2108|0.4677|0.7453|78.1%|
|local|128|1.0|0.1763|0.3285|0.4315|0.7815|68.8%|
|local|128|2.0|1.2105|1.5183|0.5195|0.6935|9.4%|
|regular|128|1.0|10.7225|11.5871|0.8197|0.3933|0.0%|
|local|512|0.0|0.0372|0.1924|0.5109|1.0487|96.9%|
|local|512|0.5|0.0368|0.2130|0.4805|1.0790|93.8%|
|local|512|1.0|0.0405|0.1639|0.4090|1.1505|93.8%|
|local|512|2.0|0.5528|0.8430|0.5791|0.9805|43.8%|
|regular|512|1.0|48.6096|49.8560|1.1826|0.3770|0.0%|
|boundary|32|0.8|0.1543|0.2536|0.2708|0.5957|34.4%|
|boundary|32|1.2|0.8339|1.0228|0.2969|0.5695|18.8%|
|boundary|128|0.8|0.2152|0.3457|0.3269|0.8861|59.4%|
|boundary|128|1.2|0.3642|0.5560|0.4206|0.7924|40.6%|
|boundary|512|0.8|0.0965|0.1977|0.4065|1.1531|81.2%|
|boundary|512|1.2|0.4537|0.7382|0.5788|0.9808|37.5%|

At n 512 and h≤1, local-shape MAE is 0.0368–0.0405 nat. At the null, adding the shape correction reduces mean absolute error versus ablation by 0.1552 nat (paired MCSE 0.0257). This is useful evidence for the mechanism, not a general finite-sample guarantee. For h 2, MAE remains 0.5528 nat at n 512, and the n 32 h 2 cell contains an error of 15.857 nat. The boundary k 1.2 cell at n 512 has MAE 0.4537 nat.

For fixed mu 1, MAE grows from 3.869 to 48.610 nat as n grows 32→512. Those are failures outside the local validity regime, not quadrature failures. Exact logBF cannot exceed the constrained maximum logLR because the prior is normalized. The local approximation violates that exact upper bound in 22 bounded-h cases, 5 boundary cases and 83 regular cases. It was not clipped or repaired after seeing results.

## Decisions and independent-test prediction

Local-shape and exact-evidence selections agree on 664/672 datasets (98.81%), despite the evidence failures above. All regular-control decisions agree because both strongly select M1, so agreement alone is insufficient validation. The predeclared 5-percentage-point disagreement margin is exceeded in 2/21 cells (each 6.25%). Choice rates and descriptive truth-label errors for all selectors are in analysis_predictions.csv.

Under bounded h, Z converges to N(h²/sqrt2,1) and BF decays as n^(-1/4) times a tight random factor. Evidence selection eventually prefers M0 even under local alternatives. We do not call that a numerical failure or require true-label consistency in that regime. The boundary grid targets competition with the logarithmic penalty but is not covered by the bounded-h theorem.

Every selector uses the **same** exact within-M1 posterior predictive and fixed M0 density. Exact LOO is computed through the full-posterior identity and independently checked with refits. WAIC uses posterior log-density variance. BMA uses exact posterior model weights; stacking optimizes the two-model LOO score. Both are aggregation baselines. Always-M0/M1 controls are retained.

For local shape versus exact evidence selection, 12 bounded-h cells have 0 lower / 9 tied / 3 higher mean test log losses; 6 boundary cells have 2 lower / 3 tied / 1 higher; 3 separated cells tie. Neither lower boundary contrast reaches the predeclared 0.001 nat/observation practical margin. Two cells are worse by more than that margin: n 32 h 2, +0.004651 (paired MCSE 0.003244), and n 32 k 1.2, +0.001255 (SE 0.001255). These are exploratory mean differences, not claims of statistically established harm or benefit.

Against BMA, local shape is lower in 2/12 bounded-h cells and higher in 10; against stacking, lower in 4 and higher in 8. None of these local-cell improvements reaches 0.001 nat/observation. In the 6 boundary cells, local shape is worse in all 6 against both BMA and stacking. Its few separated-control improvements over aggregation are shared by exact selection and therefore are not attributable to the local approximation.

All 21-cell paired risk differences and MCSEs against exact selection, BMA, stacking and shape ablation are in analysis_risk_contrasts.csv. Ordinary LOO/WAIC are implemented, but full sBIC, WBIC, generalized variance/mixture-weight models and a comprehensive current-best comparison are not. No superiority claim over omitted methods is made.

## Computation, interpretation and limits

On the cluster, median local-integral evaluation was 67 microseconds, versus 2.08 milliseconds for reference quadrature plus exact MLE; Z computation is outside the local timer. LOO/WAIC/stacking added a median 0.84 milliseconds, and independent-test posterior prediction 11.89 milliseconds. These overlapping/shared computations are not an end-to-end speed benchmark. The identical posterior predictive still requires reference integration; fast evidence alone does not make that cost disappear.

The strongest supported direction is a **regime-limited evidence approximation**, not a new superior selector. The data-dependent correction is measurably useful near zero compared with its ablation, but its validity boundary matters and predictive gains have not been demonstrated. A future globally valid approximation could combine local and regular behavior, but designing or tuning such a procedure is outside this run and was not done.

Only 32 independent replicates per cell were used. Each paired MCSE includes randomness from training and finite independent test samples; it is not a simultaneous confidence statement. The 0.1 nat evidence, 0.001 nat/observation prediction and 5-point choice margins were fixed before benchmark results. No multiple-testing significance claim, parameter tuning, optional stopping or second family was introduced. This is a first-stage exploratory pilot and requires stronger independent replication for conclusive claims.

## Publication scope

The initial repository publication contains the runnable package (including earlier modules needed by existing tests), configurations, tests with a 37 KB simulated historical reference fixture, singular runner/audit/analysis/report scripts, frozen protocol/derivation, and compact result summaries/provenance. Raw run directories, snapshots, host logs, credentials and unrelated local reports remain outside the commit. No existing GitHub history was replaced; origin had no refs when inspected.
