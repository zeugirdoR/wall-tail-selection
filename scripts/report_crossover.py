"""Export the complete frozen comparison, including negative findings."""
import csv,json,hashlib,shutil,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
p=Path(sys.argv[1]);out=Path('results/crossover-v1');out.mkdir(parents=True,exist_ok=True)
for f in p.glob('analysis_*'):shutil.copy2(f,out/f.name)
for name in ['manifest.json','config.json','provenance_validation.json']:shutil.copy2(p/name,out/name)
for name in ['frozen.json','confirmation.json','size_requirements.csv']:shutil.copy2(Path('runs/crossover-freeze-v1')/name,out/name)
for source,dest in [('runs/deltaai-crosscal-3462202/manifest.json','calibration_manifest.json'),('runs/deltaai-crosscal-3462189/manifest.json','failed_calibration_manifest.json'),('runs/local-crossover-pilot-v2/manifest.json','pilot_manifest.json'),('runs/crossover-numerical-audit.json','numerical_audit.json'),('snapshots/crossover-confirmation-v1/snapshot-sha256.txt','executed_snapshot_sha256.txt'),('snapshots/crossover-confirmation-v1/CROSSOVER_PROTOCOL.md','protocol_as_executed.md')]:shutil.copy2(source,out/dest)
regions=list(csv.DictReader((p/'analysis_regions.csv').open()));ev=list(csv.DictReader((p/'analysis_evidence.csv').open()));overview=json.loads((p/'analysis_overview.json').read_text());manifest=json.loads((p/'manifest.json').read_text())
s='''# Boundary-geometric confirmation — 26 October 2026

**The tested boundary-Laplace candidate did not meet the prespecified practical-improvement rule in any whole region against the matched strong comparators.** It substantially repairs evidence approximation, but that is not superiority over exact evidence or predictive selection. The mixture boundary region instead shows a practically meaningful expected-log-loss disadvantage to LOO.

## What was fixed before confirmation

Two distinct families versus N(0,1): the symmetric equal mixture N(±mu,1), and N(0,1+mu²). All use mu in [0,2], a normalized uniform mu prior and equal model probabilities. The transported eta=mu² prior is 1/(4 sqrt(eta)), not uniform in eta. Three sample sizes (32,128,512) and seven cells per family/sample size give 42 cells: null and local h=0.5,1,2; boundary k=0.8,1.2; fixed separated mu=1. No true parameter is used by inference.

The candidate integrates a quadratic expansion in eta around the constrained MLE, retaining boundary score. The ablation expands in mu around the same MLE with the same prior and predictor. Thus it changes coordinates and retained higher-order terms; it is a defensible coordinate ablation, not proof of a unique intrinsic geometric invariant or a novel TOI theorem. Nonnegative/nonfinite curvature or an invalid approximation triggers a recorded exact-evidence fallback. The regular limit is handled without a fitted switching threshold.

Independent seeds: calibration 2026102601, pilot 2026102602, confirmation 2026102603; numerical audit 2026102699. Each of the 12 scores used the SAME 8,192 null calibration datasets per family/n, at the prespecified order-statistic cutoff 7,784 and strict exceedance. An analytic 5% chi-square score test is included separately. Native zero-threshold decisions and alpha-matched structural decisions are distinct analyses.

The 64-replicate/cell pilot was used only through paired variances to choose size. Its uncapped conservative precision requirement was 30,850.76 replicates/cell; the predetermined cap fixed confirmation at **4,096 per cell, 172,032 datasets**. We did not add observations or select cells after seeing confirmation. Some effects remain underpowered at this cap. Failure to demonstrate a benefit is not universal equivalence or proof of zero effect.

Expected predictive log loss uses validated 64-node Gauss-Hermite integration under the simulation truth; this reduces independent-test-sample noise. Truth enters only evaluation. All selectors share exact within-model posterior predictives. Differences in risk equal differences in regret because the truth's entropy cancels. Both families have full real support; no hard-support floor is used.

## Execution, repair and verification

Calibration job **3462189 failed** in the inherited stacking optimizer's numerically unstable boundary derivative. Before any thresholds or confirmation were obtained, we substituted an algebraically identical stable exponential-ratio derivative on [0,1]. The objective, seeds, cells and scientific method remained unchanged. Failed evidence/snapshot is preserved, and the pilot was rerun under corrected code.

Calibration **3462202 completed**, with 49,152 datasets, 14.08 seconds computation and 22 seconds allocated. Confirmation **3462217 completed**, exit 0, from **2026-10-26 13:10:26 to 13:11:38 UTC**, 72 seconds allocated and 62.04 seconds computation. It used 8 CPU workers on gh022 (ARM/GH200 allocation), existing bhry-dtai-gh allocation, 16 GB host memory and a 20-minute hard cap. No GPU acceleration claim or additional campaign.

All **36 tests passed**, including likelihood derivatives, the eta prior Jacobian, exact LOO/refit agreement, the stable sBIC root, regular-limit repair, stacking-boundary cases and interval controls. The 42-case numerical audit gave maximum score refinement error 2.79e-12, expected-risk refinement error 1.55e-10, and adaptive-evidence error 8.04e-14. Independent 65,536-observation test checks were within 1.76 Monte Carlo standard errors. This supports the declared 1e-7 score / 1e-6 risk numerical tolerances; it is not an exhaustive mathematical quadrature-error bound.

All raw hashes/counts/invariants validate. Freeze predates confirmation; calibration/pilot/confirmation training hashes are disjoint. A fixed 336-case local platform subset has identical native decisions and fallback flags: score differences at most 4.81e-12 and risk differences 9.33e-15. Source hashes match between that subset and the cluster run. There were **68 eta fallbacks / 172,032 cases (0.0395%)**, all explicitly retained, and zero mu-ablation fallbacks. Published evidence MAEs evaluate this fallback-inclusive policy; exact fallback is not credited as a purely approximate calculation.

## Prespecified inferential rules

The practical margins were **5 percentage points** for structural power and **0.001 nat/observation** for predictive risk. We computed 792 simultaneous statements under a fixed Bonferroni budget of 1,000 at familywise 0.05. Predictive paired t intervals use the Monte Carlo CLT approximation. Paired structural intervals conservatively combine exact Clopper–Pearson bounds for the two discordance probabilities; marginal null rates use Clopper–Pearson bounds. Zero empirical variance in a predictive difference is not a proof of population equality; rare unobserved decision differences remain a limitation of the t-interval approximation. These are conditional on the independently frozen calibration thresholds. The conformal rank has a marginal null-control interpretation, not a guarantee conditional on every calibration sample.

Region results give equal weight to all prespecified cells; region intervals average cell bounds, preserving the simultaneous accounting. Local predictive regions include null cells; local alternative-power regions exclude them. Structural benefit additionally requires paired null-error excess upper bound ≤0.01 and candidate marginal null-error upper bound ≤0.075 at every n in that family. No post-hoc winning-cell claims were substituted for these rules.

## Predictive risk: candidate minus comparator

Negative means better candidate risk. The following compact table shows the primary exact-evidence, LOO and ablation comparisons; the CSVs retain all eight comparators and all 42 cells. Intervals use the simultaneous scheme above, not unadjusted standard errors.

|Family|Region|Versus exact evidence|Versus LOO|Versus mu ablation|
|---|---|---:|---:|---:|
'''
def fmt(fam,reg,b,target='risk'):
 r=next(r for r in regions if r['family']==fam and r['region']==reg and r['target']==target and r['baseline']==b)
 return f"{float(r['difference']):+.6f} [{float(r['lower']):+.6f}, {float(r['upper']):+.6f}]"
for fam in ['mixture','variance']:
 for reg in ['local','boundary','regular']:s+=f"|{fam}|{reg}|{fmt(fam,reg,'exact')}|{fmt(fam,reg,'loo')}|{fmt(fam,reg,'mu_laplace')}|\n"
s+='''
For mixture boundary cases, candidate minus LOO is **+0.001398 [0.001161,0.001635]**: the disadvantage exceeds the prespecified practical margin. The candidate is close to exact-evidence selection in every region; no practical improvement over it is established. Small improvements over the mu-coordinate ablation in boundary/regular regions have intervals below zero but are below the required 0.001 margin. They should not be advertised as a practically decisive model-selection breakthrough. BMA and stacking are separate aggregation references and are fully included in analysis_regions.csv; no whole-region practical superiority over them was established either.

## Matched-error structural power

|Family|Region|Power difference versus calibrated LR|Power difference versus analytic score|
|---|---|---:|---:|
'''
for fam in ['mixture','variance']:
 for reg in ['local','boundary','regular']:s+=f"|{fam}|{reg}|{fmt(fam,reg,'lrt','power')}|{fmt(fam,reg,'analytic_score','power')}|\n"
s+='''
Candidate null rates range from **4.61% to 5.64%** across the six null cells, with simultaneous bounds in analysis_null_rates.csv. They pass the predeclared guard, but empirical calibration can still give small realized level differences. After matching null error, no region shows the required 5-point power benefit. BIC, RLCT-only and sBIC have exactly identical calibrated choices to LR in this two-model setting, exposing threshold effects that might otherwise be mistaken for geometry.

In the variance family, the candidate and calibrated LR make identical decisions. The analytic chi-square test is uniformly most powerful at the same exact null error, since sum(x²) has monotone likelihood ratio for one-sided variance alternatives. Its small observed differences from empirical-calibration rules cannot violate that theorem: the thresholds have slightly different realized levels. This is a derived condition where extra geometric information cannot improve structural power, and the experiment correctly recognizes it.

## Evidence approximation is a separate positive result

Mean absolute log-BF errors, averaged equally across each whole region:

|Family|Region|Eta candidate|Mu ablation|Previous local shape|sBIC|WBIC|
|---|---|---:|---:|---:|---:|---:|
'''
for fam in ['mixture','variance']:
 for reg in ['local','boundary','regular']:
  vals=[np.mean([float(r['mae']) for r in ev if r['family']==fam and r['regime']==reg and r['method']==method]) for method in ['eta_laplace','mu_laplace','local_shape','sbic','wbic']]
  s+=f"|{fam}|{reg}|"+'|'.join(f'{v:.5f}' for v in vals)+'|\n'
s+='''
The boundary expansion repairs the previous local formula's separated-control failure: regular-region MAE drops from roughly 20–22 nats to 0.024–0.030 nats. It also reduces near-null/boundary error relative to the mu ablation. In the regular regions the ordinary mu ablation is slightly more accurate, as its regular quadratic approximation is appropriate. These are descriptive approximation results, not prespecified claims of superior predictive or structural decisions.

Executed baselines include exact evidence, numerical exact LOO, WAIC, the correct two-model Drton–Plummer sBIC root, numerical WBIC at beta=1/log n, LR, moment/analytic score tests, BIC, crude RLCT-only, previous local shape, BMA and stacking. sBIC is not relabeled RLCT-only: r²+(1−b)r−a=0 with learning coefficients 1/4 and 1/2 and equal model priors. See [Drton–Plummer](https://arxiv.org/html/1309.0911) and [Watanabe's WBIC definition](https://jmlr.org/papers/v14/watanabe13a.html). These results do not cover unknown component variance/weights, other priors, other families or every current-best selector.

Median recorded reference integration time was 0.235 ms and candidate-plus-ablation computation 0.154 ms. These are shared-operation timings, not a rigorously matched end-to-end speed benchmark; MLE/LOO/posterior prediction also cost computation, and fallback incurs reference cost. No speed-superiority claim is made.

## Reproducibility and limitations

Compact artifacts are in results/crossover-v1: all-cell summaries, simultaneous intervals, full regions/null rates, frozen thresholds/configuration, pilot size requirements, failed/successful calibration manifests, confirmation manifest, numerical/platform audits and executed snapshot hashes. Raw compressed records remain locally in runs/deltaai-crossconfirm-3462217 and remotely under /work/nvme/bhry/crodriguez3/wall-tail-selection/crossover-confirmation-v1/runs/deltaai-crossconfirm-3462217. They are excluded from Git alongside host logs and credentials. The executed snapshot/config/source hashes identify the computation before publication.

Root protocol prose may receive whitespace-only readability corrections after execution; protocol_as_executed.md preserves its exact frozen bytes and hash. No scientific value or rule changes are implied. The generic earlier pilot/calibration manifest label is historical; the confirmation run is explicitly identified by seed 2026102603 and the pre-run frozen record.

The finite budget was below the pilot's most demanding precision requirement. Simultaneous intervals reveal what can and cannot be concluded; an empty set of positive practical claims is not proof that all possible geometric methods are equivalent. Here the honest result is a useful evidence approximation, a mathematically explained no-gain structural regime, and no demonstrated practically important selection improvement over the tested strong baselines. We did not tune on these outcomes or start a follow-up campaign.
'''
Path('CROSSOVER_REPORT_20261026.md').write_text(s)
files=[Path('CROSSOVER_REPORT_20261026.md'),Path('CROSSOVER_PROTOCOL.md'),Path('CROSSOVER_DERIVATION.md')]+[x for x in out.iterdir() if x.is_file() and x.name!='summary_sha256.json']
(out/'summary_sha256.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2)+'\n');print('Report and compact artifacts saved')
