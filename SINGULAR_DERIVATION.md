# Local mixture evidence and its limits

For p_mu(x)=phi(x) exp(-mu²/2) cosh(mu x), logLR(mu)=sum logcosh(mu x_i)-n mu²/2. Expanding logcosh(y)=y²/2-y⁴/12+O(y⁶), setting mu=t n^(-1/4), gives

logLR = t² Z_n/sqrt(2) - (t⁴/12)(sum x_i⁴/n) + O(t⁶ n^(-3/2) sum |x_i|⁶),
Z_n=sum(x_i²-1)/sqrt(2n).

For bounded true h with mu0=h n^(-1/4), E x²=1+mu0² and E x⁴=3+6mu0²+mu0⁴. Thus Z_n converges to N(h²/sqrt(2),1), sum x⁴/n=3+O_P(n^(-1/2)), and the remainder is O_P(n^(-1/2)) uniformly on bounded t. Bounded higher derivatives of logcosh give a sixth-order remainder bound; mixture-normal moments are uniformly bounded for bounded h. This establishes the stated compact-t expansion, not uniform validity at arbitrary separation.

For the fixed uniform prior [0,M] with M=2, change variables mu=t n^(-1/4): BF10=(n^(-1/4)/M) integral_0^(M n^(1/4)) exp(logLR(t n^(-1/4)))dt. The leading expression replaces this integral by I(Z)=integral_0^infinity exp(t²Z/sqrt(2)-t⁴/4)dt. Under bounded h its ratio to BF tends to 1 in probability. Tail justification: on moment-control events the small-mu likelihood is bounded above by C sqrt(n)mu²-c n mu⁴ (choose a fixed small epsilon so the sixth-order term is dominated by the quartic term). The resulting rescaled tail is integrable for tight Z. On epsilon<=mu<=M, compact uniform laws of large numbers and the positive null KL gap give exponentially small contributions. Local alternatives preserve these controls since mu0 tends to 0. This is a proof outline; no finite-n uniform error bound is asserted.

I(0)=Gamma(1/4)/(2 sqrt(2)). The quartic peak is max(Z,0)²/2, at t=sqrt(sqrt(2)max(Z,0)). Numerics subtract that peak before integrating, avoiding overflow. The original exact likelihood has at most one positive stationary point: sum[x tanh(mu x)/mu]-n decreases in mu, since tanh(z)/z decreases for z>0. If sum x²<=n the constrained MLE is 0; otherwise find the root or the upper-bound optimum.

The singular learning exponent is 1/4 at zero for a prior smooth and positive there; regular separated M1 parameters have 1/2 behavior. For bounded h, I(Z)=O_P(1), so BF10 tends to 0: Bayes-consistent selection favors M0 even under these contiguous local alternatives. Nontrivial local testing power at n^(-1/4) is distinct from a consistent model-selection threshold. Heuristically the local peak must compete with (1/4)logn, suggesting a growing h of order(logn)^(1/4); this is not a theorem uniform for growing h. Fixed mu=1 lies outside the bounded-h approximation, and failure there must be reported.

Local-shape logBF=-0.25logn-logM+logI(Z). The ablation uses peak(Z)-0.25logn-logM+logI(0), removing variation of the peak-adjusted integral. RLCT-only uses the exact constrained peak minus 0.25logn, intentionally omitting unknown O(1) terms: it is a crude control, not full sBIC. BIC is a weak control at the singularity. Approximation to exact evidence and improvements in predictive decisions are different claims. Exact evidence need not minimize a finite-sample predictive loss, and a faster evidence approximation does not eliminate the cost of an identical full posterior predictive.
