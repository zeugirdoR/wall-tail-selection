"""Independent SciPy density/evidence reference for saved pilot selections."""
import csv, json, unittest
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, expon

class PilotReference(unittest.TestCase):
    def test_saved_smoke_against_scipy(self):
        p=Path(__file__).parent/'fixtures'/'pilot_reference'
        c=json.loads((p/'config.json').read_text()); s=c['selection']
        scales=np.geomspace(*s['scale_bounds'],s['scale_grid_size'])
        names=['exponential' if a is None else f'endpoint_a{a:g}' for a in s['shapes']]
        with (p/'replicates.csv').open() as handle:
            saved=list(csv.DictReader(handle))
        for ti,truth in enumerate(s['shapes']):
            for n in s['sample_sizes']:
                rng=np.random.default_rng(np.random.SeedSequence([c['seed'],ti,n,0]))
                u=rng.random(n)
                x=expon.ppf(u) if truth is None else beta.ppf(u,1,truth)
                ll=np.array([[sum(expon.logpdf(x,scale=t) if a is None else beta.logpdf(x,1,a,scale=t)) for t in scales] for a in s['shapes']])
                evidence=logsumexp(ll,axis=1)-np.log(len(scales))
                expected={'aic':names[np.argmax(ll.max(axis=1))], 'bic':names[np.argmax(ll.max(axis=1))], 'bayes_grid':names[np.argmax(evidence)]}
                for r in saved:
                    if r['truth']==names[ti] and int(r['n'])==n and int(r['replicate'])==0 and r['method'] in expected:
                        self.assertEqual(r['selected_model'],expected[r['method']])
