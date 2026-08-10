# Test fixtures

## `sales.csv` — causal-pipeline demo dataset

150 rows over `price`, `demand`, `income`, `season`, generated from a known SCM so the
causal pipeline's output can be checked against ground truth:

```
season -> price, season -> demand      (confounder)
income -> price, income -> demand      (confounder)
price  -> demand                       TRUE effect = -3.00
```

The naive unadjusted OLS slope is **-2.698**, so the file is also a demonstration that
`demand ~ price` alone is not the causal quantity.

Attach it in the UI with causal mode on, or drive the stages directly:

```python
import pandas as pd
from src.causal.discovery import reconcile_graph
from src.causal.estimation import run_identification
from src.causal.estimator import _apply_corrected_edges, _should_apply_correction

df = pd.read_csv("tests/fixtures/sales.csv")
recon = reconcile_graph(spec, df)
if _should_apply_correction(recon):
    spec = _apply_corrected_edges(spec, recon.corrected_edges)
ident, effect = run_identification(spec, df)
```

Regenerate byte-identically with:

```
python tests/fixtures/make_sales_csv.py tests/fixtures/sales.csv
```

### Two constraints if you build your own variant

Both were found by hitting them, and both are silent failures rather than errors:

1. **Noise must be non-Gaussian.** `src/causal/discovery.py` orients edges with
   DirectLiNGAM, whose identifiability result requires it. With `rng.normal()` the
   orientation is genuinely unidentifiable and LiNGAM ranked `demand` upstream of
   `price` — discovery then *reversed* the true edge.
2. **Confounding must be weak relative to the true effect.** With strong confounding,
   PC judged `price ⫫ demand` given the confounders and removed the true edge, leaving
   `corrected_edges` empty — the exact input that used to trigger the discarded-correction
   bug in `estimator.py`.

### Known limitation

Discovery removes the confounder edges on this data, so identification reports
`adjust for nothing` and estimation runs unadjusted. It still lands near truth because
the confounding is weak by construction — but this fixture does **not** exercise the
adjustment-set machinery. A dataset that yields `adjust for income, season` needs a
different coefficient balance.
