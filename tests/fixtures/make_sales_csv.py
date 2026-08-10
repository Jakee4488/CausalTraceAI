"""Generate sales.csv — a demo dataset for the TracerLensAi causal pipeline.

Ground-truth SCM (what the pipeline should recover):

    season  -> price,  season -> demand      (confounder)
    income  -> price,  income -> demand      (confounder)
    price   -> demand                        TRUE effect = -3.00

Two parameter choices are load-bearing; both were found the hard way:

1. NOISE MUST BE NON-GAUSSIAN.
   src/causal/discovery.py orients edges with DirectLiNGAM, whose
   identifiability result *requires* non-Gaussian noise. With rng.normal()
   the orientation is genuinely unidentifiable and LiNGAM ranked `demand`
   upstream of `price`, so discovery.py:120-121 reversed the true edge.
   Hence laplace/uniform exogenous terms below.

2. CONFOUNDING MUST BE WEAK RELATIVE TO THE TRUE EFFECT.
   With strong confounding (price coefficients ~1.8*season + 0.12*income),
   PC judged price _||_ demand given the confounders and *removed* the true
   edge — leaving corrected_edges empty. That produced a "corrected" verdict
   the estimator then discarded, which is exactly the bug fixed in
   estimator.py (_should_apply_correction). Keep the price coefficients small
   and the price noise large so price -> demand stays conditionally dependent.

Run:  python tests/fixtures/make_sales_csv.py tests/fixtures/sales.csv

See tests/fixtures/README.md for usage and a known limitation (this data yields
"adjust for nothing", so it does not exercise the adjustment-set machinery).
"""

import sys

import numpy as np
import pandas as pd

N_ROWS = 150          # >= 30 or reconcile_graph returns verdict="untestable"
TRUE_EFFECT = -3.0    # price -> demand
SEED = 3              # pinned: other seeds can flip the LiNGAM ordering


def make_sales(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Exogenous roots. Uniform (not normal) to keep the system non-Gaussian.
    season = rng.integers(1, 5, n).astype(float)
    income = np.round(50 + rng.uniform(-14, 14, n), 1)

    # Both confounders feed price, but weakly — see note 2 in the docstring.
    price = np.round(
        15 + 0.7 * season + 0.05 * income + rng.laplace(0, 2.2, n), 2
    )

    # Demand: the true causal effect of price, plus both confounder paths.
    demand = np.round(
        200 + TRUE_EFFECT * price + 2.0 * season + 0.30 * income
        + rng.laplace(0, 1.3, n),
        1,
    )

    return pd.DataFrame(
        {"price": price, "demand": demand, "income": income, "season": season}
    )


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sales.csv"
    df = make_sales()
    df.to_csv(out, index=False)

    # The naive slope is confounded; the pipeline's job is to beat it.
    naive = np.polyfit(df.price, df.demand, 1)[0]
    print(f"wrote {out}  ({len(df)} rows)")
    print(f"  true effect        : {TRUE_EFFECT:+.3f}")
    print(f"  naive OLS slope    : {naive:+.3f}   (confounded, unadjusted)")
    print(df.head().to_string(index=False))
