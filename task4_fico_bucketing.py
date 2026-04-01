"""
=================================================================
  FICO Score Optimal Bucketing â€” Rating Map Generator
  Retail Banking / Mortgage Book  |  QR Team Prototype
=================================================================

  Two optimisation objectives, both solved via Dynamic Programming:

  1. MSE (Mean Squared Error)
     Minimise:  sum_i  sum_{j in bucket_i}  (x_j - mean_i)^2
     Treats bucketing as an approximation problem â€” find boundaries
     that keep each FICO score as close as possible to its bucket mean.

  2. Log-Likelihood
     Maximise:  sum_i  [ k_i * ln(p_i)  +  (n_i - k_i) * ln(1 - p_i) ]
     where n_i = loans in bucket i, k_i = defaults, p_i = k_i / n_i.
     Finds boundaries that create maximally homogeneous default-rate
     segments â€” the statistically principled choice for PD modelling.

  Rating convention: 1 = best credit (high FICO), N = worst credit.
=================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# â”€â”€ visual style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BG    = '#0d1117'; PANEL = '#161b22'; GRID = '#21262d'
TEXT  = '#e6edf3'; MUTED = '#8b949e'
C     = ['#58a6ff', '#f85149', '#3fb950', '#d29922', '#bc8cff']

plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor':  PANEL,
    'axes.edgecolor':   GRID, 'axes.labelcolor': MUTED,
    'xtick.color':      MUTED,'ytick.color':     MUTED,
    'grid.color':       GRID, 'grid.linewidth':  0.6,
    'text.color':       TEXT, 'font.family':     'DejaVu Sans',
    'legend.framealpha':0.25, 'legend.edgecolor':GRID,
})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  LOAD DATA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA = '/mnt/user-data/uploads/1774904873268_Task_3_and_4_Loan_Data__1_.csv'
df   = pd.read_csv(DATA)

fico    = df['fico_score'].values
default = df['default'].values

fico_min, fico_max = int(fico.min()), int(fico.max())
N_OBS = len(fico)

print("=" * 64)
print("  FICO OPTIMAL BUCKETING â€” RATING MAP GENERATOR")
print("=" * 64)
print(f"  Borrowers  : {N_OBS:,}")
print(f"  FICO range : {fico_min} â€“ {fico_max}")
print(f"  Default rate: {default.mean():.2%}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  PRECOMPUTE SEGMENT STATISTICS (shared by both DP solvers)
#     For every contiguous segment [a, b] of unique FICO values:
#       mse_cost[a][b]  = within-segment sum of squared deviations
#       ll_gain[a][b]   = log-likelihood contribution of that segment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

unique_scores = np.arange(fico_min, fico_max + 1)
M = len(unique_scores)                       # number of distinct score values

# --- aggregate to score level (faster than row-level DP) ----
score_to_idx = {s: i for i, s in enumerate(unique_scores)}
n_at   = np.zeros(M, dtype=int)    # total borrowers at each score
k_at   = np.zeros(M, dtype=int)    # defaulters at each score

for s, d in zip(fico, default):
    i = score_to_idx[s]
    n_at[i] += 1
    k_at[i] += d

# prefix sums for O(1) segment queries
prefix_n  = np.concatenate([[0], np.cumsum(n_at)])      # total loans
prefix_k  = np.concatenate([[0], np.cumsum(k_at)])      # total defaults
prefix_s  = np.concatenate([[0], np.cumsum(n_at * unique_scores)])       # sum of FICO
prefix_s2 = np.concatenate([[0], np.cumsum(n_at * unique_scores**2)])    # sum of FICO^2


def seg_n(a, b):
    """Total borrowers in score-index range [a, b) """
    return prefix_n[b] - prefix_n[a]

def seg_k(a, b):
    """Total defaults in score-index range [a, b)"""
    return prefix_k[b] - prefix_k[a]

def seg_mse_cost(a, b):
    """
    Sum of squared deviations from mean for FICO values in [a, b).
    Uses: sum(x-mu)^2 = sum(x^2) - n*mu^2
    """
    n   = prefix_n[b]  - prefix_n[a]
    if n == 0: return 0.0
    s   = prefix_s[b]  - prefix_s[a]
    s2  = prefix_s2[b] - prefix_s2[a]
    mu  = s / n
    return s2 - n * mu ** 2

def seg_ll_gain(a, b):
    """
    Log-likelihood contribution of segment [a, b).
    ll = k*ln(p) + (n-k)*ln(1-p),  p = k/n
    Returns 0 if segment is pure (all default or all non-default) to avoid -inf.
    """
    n = seg_n(a, b)
    k = seg_k(a, b)
    if n == 0 or k == 0 or k == n:
        return 0.0
    p = k / n
    return k * np.log(p) + (n - k) * np.log(1.0 - p)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  DYNAMIC PROGRAMMING SOLVERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def dp_mse(num_buckets: int):
    """
    DP to find bucket boundaries minimising total within-bucket MSE.

    State:  dp[i][b] = minimum MSE cost using b buckets over scores [0..i)
    Trans:  dp[i][b] = min over j<i  of  dp[j][b-1] + seg_mse_cost(j, i)
    Returns list of boundary FICO scores (len = num_buckets - 1).
    """
    B = num_buckets
    # dp[i][b]: min cost for first i score-values with b buckets
    INF = float('inf')
    dp   = [[INF] * (B + 1) for _ in range(M + 1)]
    split = [[0]  * (B + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for b in range(1, B + 1):
        for i in range(b, M + 1):
            for j in range(b - 1, i):
                cost = dp[j][b - 1] + seg_mse_cost(j, i)
                if cost < dp[i][b]:
                    dp[i][b] = cost
                    split[i][b] = j

    # backtrack
    boundaries_idx = []
    i, b = M, B
    while b > 1:
        j = split[i][b]
        boundaries_idx.append(j)
        i, b = j, b - 1
    boundaries_idx.reverse()

    boundaries = [unique_scores[idx] for idx in boundaries_idx]
    return boundaries, dp[M][B]


def dp_loglik(num_buckets: int):
    """
    DP to find bucket boundaries maximising total log-likelihood.

    State:  dp[i][b] = max log-likelihood using b buckets over scores [0..i)
    Trans:  dp[i][b] = max over j<i  of  dp[j][b-1] + seg_ll_gain(j, i)
    Returns list of boundary FICO scores.
    """
    B = num_buckets
    NEG_INF = -float('inf')
    dp    = [[NEG_INF] * (B + 1) for _ in range(M + 1)]
    split = [[0]       * (B + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for b in range(1, B + 1):
        for i in range(b, M + 1):
            for j in range(b - 1, i):
                val = dp[j][b - 1] + seg_ll_gain(j, i)
                if val > dp[i][b]:
                    dp[i][b] = val
                    split[i][b] = j

    # backtrack
    boundaries_idx = []
    i, b = M, B
    while b > 1:
        j = split[i][b]
        boundaries_idx.append(j)
        i, b = j, b - 1
    boundaries_idx.reverse()

    boundaries = [unique_scores[idx] for idx in boundaries_idx]
    return boundaries, dp[M][B]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  RATING MAP BUILDER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_rating_map(boundaries: list, fico_min: int, fico_max: int,
                     method: str) -> pd.DataFrame:
    """
    Converts a list of boundary points into a human-readable rating map.
    Rating 1 = best credit (highest FICO), Rating N = worst credit.

    boundaries: interior cut-points (len = num_buckets - 1)
    """
    cuts = [fico_min] + sorted(boundaries) + [fico_max + 1]
    rows = []
    num_buckets = len(cuts) - 1
    for i in range(num_buckets):
        lo, hi = cuts[i], cuts[i + 1] - 1
        mask   = (fico >= lo) & (fico <= hi)
        n_b    = mask.sum()
        k_b    = default[mask].sum()
        pd_b   = k_b / n_b if n_b > 0 else 0.0
        rows.append({
            'Rating':      num_buckets - i,       # 1 = best (highest FICO)
            'FICO_Lower':  lo,
            'FICO_Upper':  hi,
            'FICO_Range':  f"{lo}â€“{hi}",
            'Count':       n_b,
            'Defaults':    k_b,
            'PD':          round(pd_b, 4),
            'Method':      method,
        })
    return pd.DataFrame(rows).sort_values('Rating')


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  BENCHMARK â€” Equal-Width & Equal-Frequency (baselines)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def equal_width_boundaries(num_buckets, fmin, fmax):
    step = (fmax - fmin) / num_buckets
    return [int(fmin + step * i) for i in range(1, num_buckets)]

def equal_freq_boundaries(num_buckets, scores):
    quantiles = np.linspace(0, 100, num_buckets + 1)[1:-1]
    return sorted(set(int(np.percentile(scores, q)) for q in quantiles))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  RUN FOR DIFFERENT BUCKET COUNTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("\n  Running DP optimisation (this may take ~30 s for large ranges)...")

NUM_BUCKETS_LIST = [5, 7, 10]
all_maps = {}

for nb in NUM_BUCKETS_LIST:
    print(f"\n  â”€â”€ {nb} buckets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")

    # MSE
    mse_bounds, mse_val = dp_mse(nb)
    mse_map = build_rating_map(mse_bounds, fico_min, fico_max, 'MSE-DP')
    print(f"  MSE  boundaries: {mse_bounds}  (total MSE = {mse_val:,.0f})")

    # Log-Likelihood
    ll_bounds, ll_val = dp_loglik(nb)
    ll_map = build_rating_map(ll_bounds, fico_min, fico_max, 'LogLik-DP')
    print(f"  LogL boundaries: {ll_bounds}  (log-lik = {ll_val:.2f})")

    # Baselines
    ew_bounds = equal_width_boundaries(nb, fico_min, fico_max)
    ew_map    = build_rating_map(ew_bounds, fico_min, fico_max, 'Equal-Width')

    ef_bounds = equal_freq_boundaries(nb, fico)
    ef_map    = build_rating_map(ef_bounds, fico_min, fico_max, 'Equal-Freq')

    all_maps[nb] = {
        'MSE-DP':     (mse_map, mse_bounds),
        'LogLik-DP':  (ll_map,  ll_bounds),
        'Equal-Width':(ew_map,  ew_bounds),
        'Equal-Freq': (ef_map,  ef_bounds),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7.  PRINT RATING MAPS (5-bucket case â€” canonical output)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

CANONICAL = 5
print(f"\n{'='*64}")
print(f"  RATING MAPS  ({CANONICAL} buckets)  â€”  Rating 1=Best, {CANONICAL}=Worst")
print(f"{'='*64}")

for method, (rmap, bounds) in all_maps[CANONICAL].items():
    print(f"\n  [{method}]  boundaries: {bounds}")
    print(rmap[['Rating','FICO_Range','Count','Defaults','PD']].to_string(index=False))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8.  COMPARISON TABLE â€” information value per method
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def information_value(rmap: pd.DataFrame) -> float:
    """
    IV = sum_i (WoE_i * (dist_events_i - dist_non_events_i))
    Measures how well the bucketing separates defaulters from non-defaulters.
    IV > 0.3 = strong predictor.
    """
    total_d   = rmap['Defaults'].sum()
    total_nd  = (rmap['Count'] - rmap['Defaults']).sum()
    iv = 0.0
    for _, row in rmap.iterrows():
        d  = row['Defaults']
        nd = row['Count'] - row['Defaults']
        if d == 0 or nd == 0: continue
        pct_d  = d  / total_d
        pct_nd = nd / total_nd
        woe    = np.log(pct_d / pct_nd)
        iv    += (pct_d - pct_nd) * woe
    return round(iv, 4)

print(f"\n{'â”€'*64}")
print(f"  INFORMATION VALUE COMPARISON  (all bucket sizes)")
print(f"{'â”€'*64}")
print(f"  {'Method':<14}", end='')
for nb in NUM_BUCKETS_LIST:
    print(f"  {nb}-bucket", end='')
print()

for method in ['MSE-DP', 'LogLik-DP', 'Equal-Width', 'Equal-Freq']:
    print(f"  {method:<14}", end='')
    for nb in NUM_BUCKETS_LIST:
        rmap, _ = all_maps[nb][method]
        iv = information_value(rmap)
        print(f"  {iv:>8.4f}", end='')
    print()

print(f"\n  (IV > 0.3 = strong predictor  |  IV > 0.5 = very strong)")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 9.  VISUALISATIONS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

method_colors = {
    'MSE-DP':     C[0],
    'LogLik-DP':  C[1],
    'Equal-Width':C[2],
    'Equal-Freq': C[3],
}

# â”€â”€ Fig 1: FICO distribution + default rate curve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig1, (ax_hist, ax_pd) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig1.suptitle('FICO Score Distribution & Default Rate',
               color=TEXT, fontsize=14, fontweight='bold')

# histogram
ax_hist.hist(fico[default == 0], bins=60, alpha=0.65, color=C[0],
             label='No Default', density=True)
ax_hist.hist(fico[default == 1], bins=60, alpha=0.65, color=C[1],
             label='Default',    density=True)
ax_hist.set_ylabel('Density'); ax_hist.grid(True, alpha=0.4)
ax_hist.legend(fontsize=9)
ax_hist.set_title('Distribution by Default Status', color=TEXT, fontsize=11)

# rolling default rate
scores_sorted = np.arange(fico_min, fico_max + 1)
window = 20
roll_pd = []
roll_x  = []
for s in scores_sorted:
    lo, hi = s - window, s + window
    mask = (fico >= lo) & (fico <= hi)
    if mask.sum() >= 10:
        roll_pd.append(default[mask].mean())
        roll_x.append(s)

ax_pd.plot(roll_x, roll_pd, color=C[3], lw=2, label='Rolling PD (Â±20 pts)')
ax_pd.axhline(default.mean(), color=MUTED, ls='--', lw=1,
              label=f'Overall PD = {default.mean():.2%}')
ax_pd.set_ylabel('Default Rate'); ax_pd.set_xlabel('FICO Score')
ax_pd.grid(True, alpha=0.4); ax_pd.legend(fontsize=9)
ax_pd.set_title('Rolling Default Rate by FICO Score', color=TEXT, fontsize=11)
ax_pd.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
fig1.savefig('/mnt/user-data/outputs/fico_fig1_distribution.png',
             dpi=140, bbox_inches='tight', facecolor=BG)
print("\n  Fig 1 saved.")


# â”€â”€ Fig 2: Bucket PD profiles â€” 5-bucket canonical â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle(f'Bucket PD Profiles â€” {CANONICAL} Buckets per Method',
               color=TEXT, fontsize=14, fontweight='bold')

for ax, (method, (rmap, bounds)) in zip(axes.flat, all_maps[CANONICAL].items()):
    col = method_colors[method]
    ratings = rmap['Rating'].values
    pds     = rmap['PD'].values
    counts  = rmap['Count'].values

    bars = ax.bar(ratings, pds, color=col, alpha=0.8,
                   edgecolor=BG, width=0.6)

    # annotate count
    for bar, cnt, pd_val in zip(bars, counts, pds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'n={cnt:,}\n{pd_val:.1%}', ha='center', va='bottom',
                fontsize=7.5, color=TEXT)

    # boundary lines
    ax.set_title(f'{method}  (IV={information_value(rmap):.3f})',
                  color=TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Rating  (1=Best, 5=Worst)')
    ax.set_ylabel('Probability of Default')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3, axis='y')

    ranges = rmap['FICO_Range'].values
    ax.set_xticks(ratings)
    ax.set_xticklabels([f'R{r}\n{rng}' for r, rng in zip(ratings, ranges)], fontsize=7)

plt.tight_layout()
fig2.savefig('/mnt/user-data/outputs/fico_fig2_bucket_profiles.png',
             dpi=140, bbox_inches='tight', facecolor=BG)
print("  Fig 2 saved.")


# â”€â”€ Fig 3: IV comparison across bucket sizes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig3, ax = plt.subplots(figsize=(12, 6))
fig3.suptitle('Information Value vs Number of Buckets',
               color=TEXT, fontsize=14, fontweight='bold')

for method, col in method_colors.items():
    ivs = [information_value(all_maps[nb][method][0]) for nb in NUM_BUCKETS_LIST]
    ax.plot(NUM_BUCKETS_LIST, ivs, 'o-', color=col, lw=2.5, ms=8, label=method)

ax.axhline(0.3, color=MUTED, ls='--', lw=1, label='Strong (IV=0.3)')
ax.axhline(0.5, color=MUTED, ls=':',  lw=1, label='Very Strong (IV=0.5)')
ax.set_xlabel('Number of Buckets'); ax.set_ylabel('Information Value')
ax.set_xticks(NUM_BUCKETS_LIST)
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

plt.tight_layout()
fig3.savefig('/mnt/user-data/outputs/fico_fig3_iv_comparison.png',
             dpi=140, bbox_inches='tight', facecolor=BG)
print("  Fig 3 saved.")


# â”€â”€ Fig 4: Boundary placement comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig4, ax = plt.subplots(figsize=(14, 5))
fig4.suptitle(f'Boundary Placement â€” {CANONICAL} Buckets',
               color=TEXT, fontsize=14, fontweight='bold')

# background: FICO density
ax2 = ax.twinx()
ax2.hist(fico, bins=80, color='#ffffff', alpha=0.06, density=True)
ax2.set_yticks([]); ax2.set_ylabel('')

y_positions = {'MSE-DP': 0.82, 'LogLik-DP': 0.64,
               'Equal-Width': 0.46, 'Equal-Freq': 0.28}

for method, (_, bounds) in all_maps[CANONICAL].items():
    col = method_colors[method]
    y   = y_positions[method]
    ax.axhline(y, color=col, lw=0.5, alpha=0.3,
               xmin=(fico_min - fico_min)/(fico_max - fico_min),
               xmax=1.0)
    all_bounds = [fico_min] + sorted(bounds) + [fico_max]
    for i, (lo, hi) in enumerate(zip(all_bounds[:-1], all_bounds[1:])):
        mid = (lo + hi) / 2
        ax.barh(y, hi - lo, left=lo, height=0.14, color=col, alpha=0.55,
                edgecolor=TEXT, linewidth=0.8)
        ax.text(mid, y, f'R{CANONICAL - i}', ha='center', va='center',
                fontsize=8, color=TEXT, fontweight='bold')
    for b in bounds:
        ax.axvline(b, color=col, lw=1.5, alpha=0.7, ls='--')
    ax.text(fico_min - 3, y, method, ha='right', va='center',
            color=col, fontsize=9, fontweight='bold')

ax.set_xlim(fico_min - 40, fico_max + 5)
ax.set_ylim(0.12, 1.0)
ax.set_xlabel('FICO Score'); ax.set_yticks([])
ax.grid(True, alpha=0.25, axis='x')

plt.tight_layout()
fig4.savefig('/mnt/user-data/outputs/fico_fig4_boundaries.png',
             dpi=140, bbox_inches='tight', facecolor=BG)
print("  Fig 4 saved.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 10.  FINAL RATING MAP (recommended: LogLik-DP, 5 buckets)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

RECOMMENDED_METHOD = 'LogLik-DP'
RECOMMENDED_N      = 5
rec_map, rec_bounds = all_maps[RECOMMENDED_N][RECOMMENDED_METHOD]

print(f"\n{'='*64}")
print(f"  RECOMMENDED RATING MAP  ({RECOMMENDED_METHOD}, {RECOMMENDED_N} buckets)")
print(f"  Boundaries: {rec_bounds}")
print(f"{'='*64}")
print(rec_map[['Rating','FICO_Range','Count','Defaults','PD']].to_string(index=False))
print(f"\n  Information Value: {information_value(rec_map):.4f}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 11.  GENERALISED FUNCTION FOR CHARLIE'S PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_fico_rating_map(
    fico_scores: np.ndarray,
    defaults:    np.ndarray,
    num_buckets: int  = 5,
    method:      str  = 'log_likelihood',   # 'log_likelihood' | 'mse'
) -> tuple:
    """
    Generate an optimal FICO â†’ Rating mapping.

    Parameters
    ----------
    fico_scores : array of integer FICO scores
    defaults    : array of 0/1 default flags (same length)
    num_buckets : number of rating buckets to create
    method      : 'log_likelihood'  maximises within-bucket PD homogeneity
                  'mse'             minimises within-bucket FICO dispersion

    Returns
    -------
    rating_map  : pd.DataFrame with columns
                    Rating, FICO_Lower, FICO_Upper, Count, Defaults, PD
    boundaries  : list of interior cut-points (len = num_buckets - 1)
    score_fn    : callable  fico_score -> rating (int)

    Rating convention: 1 = best credit (highest FICO)
                       N = worst credit  (lowest FICO)
    """
    assert len(fico_scores) == len(defaults), "Arrays must match in length"
    assert num_buckets >= 2,                  "Need at least 2 buckets"
    assert method in ('log_likelihood', 'mse')

    fmin = int(fico_scores.min())
    fmax = int(fico_scores.max())
    uniq = np.arange(fmin, fmax + 1)
    Mu   = len(uniq)

    s2i = {s: i for i, s in enumerate(uniq)}
    n_a = np.zeros(Mu, dtype=int)
    k_a = np.zeros(Mu, dtype=int)
    for s, d in zip(fico_scores, defaults):
        n_a[s2i[s]] += 1
        k_a[s2i[s]] += d

    pn  = np.concatenate([[0], np.cumsum(n_a)])
    pk  = np.concatenate([[0], np.cumsum(k_a)])
    ps  = np.concatenate([[0], np.cumsum(n_a * uniq)])
    ps2 = np.concatenate([[0], np.cumsum(n_a * uniq ** 2)])

    def g_mse(a, b):
        n = pn[b] - pn[a]
        if n == 0: return 0.0
        s = ps[b] - ps[a]; s2 = ps2[b] - ps2[a]
        return s2 - n * (s / n) ** 2

    def g_ll(a, b):
        n = pn[b] - pn[a]; k = pk[b] - pk[a]
        if n == 0 or k == 0 or k == n: return 0.0
        p = k / n
        return k * np.log(p) + (n - k) * np.log(1.0 - p)

    g = g_mse if method == 'mse' else g_ll
    sign = -1 if method == 'mse' else 1   # DP maximises; negate MSE

    B   = num_buckets
    INF = float('inf')
    dp    = [[-INF] * (B + 1) for _ in range(Mu + 1)]
    split = [[0]    * (B + 1) for _ in range(Mu + 1)]
    dp[0][0] = 0.0

    for b in range(1, B + 1):
        for i in range(b, Mu + 1):
            for j in range(b - 1, i):
                val = dp[j][b - 1] + sign * g(j, i)
                if val > dp[i][b]:
                    dp[i][b] = val
                    split[i][b] = j

    idx_list = []
    ii, bb = Mu, B
    while bb > 1:
        jj = split[ii][bb]
        idx_list.append(jj)
        ii, bb = jj, bb - 1
    idx_list.reverse()
    boundaries = [uniq[idx] for idx in idx_list]

    cuts = [fmin] + sorted(boundaries) + [fmax + 1]
    rows = []
    Nb = len(cuts) - 1
    for i in range(Nb):
        lo, hi = cuts[i], cuts[i + 1] - 1
        mask = (fico_scores >= lo) & (fico_scores <= hi)
        n_b  = int(mask.sum())
        k_b  = int(defaults[mask].sum())
        rows.append({
            'Rating':     Nb - i,
            'FICO_Lower': lo,
            'FICO_Upper': hi,
            'Count':      n_b,
            'Defaults':   k_b,
            'PD':         round(k_b / n_b if n_b > 0 else 0.0, 4),
        })

    rating_map = pd.DataFrame(rows).sort_values('Rating').reset_index(drop=True)

    def score_fn(fico_val: int) -> int:
        for _, row in rating_map.iterrows():
            if row['FICO_Lower'] <= fico_val <= row['FICO_Upper']:
                return int(row['Rating'])
        return Nb  # fallback: worst rating

    return rating_map, boundaries, score_fn


# â”€â”€ Quick demo of the generalised function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*64}")
print("  GENERALISED FUNCTION DEMO")
print(f"{'â”€'*64}")

rmap, bounds, score_fn = generate_fico_rating_map(
    fico, default, num_buckets=5, method='log_likelihood'
)
print(f"\n  Optimal boundaries (log-likelihood): {bounds}")
print(rmap.to_string(index=False))

test_scores = [850, 740, 680, 620, 490]
print(f"\n  Sample score â†’ rating mappings:")
for s in test_scores:
    print(f"    FICO {s:3d}  â†’  Rating {score_fn(s)}")

print(f"\n{'='*64}")
print("  All outputs saved to /mnt/user-data/outputs/")
print(f"{'='*64}")