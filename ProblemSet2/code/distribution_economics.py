"""
ECON 2080 — Problem Set 2: Distribution Economics
===================================================
This script implements a simplified Aiyagari (1994) / Huggett (1993)
incomplete-markets heterogeneous-agent model and produces key distributional
figures:

    1. Policy Function (Savings) — how optimal savings/assets next period
       depend on current assets and income state.
    2. Lorenz Curve — graphical representation of the cumulative wealth
       distribution, used to measure inequality.
    3. Stationary Asset Distribution — the ergodic distribution of wealth
       (assets) across households in the steady state.
    4. Gini Coefficient vs. Borrowing Constraint — how tightening or
       relaxing the borrowing constraint affects wealth inequality (Gini).

Economic Setting
----------------
A continuum of ex-ante identical households faces idiosyncratic labour-income
risk.  Insurance markets are incomplete: households can only self-insure by
saving in a single risk-free asset.  A lower bound (borrowing constraint) on
assets captures the inability to borrow against future labour income.

In Huggett (1993) the asset is a pure zero-net-supply bond priced at q = 1/(1+r).
In Aiyagari (1994) the asset is physical capital which is rented to a
representative firm. Here we follow a Huggett-style partial-equilibrium
specification where the interest rate r is taken as exogenous.

References
----------
Aiyagari, S. R. (1994). Uninsured idiosyncratic risk and aggregate saving.
    *Quarterly Journal of Economics*, 109(3), 659–684.
Huggett, M. (1993). The risk-free rate in heterogeneous-agent incomplete-
    insurance economies. *Journal of Economic Dynamics and Control*,
    17(5–6), 953–969.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# 0. Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Model Parameters
# ---------------------------------------------------------------------------
# Preferences
SIGMA   = 2.0    # Coefficient of relative risk aversion (u(c) = c^(1-σ)/(1-σ))
BETA    = 0.95   # Discount factor (quarterly: β = 0.95 → annual ≈ 0.815)

# Endowment / income process
R       = 0.04   # Exogenous risk-free interest rate (annual)
W       = 1.0    # Wage (labour endowment × wage rate; normalised to 1 in H-state)

# Income states: z ∈ {z_L, z_H} (low and high labour productivity)
Z_VALS  = np.array([0.5, 1.5])   # productivity levels (z_L, z_H)

# Markov transition matrix for income: P[i,j] = Prob(z_t+1=j | z_t=i)
#   Row 0 → from z_L: stay with prob 0.9, switch with prob 0.1
#   Row 1 → from z_H: switch with prob 0.1, stay with prob 0.9
P_TRANS = np.array([[0.9, 0.1],
                    [0.1, 0.9]])

# Asset grid
A_MIN   = -0.5   # Borrowing constraint (natural debt limit approximation)
A_MAX   =  8.0   # Upper bound on assets (large enough to be non-binding)
N_A     = 300    # Number of grid points for assets

# Computational parameters
VFI_TOL    = 1e-6   # Convergence tolerance for value function iteration
VFI_MAXITER = 2000  # Maximum number of VFI iterations
SIM_T      = 3000   # Length of simulation path for distribution
SIM_N      = 5000   # Number of simulated households
RNG_SEED   = 42     # Random seed for reproducibility

# ---------------------------------------------------------------------------
# 2. Asset Grid and Utility
# ---------------------------------------------------------------------------

def make_asset_grid(a_min: float = A_MIN,
                    a_max: float = A_MAX,
                    n: int = N_A) -> np.ndarray:
    """
    Create a refined asset grid denser near the borrowing constraint.

    We use a combination of a linear grid and a grid denser near zero so
    that the kink of the policy function near the constraint is well
    approximated.

    Parameters
    ----------
    a_min : float — lower bound on assets (borrowing constraint)
    a_max : float — upper bound on assets
    n     : int   — number of grid points

    Returns
    -------
    np.ndarray of shape (n,)
    """
    # Simple uniform grid (adequate for this illustration)
    return np.linspace(a_min, a_max, n)


def utility(c: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """
    CRRA (constant relative risk aversion) utility function.

        u(c) = c^(1−σ) / (1−σ)   if σ ≠ 1
             = ln(c)               if σ = 1

    We restrict c > 0 to avoid undefined values; consumption must be
    non-negative for a household with a borrowing constraint.

    Parameters
    ----------
    c     : array-like — consumption levels (must be > 0)
    sigma : float — coefficient of relative risk aversion

    Returns
    -------
    np.ndarray — per-period utility values
    """
    c = np.asarray(c, dtype=float)
    if sigma == 1.0:
        return np.log(np.maximum(c, 1e-10))
    return (np.maximum(c, 1e-10) ** (1.0 - sigma) - 1.0) / (1.0 - sigma)


# ---------------------------------------------------------------------------
# 3. Value Function Iteration (VFI)
# ---------------------------------------------------------------------------

def solve_vfi(a_grid: np.ndarray,
              z_vals: np.ndarray = Z_VALS,
              p_trans: np.ndarray = P_TRANS,
              r: float = R,
              w: float = W,
              beta: float = BETA,
              sigma: float = SIGMA,
              tol: float = VFI_TOL,
              max_iter: int = VFI_MAXITER):
    """
    Solve the household's Bellman equation by Value Function Iteration.

    The household's problem each period is:
        V(a, z) = max_{a'} [ u(c) + β · Σ_z' P(z,z') · V(a', z') ]
        s.t.  c = (1 + r) a + w z - a'
              a' ≥ A_MIN   (borrowing constraint)
              c   ≥ 0

    We iterate over the Bellman operator T until ‖V_{n+1} - V_n‖_∞ < tol.

    Parameters
    ----------
    a_grid   : array of shape (N_A,) — asset grid
    z_vals   : array of shape (N_Z,) — income states
    p_trans  : array of shape (N_Z, N_Z) — Markov transition matrix
    r        : float — interest rate
    w        : float — wage
    beta     : float — discount factor
    sigma    : float — risk aversion
    tol      : float — convergence criterion
    max_iter : int   — maximum iterations

    Returns
    -------
    V_new  : array (N_A, N_Z) — converged value function
    pol_a  : array (N_A, N_Z) — optimal next-period assets a'(a, z)
    pol_c  : array (N_A, N_Z) — optimal consumption c(a, z)
    """
    n_a  = len(a_grid)
    n_z  = len(z_vals)

    # Initialise value function: naive guess = utility at zero savings
    V = np.zeros((n_a, n_z))

    # Pre-compute cash-on-hand for each (a, z) combination
    # cash[i, j] = (1 + r) * a[i] + w * z[j]
    cash = (1.0 + r) * a_grid[:, None] + w * z_vals[None, :]  # shape (N_A, N_Z)

    # Pre-compute utility matrix for all (a, z, a') combinations
    # util_mat[i, j, k] = u(cash[i,j] - a_grid[k])
    # This is memory-intensive for large grids but pedagogically clear.
    c_mat = cash[:, :, None] - a_grid[None, None, :]   # (N_A, N_Z, N_A)
    util_mat = utility(c_mat, sigma)
    util_mat[c_mat <= 0] = -1e12   # Penalise infeasible consumption

    for iteration in range(max_iter):
        # Expected continuation value: EV[k, j] = Σ_j' P[j, j'] V[k, j']
        EV = V @ p_trans.T   # shape (N_A, N_Z)

        # Bellman operator: for each (i, j), maximise over k (choice of a')
        # Q[i, j, k] = util_mat[i, j, k] + β * EV[k, j]
        # EV has shape (N_A, N_Z); EV[k, j] maps to axis (k=2, j=1) of Q,
        # so we transpose to (N_Z, N_A) and add a leading broadcast dimension.
        Q = util_mat + beta * EV.T[None, :, :]   # (N_A, N_Z, N_A)

        # Apply borrowing constraint: a' must be on the grid and ≥ A_MIN
        # (Since a_grid[0] = A_MIN by construction, no extra masking needed.)

        V_new   = Q.max(axis=2)          # (N_A, N_Z)
        pol_idx = Q.argmax(axis=2)       # index of optimal a'

        # Check convergence
        diff = np.max(np.abs(V_new - V))
        V = V_new.copy()

        if diff < tol:
            print(f"  VFI converged in {iteration + 1} iterations (diff = {diff:.2e})")
            break
    else:
        print(f"  VFI did NOT converge after {max_iter} iterations (diff = {diff:.2e})")

    # Extract policy functions
    pol_a = a_grid[pol_idx]          # (N_A, N_Z) — optimal next-period assets
    pol_c = cash - pol_a             # (N_A, N_Z) — optimal consumption

    return V, pol_a, pol_c


# ---------------------------------------------------------------------------
# 4. Stationary Distribution via Simulation
# ---------------------------------------------------------------------------

def simulate_stationary_dist(pol_a: np.ndarray,
                              a_grid: np.ndarray,
                              z_vals: np.ndarray = Z_VALS,
                              p_trans: np.ndarray = P_TRANS,
                              n_hh: int = SIM_N,
                              n_t: int = SIM_T,
                              seed: int = RNG_SEED) -> tuple:
    """
    Simulate the stationary (ergodic) asset distribution.

    We simulate a panel of `n_hh` households for `n_t` periods.  Policy
    functions are interpolated to allow for off-grid asset choices.  After
    discarding the first half of the simulation (burn-in), we collect the
    cross-sectional asset distribution.

    Parameters
    ----------
    pol_a  : (N_A, N_Z) — optimal next-period assets on the grid
    a_grid : (N_A,) — asset grid
    z_vals : (N_Z,) — income states
    p_trans: (N_Z, N_Z) — income transition matrix
    n_hh   : int — number of simulated households
    n_t    : int — simulation length per household
    seed   : int — RNG seed

    Returns
    -------
    a_sim  : np.ndarray (n_hh,) — cross-sectional asset holdings (end of simulation)
    z_sim  : np.ndarray (n_hh,) — cross-sectional income states
    """
    rng = np.random.default_rng(seed)
    n_z = len(z_vals)

    # Build interpolating policy functions for each income state
    pol_interp = [
        interp1d(a_grid, pol_a[:, j], kind="linear", fill_value="extrapolate")
        for j in range(n_z)
    ]

    # Initialise: all households start at the mean of the asset grid
    a_cur = np.full(n_hh, a_grid.mean())
    z_cur = rng.integers(0, n_z, size=n_hh)   # Random initial income state

    # Simulate
    cumprob = np.cumsum(p_trans, axis=1)   # CDF of transitions row by row

    for t in range(n_t):
        # Update assets using policy function (interpolated)
        a_next = np.array([pol_interp[z_cur[i]](a_cur[i]) for i in range(n_hh)])
        # Enforce borrowing constraint (numerical safety)
        a_next = np.maximum(a_next, a_grid[0])

        # Update income state via Markov chain
        draws = rng.random(n_hh)
        z_next = np.array([
            np.searchsorted(cumprob[z_cur[i]], draws[i])
            for i in range(n_hh)
        ])
        z_next = np.clip(z_next, 0, n_z - 1)

        a_cur = a_next
        z_cur = z_next

    return a_cur, z_cur


# ---------------------------------------------------------------------------
# 5. Lorenz Curve and Gini Coefficient
# ---------------------------------------------------------------------------

def lorenz_curve(x: np.ndarray) -> tuple:
    """
    Compute the Lorenz curve from a vector of non-negative values.

    The Lorenz curve plots the cumulative share of total wealth held by the
    bottom p% of the population, for p ∈ [0, 1].  Perfect equality
    corresponds to the 45-degree line; greater bowing below the diagonal
    indicates higher inequality.

    Parameters
    ----------
    x : np.ndarray — cross-sectional wealth (or income) values

    Returns
    -------
    pop_share   : np.ndarray — cumulative population share (x-axis)
    wealth_share: np.ndarray — cumulative wealth share (y-axis)
    """
    x_sorted = np.sort(x)
    # Shift so all values are non-negative (assets can be negative with borrowing)
    x_shifted = x_sorted - x_sorted.min() + 1e-8
    n = len(x_shifted)
    pop_share    = np.linspace(0, 1, n + 1)
    wealth_cum   = np.concatenate([[0.0], np.cumsum(x_shifted) / x_shifted.sum()])
    return pop_share, wealth_cum


def gini_coefficient(x: np.ndarray) -> float:
    """
    Compute the Gini coefficient from a vector of non-negative values.

    The Gini coefficient equals twice the area between the Lorenz curve and
    the 45-degree line of perfect equality.  It ranges from 0 (perfect
    equality) to 1 (maximum inequality).

    Parameters
    ----------
    x : np.ndarray — non-negative values

    Returns
    -------
    float — Gini coefficient ∈ [0, 1]
    """
    x_shifted = x - x.min() + 1e-8
    x_sorted  = np.sort(x_shifted)
    n = len(x_sorted)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * x_sorted) / (n * x_sorted.sum())) - (n + 1.0) / n


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def plot_policy_function(pol_a: np.ndarray, pol_c: np.ndarray,
                          a_grid: np.ndarray) -> str:
    """
    Plot the optimal savings and consumption policy functions.

    The savings policy a'(a, z) shows that households save more when they
    have more assets or higher income. The 45-degree line (a' = a) marks
    the threshold above which households are dis-saving (moving toward their
    bliss point).

    Parameters
    ----------
    pol_a  : (N_A, N_Z) — optimal next-period assets
    pol_c  : (N_A, N_Z) — optimal consumption
    a_grid : (N_A,)     — asset grid
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = ["steelblue", "darkorange"]
    labels = [r"Low income ($z_L$)", r"High income ($z_H$)"]

    for j in range(2):
        axes[0].plot(a_grid, pol_a[:, j], color=colors[j], lw=2, label=labels[j])
        axes[1].plot(a_grid, pol_c[:, j], color=colors[j], lw=2, label=labels[j])

    # 45-degree line: if a' = a, the household is neither saving nor borrowing
    axes[0].plot(a_grid, a_grid, "k--", lw=1, alpha=0.5, label="45° line (a'= a)")
    axes[0].axhline(A_MIN, color="grey", lw=0.8, linestyle=":", alpha=0.6)

    axes[0].set_xlabel("Current Assets  a", fontsize=12)
    axes[0].set_ylabel("Next-Period Assets  a'(a, z)", fontsize=12)
    axes[0].set_title("Savings Policy Function", fontsize=13)
    axes[0].legend(fontsize=10)

    axes[1].set_xlabel("Current Assets  a", fontsize=12)
    axes[1].set_ylabel("Consumption  c(a, z)", fontsize=12)
    axes[1].set_title("Consumption Policy Function", fontsize=13)
    axes[1].legend(fontsize=10)

    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "policy_functions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lorenz_and_distribution(a_sim: np.ndarray) -> str:
    """
    Plot the stationary asset distribution (histogram) and the Lorenz curve.

    Parameters
    ----------
    a_sim : (N_HH,) — cross-sectional asset holdings from simulation
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left panel: stationary distribution histogram ---
    axes[0].hist(a_sim, bins=80, density=True, color="steelblue",
                 edgecolor="white", alpha=0.8)
    axes[0].axvline(np.mean(a_sim),  color="crimson",    lw=2, linestyle="--",
                    label=f"Mean = {np.mean(a_sim):.2f}")
    axes[0].axvline(np.median(a_sim), color="darkorange", lw=2, linestyle=":",
                    label=f"Median = {np.median(a_sim):.2f}")
    axes[0].set_xlabel("Assets  a", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Stationary Asset Distribution", fontsize=13)
    axes[0].legend(fontsize=10)

    # --- Right panel: Lorenz curve ---
    pop_share, wealth_share = lorenz_curve(a_sim)
    gini = gini_coefficient(a_sim)

    axes[1].plot(pop_share, wealth_share, color="steelblue", lw=2.5,
                 label=f"Lorenz Curve (Gini = {gini:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Perfect equality")
    axes[1].fill_between(pop_share, wealth_share, pop_share, alpha=0.15, color="steelblue")
    axes[1].set_xlabel("Cumulative Population Share", fontsize=12)
    axes[1].set_ylabel("Cumulative Wealth Share", fontsize=12)
    axes[1].set_title("Lorenz Curve — Wealth Distribution", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "lorenz_and_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gini_vs_constraint() -> str:
    """
    Plot how the Gini coefficient varies with the borrowing constraint.

    A tighter borrowing constraint (higher a_min) forces low-wealth households
    to hold more assets as a buffer, compressing the lower tail of the
    distribution and reducing the Gini coefficient.  Conversely, relaxing the
    constraint (more negative a_min) allows more heterogeneity and raises the Gini.
    """
    a_min_values = np.linspace(-2.0, 0.5, 8)   # Range of borrowing constraints
    gini_values  = []

    a_grid_base = make_asset_grid()   # Baseline grid (recalculated per constraint below)

    for a_min_val in a_min_values:
        a_grid_i = np.linspace(a_min_val, A_MAX, N_A)
        print(f"  Solving VFI for a_min = {a_min_val:.2f} …")
        _, pol_a_i, _ = solve_vfi(a_grid_i)
        a_sim_i, _    = simulate_stationary_dist(pol_a_i, a_grid_i, n_hh=2000, n_t=1500)
        gini_values.append(gini_coefficient(a_sim_i))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(a_min_values, gini_values, "o-", color="steelblue", lw=2.5, ms=6)
    ax.set_xlabel("Borrowing Constraint  $\\bar{a}$ (lower = tighter)", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title("Wealth Inequality vs. Borrowing Constraint", fontsize=13)
    ax.invert_xaxis()   # Show tighter constraint on the left
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "gini_vs_constraint.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7. Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Problem Set 2 — Distribution Economics")
    print("=" * 50)

    # Build the asset grid
    a_grid = make_asset_grid()

    # Solve the household problem via VFI
    print("\nRunning Value Function Iteration …")
    V, pol_a, pol_c = solve_vfi(a_grid)

    # Simulate the stationary distribution
    print("\nSimulating stationary distribution …")
    a_sim, z_sim = simulate_stationary_dist(pol_a, a_grid)

    # Produce and save all figures
    print("\nProducing figures …")
    paths = [
        plot_policy_function(pol_a, pol_c, a_grid),
        plot_lorenz_and_distribution(a_sim),
        plot_gini_vs_constraint(),
    ]

    for p in paths:
        print(f"  Saved: {p}")

    print("\nDone. All figures saved to ProblemSet2/results/")
