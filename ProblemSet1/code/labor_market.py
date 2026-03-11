"""
ECON 2080 — Problem Set 1: Labor Market Models
================================================
This script implements the Mortensen-Pissarides (1994) search-and-matching
model of the labor market and produces key figures:

    1. Beveridge Curve — the negative relationship between vacancies and
       unemployment in steady state as we vary labor-market tightness.
    2. Job-Finding & Job-Filling Rates — how the probability that an
       unemployed worker finds a job (p) and the probability that an open
       vacancy is filled (q) depend on market tightness (θ = v/u).
    3. Equilibrium Tightness — the (JC) job-creation and (WC) wage-
       determination curves that pin down equilibrium θ and wages w.
    4. Unemployment-Insurance Policy Experiment — comparative statics
       showing how a higher replacement rate (b) shifts equilibrium and
       raises the unemployment rate.

References
----------
Mortensen, D. T., & Pissarides, C. A. (1994). Job Creation and Job
Destruction in the Theory of Unemployment. *Review of Economic Studies*,
61(3), 397–415.

Pissarides, C. A. (2000). *Equilibrium Unemployment Theory* (2nd ed.).
MIT Press.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ---------------------------------------------------------------------------
# 0. Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Model Parameters
# ---------------------------------------------------------------------------
# These are standard calibration values used in the MP literature.
# Time unit: one quarter.

ALPHA   = 0.5    # Elasticity of the matching function with respect to unemployment
                 # (Cobb-Douglas matching function M = m * u^α * v^(1-α))
M_EFF   = 0.6    # Matching efficiency parameter (scale of the matching function)
DELTA   = 0.10   # Exogenous job-destruction (separation) rate per quarter
R       = 0.01   # Discount rate (risk-free interest rate per quarter)
BETA    = 0.5    # Worker's Nash bargaining power (symmetric: β = 1 - β = 0.5)
Y       = 1.0    # Output per worker in a filled job (normalised to 1)
C_VANCY = 0.3    # Cost of posting and maintaining one vacancy per period
B_BASE  = 0.4    # Unemployment-insurance replacement rate (baseline)


# ---------------------------------------------------------------------------
# 2. Matching Technology
# ---------------------------------------------------------------------------

def matching_function(u: float, v: float, m: float = M_EFF, alpha: float = ALPHA) -> float:
    """
    Cobb-Douglas matching function: M(u, v) = m * u^α * v^(1-α).

    The matching function captures the flow of new employment relationships
    formed per period given a stock of unemployed workers (u) and vacant
    jobs (v). It is assumed to be increasing in both arguments and
    homogeneous of degree 1 (constant returns to scale in matching).

    Parameters
    ----------
    u     : float or array — unemployment (number or rate)
    v     : float or array — vacancies (number or rate)
    m     : float — matching efficiency
    alpha : float — elasticity w.r.t. unemployment

    Returns
    -------
    float or array — flow of new matches per period
    """
    return m * (u ** alpha) * (v ** (1.0 - alpha))


def job_finding_rate(theta: float, m: float = M_EFF, alpha: float = ALPHA) -> float:
    """
    Job-finding rate p(θ) = M/u = m * θ^(1-α).

    This is the rate at which an unemployed worker transitions into
    employment. It is increasing in tightness θ = v/u: a tighter labour
    market benefits workers.

    Parameters
    ----------
    theta : float or array — labour-market tightness v/u
    """
    return m * (theta ** (1.0 - alpha))


def vacancy_filling_rate(theta: float, m: float = M_EFF, alpha: float = ALPHA) -> float:
    """
    Vacancy-filling rate q(θ) = M/v = m * θ^(-α).

    This is the rate at which a firm with an open vacancy fills that
    vacancy. It is decreasing in tightness θ: a tighter market makes
    recruiting harder for firms.

    Parameters
    ----------
    theta : float or array — labour-market tightness v/u
    """
    return m * (theta ** (-alpha))


# ---------------------------------------------------------------------------
# 3. Steady-State Unemployment
# ---------------------------------------------------------------------------

def steady_state_unemployment(theta: float,
                               delta: float = DELTA,
                               m: float = M_EFF,
                               alpha: float = ALPHA) -> float:
    """
    Steady-state unemployment rate u* = δ / (δ + p(θ)).

    In steady state the inflow to unemployment (job destruction at rate δ)
    equals the outflow (job finding at rate p(θ)).  Solving for u gives
    the Beveridge curve relationship between θ (or equivalently vacancies)
    and unemployment.

    Parameters
    ----------
    theta : float or array — labour-market tightness
    delta : float — job-destruction rate
    """
    p = job_finding_rate(theta, m, alpha)
    return delta / (delta + p)


def steady_state_vacancies(theta: float,
                            delta: float = DELTA,
                            m: float = M_EFF,
                            alpha: float = ALPHA) -> float:
    """
    Steady-state vacancy rate v* = θ · u*(θ).

    Parameters
    ----------
    theta : float or array — labour-market tightness
    """
    u = steady_state_unemployment(theta, delta, m, alpha)
    return theta * u


# ---------------------------------------------------------------------------
# 4. Job Creation Condition (Free-Entry / Zero-Profit)
# ---------------------------------------------------------------------------

def wage_nash(theta: float,
              beta: float = BETA,
              y: float = Y,
              b: float = B_BASE,
              c: float = C_VANCY,
              r: float = R,
              delta: float = DELTA,
              m: float = M_EFF,
              alpha: float = ALPHA) -> float:
    """
    Nash-bargained wage w(θ).

    In the MP model the wage satisfies the standard Nash-bargaining
    condition. With Cobb-Douglas matching and linear utilities:

        w(θ) = β · [y + c·θ] + (1 - β) · b

    where β is the worker's bargaining power, y is output, c·θ is the
    expected cost of recruiting (proportional to tightness), and b is
    the worker's outside option (unemployment benefits + home production).

    Parameters
    ----------
    theta : float or array — labour-market tightness
    beta  : float — worker's bargaining power
    y     : float — output per match
    b     : float — worker's outside option (replacement rate × y here)
    c     : float — vacancy-posting cost
    r     : float — discount rate
    delta : float — separation rate
    """
    return beta * (y + c * theta) + (1.0 - beta) * b


def jc_condition(theta: float,
                 c: float = C_VANCY,
                 r: float = R,
                 delta: float = DELTA,
                 m: float = M_EFF,
                 alpha: float = ALPHA,
                 **kwargs) -> float:
    """
    Job-Creation (JC) curve: rearrangement of the free-entry condition.

    A firm posts a vacancy as long as the expected profit from a filled job
    covers the vacancy cost. Free entry drives profits to zero:

        c / q(θ) = (y - w) / (r + δ)

    Re-expressed as output minus wage implied by the free-entry locus:

        y - w_JC(θ) = c·(r + δ) / q(θ)

    Equivalently, the JC curve gives the maximum wage the firm can afford
    to pay as a function of θ: w_JC = y - c·(r + δ)/q(θ).

    Parameters
    ----------
    theta : float or array — labour-market tightness
    """
    q = vacancy_filling_rate(theta, m, alpha)
    return Y - c * (r + delta) / q


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_beveridge_curve() -> str:
    """
    Plot the Beveridge Curve: steady-state (u, v) combinations as θ varies.

    The Beveridge Curve is the locus of (unemployment, vacancy) pairs
    consistent with steady-state labour-market flows.  It is downward
    sloping because higher tightness (more vacancies relative to
    unemployment) raises the job-finding rate and reduces steady-state
    unemployment.
    """
    theta_grid = np.linspace(0.05, 5.0, 500)
    u_star = steady_state_unemployment(theta_grid)
    v_star = steady_state_vacancies(theta_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(u_star, v_star, color="steelblue", lw=2.5, label="Beveridge Curve")

    # Mark the 45-degree line (u = v) for reference
    lim = max(u_star.max(), v_star.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4, label="45° line (u = v)")

    # Mark the baseline equilibrium (θ = 1)
    u0 = steady_state_unemployment(1.0)
    v0 = steady_state_vacancies(1.0)
    ax.scatter([u0], [v0], color="crimson", s=60, zorder=5,
               label=f"Equilibrium (θ=1): u={u0:.2f}, v={v0:.2f}")

    ax.set_xlabel("Unemployment Rate  u", fontsize=12)
    ax.set_ylabel("Vacancy Rate  v", fontsize=12)
    ax.set_title("Beveridge Curve\n(Mortensen-Pissarides Model)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, lim)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "beveridge_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_matching_rates() -> str:
    """
    Plot the job-finding rate p(θ) and vacancy-filling rate q(θ) vs. tightness.

    These two curves illustrate how market tightness affects the fortunes of
    workers (higher θ → higher p, easier to find a job) and firms (higher θ →
    lower q, harder to fill a vacancy).  The intersection at θ = 1 is not
    special economically — it depends on the unit normalisation.
    """
    theta_grid = np.linspace(0.05, 5.0, 500)
    p = job_finding_rate(theta_grid)
    q = vacancy_filling_rate(theta_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(theta_grid, p, color="steelblue", lw=2.5, label="Job-finding rate  p(θ)")
    ax.plot(theta_grid, q, color="darkorange", lw=2.5, label="Vacancy-filling rate  q(θ)")
    ax.axvline(1.0, color="grey", lw=1, linestyle=":", alpha=0.7, label="θ = 1")

    ax.set_xlabel("Market Tightness  θ = v/u", fontsize=12)
    ax.set_ylabel("Rate (per quarter)", fontsize=12)
    ax.set_title("Job-Finding and Vacancy-Filling Rates", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 2.0)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "matching_rates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equilibrium_jc_wage() -> str:
    """
    Plot the Job-Creation (JC) and Wage (WC) curves to find equilibrium θ.

    The JC curve (free-entry condition) is decreasing in θ because a tighter
    market makes recruiting costlier.  The WC curve (Nash bargaining) is
    increasing in θ because workers extract higher wages when the market is
    tight.  Their intersection determines the equilibrium tightness θ* and
    wage w*.
    """
    theta_grid = np.linspace(0.05, 5.0, 500)
    w_jc = jc_condition(theta_grid)      # Max wage firms will pay (JC locus)
    w_wc = wage_nash(theta_grid)          # Wage workers will accept (WC locus)

    # Find equilibrium by interpolation (where JC = WC)
    diff = w_jc - w_wc
    idx  = np.argmin(np.abs(diff))
    theta_eq = theta_grid[idx]
    w_eq     = 0.5 * (w_jc[idx] + w_wc[idx])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(theta_grid, w_jc, color="steelblue", lw=2.5, label="Job-Creation (JC) curve")
    ax.plot(theta_grid, w_wc, color="darkorange", lw=2.5, label="Wage (WC) curve")
    ax.scatter([theta_eq], [w_eq], color="crimson", s=80, zorder=5,
               label=f"Equilibrium: θ*={theta_eq:.2f}, w*={w_eq:.2f}")
    ax.axvline(theta_eq, color="grey", lw=1, linestyle="--", alpha=0.6)
    ax.axhline(w_eq,     color="grey", lw=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("Market Tightness  θ = v/u", fontsize=12)
    ax.set_ylabel("Wage  w", fontsize=12)
    ax.set_title("Equilibrium Determination\n(JC & WC Curves)", fontsize=13)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=10)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "equilibrium_jc_wc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ui_policy_experiment() -> str:
    """
    Policy experiment: effect of raising unemployment-insurance (UI) benefits.

    We vary the replacement rate b from 0.1 to 0.8 and compute the
    equilibrium tightness θ*(b) and steady-state unemployment u*(b).

    Economic intuition: higher b increases the worker's outside option,
    which pushes up the Nash bargaining wage.  This reduces the incentive
    for firms to post vacancies, lowering θ* and raising unemployment.
    """
    b_values = np.linspace(0.05, 0.90, 200)
    theta_grid = np.linspace(0.01, 10.0, 2000)

    theta_eq_list = []
    u_eq_list     = []

    for b in b_values:
        # For each b, compute JC and WC curves and find intersection
        w_jc = jc_condition(theta_grid)
        w_wc = wage_nash(theta_grid, b=b)
        diff = w_jc - w_wc
        # The intersection is where diff changes sign (JC crosses WC from above)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            theta_eq_list.append(np.nan)
            u_eq_list.append(np.nan)
        else:
            idx = sign_changes[0]
            # Linear interpolation between grid points
            t0, t1 = theta_grid[idx], theta_grid[idx + 1]
            d0, d1 = diff[idx], diff[idx + 1]
            theta_star = t0 - d0 * (t1 - t0) / (d1 - d0)
            theta_eq_list.append(theta_star)
            u_eq_list.append(steady_state_unemployment(theta_star))

    theta_eq_arr = np.array(theta_eq_list)
    u_eq_arr     = np.array(u_eq_list)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(b_values, theta_eq_arr, color="steelblue", lw=2.5)
    axes[0].set_xlabel("Replacement Rate  b", fontsize=12)
    axes[0].set_ylabel("Equilibrium Tightness  θ*", fontsize=12)
    axes[0].set_title("Effect of UI Benefits on\nLabour-Market Tightness", fontsize=12)

    axes[1].plot(b_values, u_eq_arr * 100, color="crimson", lw=2.5)
    axes[1].set_xlabel("Replacement Rate  b", fontsize=12)
    axes[1].set_ylabel("Equilibrium Unemployment Rate  u* (%)", fontsize=12)
    axes[1].set_title("Effect of UI Benefits on\nUnemployment Rate", fontsize=12)

    fig.suptitle("Policy Experiment: Unemployment Insurance", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "ui_policy_experiment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Problem Set 1 — Labor Market Models")
    print("=" * 50)

    paths = [
        plot_beveridge_curve(),
        plot_matching_rates(),
        plot_equilibrium_jc_wage(),
        plot_ui_policy_experiment(),
    ]

    for p in paths:
        print(f"  Saved: {p}")

    print("\nDone. All figures saved to ProblemSet1/results/")
