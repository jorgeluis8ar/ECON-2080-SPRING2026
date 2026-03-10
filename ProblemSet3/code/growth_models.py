"""
ECON 2080 — Problem Set 3: Growth Models
==========================================
This script implements and solves three canonical growth models and produces
key diagnostic figures:

    1. Solow-Swan Model — steady state, golden rule capital, and
       transitional dynamics (numerical simulation).
    2. Ramsey-Cass-Koopmans (RCK) Model — phase diagram in the (k, c)
       plane showing the saddle path to the balanced growth path (BGP).
    3. Transitional Dynamics under Different Savings Rates — comparative
       analysis of convergence speeds under the Solow model.
    4. AK Endogenous Growth — illustration of sustained growth with
       constant returns to capital.

All models are specified in per-capita, detrended (intensive-form) terms.
The capital stock k = K/AL denotes effective capital per unit of effective
labour, and c = C/AL denotes effective consumption per unit of effective
labour.

References
----------
Solow, R. M. (1956). A contribution to the theory of economic growth.
    *Quarterly Journal of Economics*, 70(1), 65–94.
Cass, D. (1965). Optimum growth in an aggregative model of capital
    accumulation. *Review of Economic Studies*, 32(3), 233–240.
Koopmans, T. C. (1965). On the concept of optimal economic growth.
    Cowles Foundation Discussion Paper 163.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# 0. Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Shared Parameters
# ---------------------------------------------------------------------------
# Technology / demographics
ALPHA  = 0.33    # Capital share in Cobb-Douglas production: f(k) = k^α
DELTA  = 0.08    # Capital depreciation rate (annual)
N      = 0.01    # Population growth rate (annual)
G      = 0.02    # Exponent of labour-augmenting technological progress (annual)
# Preferences (for Ramsey model)
RHO    = 0.04    # Rate of pure time preference (utility discount rate)
THETA  = 2.0     # Inverse of intertemporal elasticity of substitution (IES)
# Solow model
S_SOLOW = 0.25   # Constant savings rate in the Solow-Swan model


# ---------------------------------------------------------------------------
# 2. Production Technology
# ---------------------------------------------------------------------------

def production(k: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Intensive-form Cobb-Douglas production function: f(k) = k^α.

    'Intensive form' means output per effective worker as a function of
    capital per effective worker.  The function exhibits:
    - Positive and diminishing marginal product: f'(k) = α k^(α-1) > 0, f''<0
    - Inada conditions: f'(k)→∞ as k→0, f'(k)→0 as k→∞

    Parameters
    ----------
    k     : array-like — capital per effective worker
    alpha : float — capital's output elasticity

    Returns
    -------
    np.ndarray — output per effective worker
    """
    return np.asarray(k, float) ** alpha


def mpk(k: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Marginal product of capital: f'(k) = α k^(α-1).

    The rental rate of capital in competitive equilibrium equals the MPK.
    Higher capital stock implies lower MPK (diminishing returns).

    Parameters
    ----------
    k     : array-like — capital per effective worker
    alpha : float — capital elasticity
    """
    return alpha * np.asarray(k, float) ** (alpha - 1.0)


# ---------------------------------------------------------------------------
# 3. Solow-Swan Model
# ---------------------------------------------------------------------------

def solow_steady_state(s: float = S_SOLOW,
                       delta: float = DELTA,
                       n: float = N,
                       g: float = G,
                       alpha: float = ALPHA) -> float:
    """
    Compute the Solow steady-state capital per effective worker.

    In steady state dk/dt = 0, which requires:
        s f(k*) = (δ + n + g) k*
    ⟹  s (k*)^α = (δ + n + g) k*
    ⟹  k* = [s / (δ + n + g)]^(1/(1-α))

    Parameters
    ----------
    s     : float — savings rate
    delta : float — depreciation rate
    n     : float — population growth rate
    g     : float — technology growth rate
    alpha : float — capital share
    """
    return (s / (delta + n + g)) ** (1.0 / (1.0 - alpha))


def solow_golden_rule(delta: float = DELTA,
                      n: float = N,
                      g: float = G,
                      alpha: float = ALPHA) -> float:
    """
    Capital stock at the Golden Rule: maximises steady-state consumption.

    At the Golden Rule, the net marginal product of capital equals the
    effective depreciation:
        f'(k_gr) = δ + n + g
    ⟹  α k_gr^(α-1) = δ + n + g
    ⟹  k_gr = [α / (δ + n + g)]^(1/(1-α))

    Note: the Golden Rule savings rate is s_gr = (δ+n+g) k_gr / f(k_gr) = α.
    Under Cobb-Douglas production the Golden Rule savings rate equals the
    capital share α (a well-known result).

    Parameters
    ----------
    delta : float — depreciation rate
    n     : float — population growth rate
    g     : float — technology growth rate
    alpha : float — capital share
    """
    return (alpha / (delta + n + g)) ** (1.0 / (1.0 - alpha))


def solow_rhs(k: np.ndarray,
              s: float = S_SOLOW,
              delta: float = DELTA,
              n: float = N,
              g: float = G,
              alpha: float = ALPHA) -> np.ndarray:
    """
    Right-hand side of the Solow capital-accumulation equation.

    dk/dt = s f(k) − (δ + n + g) k

    A positive value means k is rising; a negative value means k is falling.
    The steady state is where dk/dt = 0.

    Parameters
    ----------
    k : array-like — capital per effective worker
    s : float — savings rate
    """
    k = np.asarray(k, float)
    return s * production(k, alpha) - (delta + n + g) * k


def simulate_solow(k0: float,
                   T: float = 100.0,
                   s: float = S_SOLOW,
                   delta: float = DELTA,
                   n: float = N,
                   g: float = G,
                   alpha: float = ALPHA) -> tuple:
    """
    Simulate the Solow model from initial capital k0 over T periods.

    We integrate the ODE dk/dt = s f(k) − (δ+n+g)k numerically using
    an adaptive Runge-Kutta solver (RK45).

    Parameters
    ----------
    k0    : float — initial capital per effective worker
    T     : float — time horizon (years)
    s     : float — savings rate
    delta, n, g, alpha : model parameters

    Returns
    -------
    t_arr : np.ndarray — time points
    k_arr : np.ndarray — capital path k(t)
    y_arr : np.ndarray — output path y(t) = f(k(t))
    """
    def ode(t, k):
        return [solow_rhs(k[0], s, delta, n, g, alpha)]

    sol = solve_ivp(ode, [0, T], [k0], dense_output=True,
                    rtol=1e-8, atol=1e-10)
    t_arr = np.linspace(0, T, 500)
    k_arr = sol.sol(t_arr)[0]
    y_arr = production(k_arr, alpha)
    return t_arr, k_arr, y_arr


# ---------------------------------------------------------------------------
# 4. Ramsey-Cass-Koopmans Model
# ---------------------------------------------------------------------------

def ramsey_kss(rho: float = RHO,
               theta: float = THETA,
               delta: float = DELTA,
               n: float = N,
               g: float = G,
               alpha: float = ALPHA) -> float:
    """
    Steady-state (BGP) capital in the Ramsey-Cass-Koopmans model.

    Along the Balanced Growth Path (BGP) the modified golden rule holds:
        f'(k**) − δ = ρ + θ g
    ⟹  α (k**)^(α-1) − δ = ρ + θ g
    ⟹  k** = [α / (ρ + θ g + δ)]^(1/(1-α))

    Unlike the Solow Golden Rule, this steady state depends on preferences
    (ρ and θ): more impatient agents (higher ρ) accumulate less capital.

    Parameters
    ----------
    rho   : float — rate of pure time preference
    theta : float — inverse IES
    delta : float — depreciation rate
    n     : float — population growth rate
    g     : float — technology growth rate
    alpha : float — capital share
    """
    return (alpha / (rho + theta * g + delta)) ** (1.0 / (1.0 - alpha))


def ramsey_css(k_ss: float,
               delta: float = DELTA,
               n: float = N,
               g: float = G,
               alpha: float = ALPHA) -> float:
    """
    Steady-state consumption in the Ramsey model.

    At the steady state, consumption equals output minus investment:
        c** = f(k**) − (δ + n + g) k**

    Parameters
    ----------
    k_ss : float — steady-state capital (from ramsey_kss)
    """
    return production(k_ss, alpha) - (delta + n + g) * k_ss


def ramsey_dynamics(t, state,
                    rho=RHO, theta=THETA,
                    delta=DELTA, n=N, g=G, alpha=ALPHA):
    """
    System of ODEs for the Ramsey model in intensive form.

    The two-equation system (after detrending):
        dk/dt = f(k) − (δ + n + g) k − c
        dc/dt = c · [f'(k) − δ − ρ − θ g] / θ

    Derivation:
    - The capital accumulation equation gives dk/dt.
    - The Euler equation for consumption (households maximise lifetime utility
      subject to the budget constraint) implies the growth rate of consumption:
        ċ/c = [r − ρ − θ g] / θ   where r = f'(k) − δ.

    Parameters
    ----------
    t     : float — time (not used explicitly; system is autonomous)
    state : list  — [k, c] current capital and consumption
    """
    k, c = state
    k = max(k, 1e-8)
    c = max(c, 1e-8)
    dk = production(k, alpha) - (delta + n + g) * k - c
    dc = c * (mpk(k, alpha) - delta - rho - theta * g) / theta
    return [dk, dc]


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_solow_diagram() -> str:
    """
    Plot the Solow diagram: sf(k) vs. (δ+n+g)k, with steady state and
    Golden Rule annotated.

    The Solow diagram is the most intuitive tool for understanding the model:
    - Above the intersection (k < k*): saving exceeds depreciation → k rises.
    - Below the intersection (k > k*): saving is less than depreciation → k falls.
    - At k*: saving exactly replaces depreciation → k is constant.
    """
    k_grid = np.linspace(0.01, 6.0, 500)
    invest = S_SOLOW * production(k_grid)              # Savings/investment curve
    req    = (DELTA + N + G) * k_grid                  # Required investment line
    output = production(k_grid)                        # Output curve

    k_star = solow_steady_state()
    k_gr   = solow_golden_rule()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_grid, output,  color="seagreen",   lw=2.0, label=r"Output  $f(k) = k^\alpha$")
    ax.plot(k_grid, invest,  color="steelblue",  lw=2.5, label=r"Savings  $s \cdot f(k)$")
    ax.plot(k_grid, req,     color="crimson",    lw=2.5, linestyle="--",
            label=r"Required investment  $(\delta+n+g)k$")

    # Mark steady state
    k_star_y = S_SOLOW * production(k_star)
    ax.scatter([k_star], [k_star_y], color="steelblue", s=80, zorder=5)
    ax.axvline(k_star, color="steelblue", lw=1, linestyle=":", alpha=0.7)
    ax.text(k_star + 0.05, k_star_y * 0.6,
            f"$k^*={k_star:.2f}$", fontsize=10, color="steelblue")

    # Mark Golden Rule
    k_gr_y = S_SOLOW * production(k_gr)
    ax.scatter([k_gr], [(DELTA + N + G) * k_gr], color="gold", s=80, zorder=5,
               edgecolors="darkorange", linewidths=1.5)
    ax.text(k_gr + 0.05, (DELTA + N + G) * k_gr * 1.05,
            f"$k_{{gr}}={k_gr:.2f}$", fontsize=10, color="darkorange")

    ax.set_xlabel("Capital per Effective Worker  k", fontsize=12)
    ax.set_ylabel("Per-Effective-Worker Quantities", fontsize=12)
    ax.set_title("Solow-Swan Diagram", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 6); ax.set_ylim(0, 2.5)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "solow_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_solow_dynamics() -> str:
    """
    Plot transitional dynamics in the Solow model for different initial
    capital levels and different savings rates.

    Left panel:  Capital paths k(t) starting from above and below k*.
    Right panel: Output paths y(t) = f(k(t)) for different savings rates.
    """
    k_star = solow_steady_state()
    T      = 120.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: convergence from different initial conditions ---
    k0_vals  = [0.2 * k_star, 0.5 * k_star, 2.0 * k_star, 3.0 * k_star]
    colors   = ["#1a6faf", "#3b9fd4", "#f0882d", "#d94f0a"]
    for k0, col in zip(k0_vals, colors):
        t_arr, k_arr, _ = simulate_solow(k0)
        axes[0].plot(t_arr, k_arr, color=col, lw=2.0,
                     label=f"$k_0={k0:.1f}$")
    axes[0].axhline(k_star, color="black", lw=1.5, linestyle="--",
                    label=f"Steady state $k^*={k_star:.2f}$")
    axes[0].set_xlabel("Time (years)", fontsize=12)
    axes[0].set_ylabel("Capital per Effective Worker  k(t)", fontsize=12)
    axes[0].set_title("Convergence from Different Initial Conditions", fontsize=12)
    axes[0].legend(fontsize=9)

    # --- Right: effect of savings rate on output paths ---
    s_vals    = [0.10, 0.20, 0.30, 0.40]
    colors_s  = ["#1a6faf", "#3b9fd4", "#f0882d", "#d94f0a"]
    for s, col in zip(s_vals, colors_s):
        k_star_s = solow_steady_state(s=s)
        t_arr, _, y_arr = simulate_solow(k0=0.1, T=T, s=s)
        axes[1].plot(t_arr, y_arr, color=col, lw=2.0, label=f"s = {s:.2f}")
    axes[1].set_xlabel("Time (years)", fontsize=12)
    axes[1].set_ylabel("Output per Effective Worker  y(t)", fontsize=12)
    axes[1].set_title("Effect of Savings Rate on Output Dynamics", fontsize=12)
    axes[1].legend(fontsize=10)

    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "solow_dynamics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ramsey_phase_diagram() -> str:
    """
    Plot the phase diagram of the Ramsey-Cass-Koopmans model in the (k, c) plane.

    Two loci define the dynamics:
    - k-dot = 0 locus:  c = f(k) − (δ+n+g)k  (inverted-U in (k,c) space)
    - c-dot = 0 locus:  k = k** (vertical line where f'(k)=δ+ρ+θg)

    The intersection of the two loci is the steady state.  The stable manifold
    (saddle path) is the unique path along which the economy converges to the
    steady state satisfying the transversality condition.
    """
    k_ss = ramsey_kss()
    c_ss = ramsey_css(k_ss)

    # k-dot = 0 locus: c = f(k) - (δ+n+g)k
    k_kdot_max = brentq(
        lambda k: production(k) - (DELTA + N + G) * k, 0.01, 500.0
    )
    k_grid  = np.linspace(0.01, k_kdot_max * 0.99, 400)
    c_kdot  = production(k_grid) - (DELTA + N + G) * k_grid

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(k_grid, c_kdot, color="steelblue", lw=2.5, label=r"$\dot{k}=0$ locus")
    ax.axvline(k_ss, color="crimson", lw=2.5, linestyle="--",
               label=r"$\dot{c}=0$ locus (k = k**)")
    ax.scatter([k_ss], [c_ss], color="black", s=80, zorder=5,
               label=f"Steady state: k**={k_ss:.2f}, c**={c_ss:.2f}")

    # Simulate several trajectories to sketch the saddle path and diverging paths
    eps_k = 0.10 * k_ss   # small perturbations in k
    eps_c = [-0.12, -0.06, 0.0, 0.06, 0.12]   # perturbations in c

    # We numerically find the saddle path by starting very close to the SS
    for de in eps_c:
        for sign in [+1, -1]:
            k0 = k_ss + sign * eps_k
            c0 = c_ss + de
            try:
                sol = solve_ivp(ramsey_dynamics, [0, 60],
                                [k0, c0], dense_output=True,
                                rtol=1e-7, atol=1e-9,
                                events=lambda t, s: s[1] - 0.01)  # stop if c→0
                t_ = np.linspace(0, sol.t[-1], 200)
                path_k, path_c = sol.sol(t_)
                mask = (path_k > 0) & (path_c > 0) & (path_k < k_kdot_max * 1.2)
                if mask.sum() > 5:
                    ax.plot(path_k[mask], path_c[mask],
                            color="grey", lw=0.9, alpha=0.5, zorder=1)
            except Exception:
                pass

    ax.set_xlabel("Capital per Effective Worker  k", fontsize=12)
    ax.set_ylabel("Consumption per Effective Worker  c", fontsize=12)
    ax.set_title("Ramsey-Cass-Koopmans Phase Diagram", fontsize=13)
    ax.set_xlim(0, k_kdot_max * 0.85)
    ax.set_ylim(0, max(c_kdot) * 1.1)
    ax.legend(fontsize=10, loc="upper right")

    # Add arrows to show direction of motion in each quadrant
    for kv, cv in [(0.3*k_ss, 0.3*c_ss), (1.6*k_ss, 0.3*c_ss),
                   (0.3*k_ss, 1.3*c_ss), (1.6*k_ss, 1.3*c_ss)]:
        dk, dc = ramsey_dynamics(0, [kv, cv])
        ax.annotate("", xytext=(kv, cv),
                    xy=(kv + 0.08 * np.sign(dk) * k_ss,
                        cv + 0.08 * np.sign(dc) * c_ss),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))

    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "ramsey_phase_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ak_growth() -> str:
    """
    Plot AK endogenous growth model: capital and output grow at a constant rate.

    In the AK model, production is linear in capital: Y = A K, which
    implies constant marginal product of capital A.  If A > ρ + δ, the
    economy grows indefinitely at rate g* = (A − δ − ρ) / θ — there is no
    diminishing returns to halt growth.

    We contrast this with the Solow model (concave production) where output
    eventually stagnates.
    """
    A_AK   = 0.20    # Marginal product of capital in the AK model
    k0     = 1.0     # Initial capital
    T      = 80.0
    t_arr  = np.linspace(0, T, 500)

    # AK growth rate of capital: dk/dt = (A - δ) k  → k(t) = k0 exp((A-δ)t)
    g_ak   = A_AK - DELTA                  # Net growth rate of capital/output
    k_ak   = k0 * np.exp(g_ak * t_arr)    # Capital path (AK model)
    y_ak   = A_AK * k_ak                  # Output path

    # Solow comparison: simulate from same k0 with s calibrated to A_AK baseline
    s_solow = A_AK * S_SOLOW / production(1.0)   # Match initial investment share
    t_solow, k_solow, y_solow = simulate_solow(k0=k0, T=T, s=S_SOLOW)

    # Normalise both to 1 at t = 0 for cleaner comparison
    k_ak_n   = k_ak   / k_ak[0]
    k_sol_n  = k_solow / k_solow[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(t_arr,    k_ak_n,  color="darkorange", lw=2.5,
                 label=f"AK model (g={g_ak:.0%})")
    axes[0].plot(t_solow,  k_sol_n, color="steelblue",  lw=2.5,
                 label=f"Solow model (s={S_SOLOW})")
    axes[0].set_xlabel("Time (years)", fontsize=12)
    axes[0].set_ylabel("Normalised Capital  k(t)/k(0)", fontsize=12)
    axes[0].set_title("Capital Accumulation:\nAK vs. Solow Model", fontsize=12)
    axes[0].legend(fontsize=10)

    # Growth rate of output in the AK model is constant
    gc_ak  = np.full_like(t_arr, g_ak * 100)

    # Growth rate in the Solow model is declining toward zero
    y_solow_arr = production(k_solow)
    gc_solow    = np.gradient(np.log(y_solow_arr), t_solow) * 100

    axes[1].plot(t_arr,   gc_ak,    color="darkorange", lw=2.5, label="AK model")
    axes[1].plot(t_solow, gc_solow, color="steelblue",  lw=2.5, label="Solow model")
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Time (years)", fontsize=12)
    axes[1].set_ylabel("Output Growth Rate (%)", fontsize=12)
    axes[1].set_title("Output Growth Rate:\nAK vs. Solow Model", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(-1, g_ak * 100 * 1.5)

    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "ak_vs_solow_growth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Problem Set 3 — Growth Models")
    print("=" * 50)

    paths = [
        plot_solow_diagram(),
        plot_solow_dynamics(),
        plot_ramsey_phase_diagram(),
        plot_ak_growth(),
    ]

    for p in paths:
        print(f"  Saved: {p}")

    print("\nDone. All figures saved to ProblemSet3/results/")
