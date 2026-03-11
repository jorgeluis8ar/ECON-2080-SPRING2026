# =============================================================================
# AIYAGARI MODEL — Full Solution (Option 1)
# =============================================================================
# This script solves the Aiyagari (1994) heterogeneous-agent model using:
#   - Value Function Iteration (VFI) to solve the household problem
#   - Stationary distribution iteration to get the wealth distribution
#   - General equilibrium loop to find the equilibrium interest rate
# =============================================================================

using Plots

# =============================================================================
# STEP 1: PARAMETERS AND GRIDS
# =============================================================================

# Preferences
β = 0.96        # Discount factor: households value future utility at 96% of present
α = 0.33        # Capital share in production (Cobb-Douglas)
δ = 0.05        # Depreciation rate of capital

# Idiosyncratic productivity states
# z ∈ {0.1, 1.0}: low and high productivity workers
z_vals = [0.1, 1.0]
nz     = 2

# Markov transition matrix for productivity
# P[i,j] = probability of going from state i to state j
# Diagonal 0.9 means productivity is persistent (90% chance of staying in same state)
P = [0.9 0.1;
     0.1 0.9]

# Asset grid: na evenly spaced points from near-zero to 20
# We use 1e-10 instead of 0 to avoid log(0) in the utility function
na     = 200
a_grid = collect(range(1e-10, 20.0, length=na))

# Log utility: u(c) = log(c)
# This implies a coefficient of relative risk aversion (CRRA) = 1
function u(c)
    return log(c)
end

# =============================================================================
# STEP 2: VALUE FUNCTION ITERATION (VFI)
# =============================================================================
# Solves the household Bellman equation:
#   V(a,z) = max_{a'} { u(wz + (1+r)a - a') + β * E[V(a',z')|z] }
# subject to: a' ≥ 0 and c = wz + (1+r)a - a' > 0
#
# Inputs:  r (interest rate), w (wage)
# Outputs: converged value function V [na × nz], policy indices pol [na × nz]

function vfi(r, w; tol=1e-6, max_iter=2000)

    # Initial guess: value of consuming all wealth forever (no saving)
    # V0(a,z) = u(c0) / (1 - β), where c0 = (1+r)*a + w*z
    V = zeros(na, nz)
    for i in 1:na, iz in 1:nz
        c0 = (1 + r) * a_grid[i] + w * z_vals[iz]
        V[i, iz] = c0 > 0 ? u(c0) / (1 - β) : -1e10
    end

    # pol[ia, iz] stores the INDEX on a_grid of optimal next-period assets
    pol = zeros(Int, na, nz)

    for iter in 1:max_iter
        V_new = zeros(na, nz)

        # Outer loops: iterate over all (asset, productivity) states
        for iz in 1:nz, ia in 1:na
            a        = a_grid[ia]
            best_val = -1e20
            best_ia  = 1

            # Inner loop: search over all possible next-period asset choices a'
            for ia_next in 1:na
                c = w * z_vals[iz] + (1 + r) * a - a_grid[ia_next]

                # Skip infeasible choices (negative consumption)
                if c <= 0
                    continue
                end

                # Expected continuation value: sum over next-period productivity states
                # EV = Σ_{z'} P(z, z') * V(a', z')
                EV  = sum(P[iz, iz2] * V[ia_next, iz2] for iz2 in 1:nz)
                val = u(c) + β * EV

                if val > best_val
                    best_val = val
                    best_ia  = ia_next
                elseif val < best_val
                    # Value function is concave in a', so once it starts falling
                    # we can stop — no higher value will be found further along the grid
                    break
                end
            end

            V_new[ia, iz] = best_val
            pol[ia, iz]   = best_ia
        end

        # Check convergence: stop when the maximum change across all states is tiny
        if maximum(abs.(V_new - V)) < tol
            println("VFI converged in $iter iterations")
            return V_new, pol
        end

        V = V_new
    end

    println("VFI did not converge")
    return V, pol
end

# =============================================================================
# STEP 3: POLICY FUNCTION PLOT (at r = 0.03, w = 0.956)
# =============================================================================
# The policy function a*(a,z) tells us: given current assets a and productivity z,
# how much does the household choose to save for next period?

r = 0.03
w = 0.956

V_sol, pol_sol = vfi(r, w)

# Convert policy indices to actual asset values for plotting
a_next_low  = [a_grid[pol_sol[ia, 1]] for ia in 1:na]   # z = 0.1 (low productivity)
a_next_high = [a_grid[pol_sol[ia, 2]] for ia in 1:na]   # z = 1.0 (high productivity)

plot(a_grid, a_next_low,  label="z = 0.1", linewidth=2)
plot!(a_grid, a_next_high, label="z = 1.0", linewidth=2)
plot!(a_grid, a_grid,      label="45° line", linestyle=:dash, color=:black)

xlabel!("Current assets (a)")
ylabel!("Next-period assets (a')")
title!("Policy Function — Asset Accumulation")
savefig("policy_function.png")

# =============================================================================
# STEP 4: STATIONARY DISTRIBUTION
# =============================================================================
# Given the policy function, we iterate the distribution forward until it
# stops changing. Each period, all households in state (a,z) move their mass
# to state (a*(a,z), z') with probability P(z,z').
#
# Formally: μ'(a', z') = Σ_{(a,z): a*(a,z)=a'} μ(a,z) * P(z,z')
#
# The aggregate capital stock is K = Σ_{a,z} a * μ(a,z)

function stationary_dist(pol; tol=1e-10, max_iter=10000)

    # Initialize: put all mass at the lowest asset level, split equally across z states
    μ = zeros(na, nz)
    for iz in 1:nz
        μ[1, iz] = 1.0 / nz
    end

    for iter in 1:max_iter
        μ_new = zeros(na, nz)

        # For every current state (ia, iz), distribute mass to next-period states
        for iz in 1:nz, ia in 1:na
            ia_next = pol[ia, iz]   # optimal next-period asset index
            for iz2 in 1:nz
                # Add mass to (ia_next, iz2) weighted by transition probability
                μ_new[ia_next, iz2] += μ[ia, iz] * P[iz, iz2]
            end
        end

        # Check convergence: stop when distribution barely moves
        if maximum(abs.(μ_new - μ)) < tol
            println("Distribution converged in $iter iterations")
            return μ_new
        end

        μ = μ_new
    end

    println("Distribution did not converge")
    return μ
end

μ_sol = stationary_dist(pol_sol)

# Aggregate capital: wealth-weighted sum across all (a,z) states
K = sum(a_grid[ia] * μ_sol[ia, iz] for ia in 1:na, iz in 1:nz)
println("Aggregate capital K = ", round(K, digits=4))

# =============================================================================
# STEP 5: GENERAL EQUILIBRIUM
# =============================================================================
# In equilibrium, the interest rate r* equates capital supply (from households)
# with capital demand (from firms).
#
# Firm FOCs (with N=1):
#   r = α * K^(α-1) - δ       (capital demand)
#   w = (1-α) * (α/(r+δ))^(α/(1-α))   (wage consistent with r)
#
# We compute supply and demand over a grid of r values and find the intersection.

# Equilibrium wage implied by a given interest rate (from firm's FOC)
function eq_wage(r)
    return (1 - α) * (α / (r + δ))^(α / (1 - α))
end

# Capital supply: solve HH problem → get distribution → compute aggregate K
function capital_supply(r)
    w   = eq_wage(r)
    _, pol = vfi(r, w)
    μ   = stationary_dist(pol)
    K   = sum(a_grid[ia] * μ[ia, iz] for ia in 1:na, iz in 1:nz)
    return K
end

# Capital demand: firm's FOC rearranged to give r as a function of K
function capital_demand(K)
    return α * K^(α - 1) - δ
end

# Grid of 20 interest rates between 0.005 and 0.04
r_grid = collect(range(0.005, 0.04, length=20))

println("Computing capital supply across interest rates...")
K_supply  = [capital_supply(r) for r in r_grid]    # household saving at each r
K_demand  = [capital_demand(K) for K in K_supply]  # firm's implied r at each K

# =============================================================================
# STEP 6: SAVE PLOTS
# =============================================================================

# Plot 2: Capital market equilibrium
# Equilibrium is where supply and demand curves intersect
plot(K_supply, r_grid,   label="Capital Supply",  linewidth=2)
plot!(K_supply, K_demand, label="Capital Demand", linewidth=2)

xlabel!("Capital (K)")
ylabel!("Interest Rate (r)")
title!("Capital Market Equilibrium")
savefig("capital_equilibrium.png")

println("Done. Plots saved: policy_function.png, capital_equilibrium.png")