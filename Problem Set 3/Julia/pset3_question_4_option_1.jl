# Author: Jorge Luis Ochoa Rincon
# ECON 2080 - Macroeconomics
# Dynamic Programming on Julia
# Aiyagari Model Implementation

using Interpolations          # Linear interpolation package
using Optim                   # Root finding method package
using Plots                   # Plotting package

# ============================================================================
# PART 1: Set up parameters and grid
# ============================================================================

# Parameters
β = 0.96                        # Discount factor
α = 0.33                        # Production function parameter
δ = 0.05                        # Depreciation rate
ε = 10e-6                       # Tolerance level for convergence   

# Productivity
z_vals = [0.1, 1.0]             # Productivity states
nz = 2                          # Number of productivity states
P = [0.9 0.1; 0.1 0.9]          # Transition matrix for productivity states 

# Asset grid
na = 200                                            # Number of asset grid points
a_grid = collect(range(1e-10, 20.0, length=na))     # Asset grid from a small positive number to 20

# Value function and policy function
V = zeros(na, nz)
policy = zeros(Int, na, nz)

"""
    u(c)

Computes utility for consumption c.

# Arguments
- `c`: Consumption level

# Returns
- `u(c)`: Utility value
"""
function u(c)
    return log(c)
end

# ============================================================================
# PART 2: Implement the VFI algorithm
# ============================================================================

"""
    vfi(r, w; tol=1e-6, max_iter=2000)

Implements the Value Function Iteration (VFI) to solve the Bellman equation.

# Arguments
- `r`: Interest rate
- `w`: Wage rate
- `tol`: Tolerance level for convergence (default: 1e-6)
- `max_iter`: Maximum number of iterations (default: 2000)

# Returns
- `V`: Value function
- `policy`: Policy function (index of next period's asset choice)

# Mathematical formulation:
The Bellman equation is:
    V(a,z) = max_{a'} [u(w*z + (1+r)*a - a') + β * E[V(a', z'|z)]]

where:
- a is current assets
- a' is next period assets
- f(a) = w*z + (1+r)*a is the production function (income from assets)
- u(c) = log(c) is the utility function
- c = f(a) - a' is consumption

"""
function vfi(r, w; tol=1e-6, max_iter=2000)
    
    # Initial guess
    V = zeros(na, nz)
    for i in 1:na, iz in 1:nz
        c0 = (1 + r) * a_grid[i] + w * z_vals[iz]
        V[i, iz] = c0 > 0 ? u(c0) / (1 - β) : -1e10
    end

    policy = zeros(Int, na, nz)

    for iter in 1:max_iter
        V_new = zeros(na, nz)

        for iz in 1:nz, ia in 1:na
            a = a_grid[ia]
            best_val = -1e20
            best_ia = 1

            for ia_next in 1:na
                c = w * z_vals[iz] + (1 + r) * a - a_grid[ia_next]
                if c <= 0
                    continue
                end

                # Expected continuation value
                EV = sum(P[iz, iz2] * V[ia_next, iz2] for iz2 in 1:nz)
                val = u(c) + β * EV

                if val > best_val
                    best_val = val
                    best_ia = ia_next
                elseif val < best_val
                    break  # Exploit concavity
                end
            end

            V_new[ia, iz] = best_val
            policy[ia, iz] = best_ia
        end

        if maximum(abs.(V_new - V)) < tol
            println("VFI converged in $iter iterations")
            return V_new, policy
        end

        V = V_new
    end

    println("VFI did not converge")
    return V, policy
end

# ============================================================================
# PART 3: Evaluate the VFI function with set parameters and plot the policy function
# ============================================================================


# Step 3: Solve and plot policy function
r = 0.03
w = 0.956

V_sol, pol_sol = vfi(r, w)

# Convert policy indices to asset values
a_next_low  = [a_grid[pol_sol[ia, 1]] for ia in 1:na]  # z = 0.1
a_next_high = [a_grid[pol_sol[ia, 2]] for ia in 1:na]  # z = 1.0

plot(a_grid, a_next_low,  label="z = 0.1", linewidth=2, color = "#700707")
plot!(a_grid, a_next_high, label="z = 1.0", linewidth=2, color = "#0b2066")
plot!(a_grid, a_grid, label="45° line", linestyle=:dash, color=:black)

xlabel!("Current assets (a)")
ylabel!("Next-period assets (a')")
title!("Policy Function - Asset Accumulation")
savefig("~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/Problem Sets/Problem Set 3/Julia/Outputs/assets_accumulation_part3.pdf")

# ============================================================================
# PART 4: Compute the stationary distribution of households over assets and productivity states
# ============================================================================

function stationary_dist(pol; tol=1e-10, max_iter=10000)
    
    # Initialize: all mass at lowest asset level
    μ = zeros(na, nz)
    for iz in 1:nz
        μ[1, iz] = 1.0 / nz
    end

    for iter in 1:max_iter
        μ_new = zeros(na, nz)

        for iz in 1:nz, ia in 1:na
            ia_next = pol[ia, iz]
            for iz2 in 1:nz
                μ_new[ia_next, iz2] += μ[ia, iz] * P[iz, iz2]
            end
        end

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

# Aggregate capital
K = sum(a_grid[ia] * μ_sol[ia, iz] for ia in 1:na, iz in 1:nz)
println("Aggregate capital K = ", round(K, digits=4))

# ============================================================================
# PART 5: Implement the general equilibrium
# ============================================================================

# Equilibrium wage from firm's FOC given r
function eq_wage(r)
    return (1 - α) * (α / (r + δ))^(α / (1 - α))
end

# Capital supply: solve HH problem and compute aggregate K
function capital_supply(r)
    w = eq_wage(r)
    _, pol = vfi(r, w)
    μ = stationary_dist(pol)
    K = sum(a_grid[ia] * μ[ia, iz] for ia in 1:na, iz in 1:nz)
    return K
end

# Capital demand: firm's FOC
function capital_demand(K)
    return α * K^(α - 1) - δ
end

# Grid of interest rates
r_grid = collect(range(0.005, 0.04, length=20))

println("Computing capital supply across interest rates...")
K_supply = [capital_supply(r) for r in r_grid]
K_demand = [capital_demand(K) for K in K_supply]

# Plot
plot(K_supply, r_grid,  label="Capital Supply",  linewidth=2, color = "#700707")
plot!(K_supply, K_demand, label="Capital Demand", linewidth=2, color = "#0b2066")

xlabel!("Capital (K)")
ylabel!("Interest Rate (r)")
title!("Capital Market Equilibrium")
savefig("~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/Problem Sets/Problem Set 3/Julia/Outputs/equilibrium_part5.pdf")



# ============================================================================
# PART 6: Generate and save two plots
# ============================================================================

plot(a_grid, a_next_low,  label="z = 0.1", linewidth=2, color = "#700707")
plot!(a_grid, a_next_high, label="z = 1.0", linewidth=2, color = "#0b2066")
plot!(a_grid, a_grid, label="45° line", linestyle=:dash, color=:black)

xlabel!("Current assets (a)")
ylabel!("Next-period assets (a')")
title!("Policy Function - Asset Accumulation")
savefig("~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/Problem Sets/Problem Set 3/Julia/Outputs/policy_function.pdf")
