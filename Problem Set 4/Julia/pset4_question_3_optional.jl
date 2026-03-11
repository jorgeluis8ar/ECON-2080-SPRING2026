# Author: Jorge Luis Ochoa Rincon
# ECON 2080 - Macroeconomics
# Dynamic Programming on Julia
# McCall Search Model with Beta-Binomial Wage Distribution

# ============================================================================
# PART 1: Set up parameters and grid
# ============================================================================

using Distributions, Plots, LinearAlgebra

# Parameters
β    = 0.99
b    = 25.0
N    = 50
w_min = 10.0
w_max = 60.0

# Wage grid
w_grid = collect(range(w_min, w_max, length=N))

# Beta-Binomial wage distribution
dist    = BetaBinomial(N-1, 200, 100)
p_probs = pdf.(dist, support(dist))
p_probs = p_probs ./ sum(p_probs)

# Plot the PMF
bar(w_grid, p_probs,
    label     = "Probability",
    xlabel    = "Wage",
    ylabel    = "Probability",
    title     = "Wage Offer Distribution (Beta-Binomial)",
    linewidth = 0)

savefig("~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/ECON-2080-SPRING2026/Problem Set 4/Julia/Outputs/wage_distribution.pdf")


# ============================================================================
# PART 2: Defining the VFI function
# ============================================================================


function solve_mccall_vfi(w_grid, p_probs, b, β; tol=1e-6, max_iter=2000)
    
    # Initial guess: value of accepting every wage offer
    v = w_grid ./ (1 - β)

    for iter in 1:max_iter
        
        # Expected value of next period's offer: dot product of v and probabilities
        Ev = dot(v, p_probs)
        
        # Value of rejecting current offer (constant across all wages)
        v_reject = b + β * Ev
        
        # Update: for each wage, take the max of accepting or rejecting
        v_new = max.(w_grid ./ (1 - β), v_reject)

        if norm(v_new - v, Inf) < tol
            println("VFI converged in $iter iterations")
            return v_new
        end

        v = v_new
    end

    println("VFI did not converge")
    return v
end


# ============================================================================
# PART 3: Reservation wage
# ============================================================================

v_star = solve_mccall_vfi(w_grid, p_probs, b, β)

# RHS of Bellman equation: value of rejecting (constant across w)
Ev_star   = dot(v_star, p_probs)
v_reject  = b + β * Ev_star

# Reservation wage: the wage that makes the worker indifferent
# i.e., find w* such that w*/(1-β) = v_reject
w_star = v_reject * (1 - β)

println("Reservation wage w* = ", round(w_star, digits=4))


# ============================================================================
# PART 4: Visualization
# ============================================================================

v_accept = w_grid ./ (1 - β)
v_reject_line = fill(v_reject, N)

plot(w_grid, v_star,        label="Value function v*(w)",  linewidth=2)
plot!(w_grid, v_accept,     label="Value of accepting: w/(1-β)", linewidth=2)
plot!(w_grid, v_reject_line, label="Value of rejecting: b + βE[v*(w')]", linewidth=2, linestyle=:dash)
vline!([w_star],            label="Reservation wage w* = $(round(w_star, digits=2))", linestyle=:dot, color=:black)

xlabel!("Wage (w)")
ylabel!("Value")
title!("McCall Search Model — Value Functions")
savefig("~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/ECON-2080-SPRING2026/Problem Set 4/Julia/Outputs/mccall_value_functions.pdf")