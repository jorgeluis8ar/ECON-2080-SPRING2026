# Author: Jorge Luis Ochoa Rincon
# ECON 2080 - Macroeconomics
# Dynamic Programming on Julia
# Deterministic Growth Model - Value Function Iteration
# This code solves the dynamic programming problem using the Bellman operator

# using Pkg                   # Package manager package 
using Interpolations          # Linear interpolation package
using Optim                   # Root finding method package

# ============================================================================
# PART 1: Set up parameters and grid
# ============================================================================

# Parameters
alpha = 0.65                    # Production function parameter
beta = 0.95                     # Discount factor
n = 150                         # Number of grid points

# Capital grid
k_min = 1e-6                # Small positive number to avoid log(0)
k_max = 2.0                 # Maximum capital
k_grid = range(k_min, k_max, length=n)  # Creates equidistant grid

# ============================================================================
# PART 2: Define the Bellman operator function
# ============================================================================

# Here I define the function that implements the Bellman operator. Documentation is also provided.
"""
    bellman_operator(k_grid, V_current)

Implements the Bellman operator for the deterministic growth model.

# Arguments
- `k_grid`: Vector of capital grid points. I.e., range(from,to,length)
- `V_current`: Current guess of the value function (vector of same length as k_grid). I.e., zeros(n)

# Returns
- `V_new`: Updated value function after applying the Bellman operator. I.e., vector of same length as k_grid

# Mathematical formulation:
The Bellman equation is:
    v(k) = max_{0 ≤ k' ≤ f(k)} {u(f(k) - k') + β·v(k')}

where:
- k is current capital
- k' is next period capital
- f(k) = k^alpha is the production function
- u(c) = log(c) is the utility function
- c = f(k) - k' is consumption
"""
function bellman_operator(k_grid, V_current)
    
    # Get number of grid points
    n = length(k_grid)
    
    # Initialize the new value function. This object will save the final estimation
    V_new = zeros(n)
    
    # Create interpolation object for the current value function
    interp = linear_interpolation(k_grid, V_current, extrapolation_bc = Line())
    
    # Loop over each point in the capital grid
    for (i, k) in enumerate(k_grid)
        
        # Production in current period
        y = k^alpha
        
        # Define the objective function to MAXIMIZE
        objective(k_prime) = -(log(y - k_prime[1]) + beta * interp(k_prime[1]))
        
        # Constraints for optimization:
        lower_bound = k_min         # Remember that k_min = 1e-5
        upper_bound = y - 1e-10     # Slightly less than y to ensure c > 0
        
        # Solve the maximization problem
        # We use optimize() from Optim.jl package
        result = optimize(
            objective,                  # Function to minimize (we negated it)
            [lower_bound],              # Lower bound for k'
            [upper_bound],              # Upper bound for k'
            [(k_min + upper_bound)/2],  # Initial guess (middle of feasible range)
            Fminbox(LBFGS())            # Optimization algorithm with box constraints
        )
        
        # Store the optimal value (negate back since we minimized the negative)
        V_new[i] = -result.minimum
    end
    
    return V_new
end

bellman_operator(k_grid, zeros(n))  # Test call to ensure function works
# The function returns the  grid after applying the Bellman operator to the current V.


# ============================================================================
# PART 3: Value Function Iteration Algorithm
# ============================================================================

"""
    value_function_iteration(k_grid, V_initial; max_iter=1000, tol=1e-6)

Solves the deterministic growth model using Value Function Iteration.

# Arguments
- `k_grid`: Vector of capital grid points
- `V_initial`: Initial guess of the value function
- `max_iter`: Maximum number of iterations (default: 1000)
- `tol`: Convergence tolerance (default: 1e-6)

# Returns
- `V_sequence`: Array where each column is the value function at iteration i
- `num_iterations`: Actual number of iterations performed

# Convergence criterion:
The algorithm iterates until ||V_{n+1} - V_n||_∞ < tol or until max_iter is reached.
"""
function value_function_iteration(k_grid, V_initial; max_iter=1000, tol=1e-6)
    # Get the number of grid points
    n = length(k_grid)
    
    # Initialize storage for the sequence of value functions
    # Each column will store the value function at one iteration
    V_sequence = zeros(n, max_iter + 1)
    
    # Store the initial guess in the first column
    V_sequence[:, 1] = V_initial
    
    # Current value function
    V_current = copy(V_initial)
    
    # Actual number of iterations performed
    num_iterations = 0
    
    println("Starting Value Function Iteration...")
    println("=" ^ 60)
    println("Initial guess: $(V_initial[1] == 0.0 ? "zeros" : "custom")")
    println()
    
    # Main VFI loop
    for it in 1:max_iter
        # Apply Bellman operator to get updated value function
        V_new = bellman_operator(k_grid, V_current)
        
        # Store the new value function in the sequence
        V_sequence[:, it + 1] = V_new
        
        # Check convergence: maximum absolute difference (sup norm)
        diff = maximum(abs.(V_new .- V_current))
        
        # Print progress every 10 iterations
        if it % 10 == 0
            println("Iteration $it: ||V_{n+1} - V_n||_∞ = $(round(diff, digits=8))")
        end
        
        # Check if converged
        if diff < tol
            println()
            println("=" ^ 60)
            println("✓ CONVERGED after $it iterations!")
            println("Final sup norm: ||V_{n+1} - V_n||_∞ = $(round(diff, digits=10))")
            println("Tolerance: $tol")
            println("=" ^ 60)
            num_iterations = it
            # Trim the sequence to only include computed iterations
            V_sequence = V_sequence[:, 1:(it + 1)]
            break
        end
        
        # Update current value function for next iteration
        V_current = V_new
        
        # Check if we hit max iterations
        if it == max_iter
            println()
            println("=" ^ 60)
            println("⚠ Warning: Maximum iterations ($max_iter) reached")
            println("Final sup norm: ||V_{n+1} - V_n||_∞ = $(round(diff, digits=10))")
            println("Algorithm did not converge to tolerance $tol")
            println("=" ^ 60)
            num_iterations = it
            # V_sequence already contains all iterations
        end
    end
    
    return V_sequence, num_iterations
end

# ============================================================================
# PART 4: Run the VFI algorithm and plot convergence
# ============================================================================

using Plots

# Set initial guess for value function: v₀(k) = 5log(k) - 25
V_initial = 5 .* log.(k_grid) .- 25

# Run VFI algorithm with specified parameters
max_iterations = 500
tolerance = 1e-6

println("Running VFI with max_iter=$max_iterations, tol=$tolerance")
println()

V_sequence, num_iterations = value_function_iteration(
    k_grid, 
    V_initial, 
    max_iter=max_iterations, 
    tol=tolerance
)

# Extract the final converged value function
V_final = V_sequence[:, end]

# Display results
println("\n" * "=" ^ 60)
println("Solution Summary:")
println("=" ^ 60)
println("Number of iterations: $num_iterations")
println("Shape of V_sequence: $(size(V_sequence))")
println("  - Rows: $(size(V_sequence, 1)) (capital grid points)")
println("  - Columns: $(size(V_sequence, 2)) (iterations: 0 to $num_iterations)")
println()
println("Final value function at selected capital levels:")
println("-" ^ 60)

# ============================================================================
# Plot the convergence of the algorithm
# ============================================================================

println("\nGenerating convergence plot...")

# Select iterations to plot (every 5 iterations)
iterations_to_plot = 1:10:size(V_sequence, 2)
iterations_to_plot = collect(iterations_to_plot)
push!(iterations_to_plot, size(V_sequence,2))
# Create the plot
p = plot(
    xlabel="Capital stock k",
    ylabel="Value function v(k)",
    title="Convergence of Value Function Iteration",
    legend=:bottomright,
    size=(800, 600),
    linewidth=2,
    grid=true
)

# Plot value functions at different iterations
for (i, iter_idx) in enumerate(iterations_to_plot)
    iteration_number = iter_idx - 1  # Subtract 1 because column 1 is iteration 0
    
    # Determine line style
    if iteration_number == 0
        # Initial guess - make it dashed and distinctive
        plot!(p, k_grid, V_sequence[:, iter_idx], 
              label="Iteration 0 (initial)", 
              linestyle=:dot,
              color=:blue,
              linewidth=2.5)
    elseif iter_idx == size(V_sequence, 2)
        # Final iteration - make it bold
        plot!(p, k_grid, V_sequence[:, iter_idx], 
              label="Iteration $iteration_number (final)", 
              color=:green,
              linewidth=3)
    else
        # Intermediate iterations
        plot!(p, k_grid, V_sequence[:, iter_idx], 
              label="Iteration $iteration_number",
              alpha=0.6)
    end
end

# Save the plot
savefig(p, "~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/Problem Sets/Problem Set 1/Julia/Outputs/converge_plot_part4.pdf")


# Display the plot (if in interactive environment)
display(p)


# ============================================================================
# PART 5: Plotting the analytical solution to the problem
# ============================================================================


V_star = (1 ./(1 .-beta)).*(log(1 .-alpha .* beta) .+ ((alpha .* beta)./(1 .-alpha .* beta)) .* log.(alpha.*beta)) .+ (alpha./(1 .-alpha.*beta)).*log.(k_grid)

plot!(p, k_grid, V_star, 
              label="Analytical Solution", 
              linestyle=:dash,
              color=:red,
              linewidth=2.5)

savefig(p, "~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/Problem Sets/Problem Set 1/Julia/Outputs/converge_plot_part5.pdf")

# Display the plot (if in interactive environment)
display(p)

# ============================================================================
# PART 6: Does a good initial guess matter?
# ============================================================================

using Random

# Setting a seed for reproducibility
Random.seed!(42)
# Set initial guess for value function: v₀(k) = 5log(k) - 25
V_initial_part6 = rand(1:10) .* log.(k_grid) .- rand(0:50) .* (.-1)

V_sequence_part6, num_iterations = value_function_iteration(
    k_grid, 
    V_initial_part6, 
    max_iter=max_iterations, 
    tol=tolerance
)

# Extract the final converged value function
V_final_part6 = V_sequence_part6[:, end]

# Display results
println("\n" * "=" ^ 60)
println("Solution Summary:")
println("=" ^ 60)
println("Number of iterations: $num_iterations")
println("Shape of V_sequence_part6: $(size(V_sequence_part6))")
println("  - Rows: $(size(V_sequence_part6, 1)) (capital grid points)")
println("  - Columns: $(size(V_sequence_part6, 2)) (iterations: 0 to $num_iterations)")
println()
println("-" ^ 60)

# Select iterations to plot (every 5 iterations)
iterations_to_plot_p6 = 1:20:size(V_sequence_part6, 2)
iterations_to_plot_p6 = collect(iterations_to_plot_p6)
push!(iterations_to_plot_p6, size(V_sequence_part6,2))
# Create the plot
plot_part6 = plot(
    xlabel="Capital stock k",
    ylabel="Value function v(k)",
    title="Convergence of Value Function Iteration",
    legend=:bottomright,
    size=(800, 600),
    linewidth=2,
    grid=true
)

# Plot value functions at different iterations
for (i, iter_idx) in enumerate(iterations_to_plot_p6)
    iteration_number = iter_idx - 1  # Subtract 1 because column 1 is iteration 0
    
    # Determine line style
    if iteration_number == 0
        # Initial guess - make it dashed and distinctive
        plot!(plot_part6, k_grid, V_sequence_part6[:, iter_idx], 
              label="Iteration 0 (initial)", 
              linestyle=:dot,
              color=:blue,
              linewidth=2.5)
    elseif iter_idx == size(V_sequence_part6, 2)
        # Final iteration - make it bold
        plot!(plot_part6, k_grid, V_sequence_part6[:, iter_idx], 
              label="Iteration $iteration_number (final)", 
              color=:green,
              linewidth=3)
    else
        # Intermediate iterations
        plot!(plot_part6, k_grid, V_sequence_part6[:, iter_idx], 
              label="Iteration $iteration_number",
              alpha=0.6)
    end
end

plot!(plot_part6, k_grid, V_star, 
              label="Analytical Solution", 
              linestyle=:dash,
              color=:red,
              linewidth=2.5)

savefig(plot_part6, "~/Library/Mobile Documents/com~apple~CloudDocs/PhD Economics - Brown/Spring 2026/Macroeconomics 2/Problem Sets/Problem Set 1/Julia/Outputs/converge_plot_part6.pdf")

display(plot_part6)

# ============================================================================
# PART 7: Optional Different values of parameters and problem 1?
# ============================================================================