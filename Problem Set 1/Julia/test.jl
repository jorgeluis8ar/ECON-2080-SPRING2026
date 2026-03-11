for (i, k) in enumerate(k_grid)
  println("Starting Value Function Iteration...")
  println(i)
  println("The value for the ",i,"-th capital grid point is ",k)
end

function bellman_operator(k_grid, V_current)
    n = length(k_grid)
    
    V_new = zeros(n)

    interp = linear_interpolation(k_grid, V_current, extrapolation_bc = Line())
    
    # Loop over each point in the capital grid
    for (i, k) in enumerate(k_grid)
        
        y = k^alpha
        # The objective is: u(y - k') + β·V(k')
        objective(k_prime) = -(log(y - k_prime[1]) + beta * interp(k_prime[1]))
        
        # Constraints for optimization:
        lower_bound = k_min
        upper_bound = y - 1e-10  # Slightly less than y to ensure c > 0
        
        # Solve the maximization problem
        # We use optimize() from Optim.jl package
        result = optimize(
            objective,           # Function to minimize (we negated it)
            [lower_bound],      # Lower bound for k'
            [upper_bound],      # Upper bound for k'
            [(k_min + upper_bound)/2],  # Initial guess (middle of feasible range)
            Fminbox(LBFGS())    # Optimization algorithm with box constraints
        )
        
        # Store the optimal value (negate back since we minimized the negative)
        V_new[i] = -result.minimum
    end
    
    return V_new
end

V_new = zeros(n)
interp = linear_interpolation(k_grid, zeros(n), extrapolation_bc = Line())
for (i, k) in enumerate(k_grid)
        
        y = k^alpha
        # The objective is: u(y - k') + β·V(k')
        objective(k_prime) = -(log(y - k_prime[1]) + beta * interp(k_prime[1]))
        
        # Constraints for optimization:
        lower_bound = k_min
        upper_bound = y - 1e-10  # Slightly less than y to ensure c > 0
        
        # Solve the maximization problem
        # We use optimize() from Optim.jl package
        result = optimize(
            objective,           # Function to minimize (we negated it)
            [lower_bound],      # Lower bound for k'
            [upper_bound],      # Upper bound for k'
            [(k_min + upper_bound)/2],  # Initial guess (middle of feasible range)
            Fminbox(LBFGS())    # Optimization algorithm with box constraints
        )
        
        # Store the optimal value (negate back since we minimized the negative)
        V_new[i] = -result.minimum
        println("The mininum value for the ",i,"-th capital grid point is ",V_new[i], "compared to the original k of ",k)
    end





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


value_function_iteration(k_grid, zeros(n), 
    max_iter=max_iterations, 
    tol=tolerance
)
