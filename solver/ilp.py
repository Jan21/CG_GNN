import pulp
import numpy as np

def solve_ilp(A, b, c):
    """
    Parameters:
    A (numpy.ndarray): Constraint coefficient matrix
    b (numpy.ndarray): Right-hand side values of constraints
    c (numpy.ndarray): Objective function coefficients
    
    Returns:
    tuple: (optimal solution vector, optimal objective value)
    """
    num_variables = len(c)
    
    # Create the model
    prob = pulp.LpProblem("ILP", pulp.LpMaximize)
    
    # Create binary variables
    x = pulp.LpVariable.dicts("x", range(num_variables), cat='Binary')
    
    # Set objective function
    prob += pulp.lpSum(c[i] * x[i] for i in range(num_variables))
    
    # Add constraints
    for i in range(len(b)):
        prob += pulp.lpSum(A[i][j] * x[j] for j in range(num_variables)) <= b[i]
    
    # Solve the problem
    # Disable pulp logging
    pulp.LpSolverDefault.msg = 0
    prob.solve()
    
    # Get the solution
    solution = np.zeros(num_variables)
    for i in range(num_variables):
        solution[i] = pulp.value(x[i])
    
    return solution, np.float32(pulp.value(prob.objective))
