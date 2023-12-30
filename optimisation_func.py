from read_csv import read_data
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint





def hill_climbing(initial_guess, time_points, observed_data, iterations, step_size, obj_func):
    """
    Perform hill climbing optimization.

    Parameters:
    - initial_guess (array): Initial guess for the optimization.
    - time_points (array): Time points for the observed data.
    - observed_data (tuple): Tuple containing x and y observed data arrays.
    - iterations (int): Number of iterations for the optimization.
    - step_size (float): Size of the step to generate a random neighbor.
    - obj_func (str): Objective function type (MSE, RMSE, or MAE).

    Returns:
    - array: Optimized parameters.
    """
    
    x_observed, y_observed = observed_data
    current_solution = initial_guess
    if obj_func == "MSE":
        current_objective = objective_function_MSE(current_solution, x_observed, y_observed, time_points)
    elif obj_func == "RMSE":
            current_objective = objective_function_RMSE(current_solution, x_observed, y_observed, time_points)
    elif obj_func == "MAE":
            current_objective = objective_function_MAE(current_solution, x_observed, y_observed, time_points)


    
    for _ in range(iterations):
        
        # Generate a random neighbor within the specified step_size
        neighbor = current_solution + np.random.uniform(-step_size, step_size, size=len(current_solution))
        
        # Evaluate the objective function for the neighbor
        if obj_func == "MSE":
            neighbor_objective = objective_function_MSE(neighbor, x_observed, y_observed, time_points)
        elif obj_func == "RMSE":
            neighbor_objective = objective_function_RMSE(neighbor, x_observed, y_observed, time_points)
        elif obj_func == "MAE":
            neighbor_objective = objective_function_MAE(neighbor, x_observed, y_observed, time_points)

        # If the neighbor is better, accept the move
        if neighbor_objective < current_objective:
            current_solution = neighbor
            current_objective = neighbor_objective

    return current_solution
 


def simulated_annealing(initial_guess, time_points, observed_data, iterations, temp, obj_func, cooling_type):
    """
    Perform simulated annealing optimization.

    Parameters:
    - initial_guess (array): Initial guess for the optimization.
    - time_points (array): Time points for the observed data.
    - observed_data (tuple): Tuple containing x and y observed data arrays.
    - iterations (int): Number of iterations for the optimization.
    - temp (float): Initial temperature for simulated annealing.
    - obj_func (str): Objective function type (MSE, RMSE, or MAE).
    - cooling_type (str): Type of cooling schedule (Linear, Exp, or Logarithmic).

    Returns:
    - array: Optimized parameters.
    """
    
    x_value, y_value = observed_data
    current_temperature  = temp
    step_size = 0.1
    
    # generate an initial point
    best = initial_guess
    
    if obj_func == "RMSE":
        best_eval = objective_function_RMSE(best, x_value, y_value, time_points)      # evaluate the initial point
    elif obj_func == "MSE":
        best_eval = objective_function_MSE(best, x_value, y_value, time_points)  
    elif obj_func == "MAE":
        best_eval = objective_function_MAE(best, x_value, y_value, time_points)  
        
    curr, curr_eval = best, best_eval                           # current working solution
    
    for i in range(1, iterations+1):
        
        # take a step
        candidate = curr + step_size * np.random.normal(size=len(curr))
        
        if obj_func == "RMSE":
            candidate_eval = objective_function_RMSE(candidate, x_value, y_value, time_points)                   
        elif obj_func == "MSE":
            candidate_eval = objective_function_MSE(candidate, x_value, y_value, time_points)
        elif obj_func == "MAE":
            candidate_eval = objective_function_MAE(candidate, x_value, y_value, time_points)
        
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval                   # store new best point
        
        if cooling_type == "Linear":
            current_temperature = linear_cooling(current_temperature, 0.001)
        elif cooling_type == "Exp":
            current_temperature = exponential_cooling(current_temperature, 0.95)
        elif cooling_type == "Logarithmic":
            current_temperature = logarithmic_cooling(temp, i)
    
        diff = candidate_eval - curr_eval 
        
        metropolis = np.exp(-diff / current_temperature)                           # calculate metropolis acceptance criterion
        if diff < 0 or np.random.rand() < metropolis:         # check if we should keep the new point
            curr, curr_eval = candidate, candidate_eval        # store the new current point
    return best

