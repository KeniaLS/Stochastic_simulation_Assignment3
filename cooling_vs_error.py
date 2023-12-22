from read_csv import read_data
from model import *
from optimisation_func import * 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

# Read data from CSV file
time_points, observed_x, observed_y = read_data()
given_data = [observed_x, observed_y]

# Initialize empty lists to store MSE errors for different models
opt_mse_log = []
opt_mse_lin = []
opt_mse_exp = []

# Define a range of iterations for simulated annealing
no_iter = range(10, 800, 50)

# Loop through different numbers of iterations for simulated annealing
for iterations in no_iter:
    
    # Initial guess for model parameters
    guess = np.array([2.58362893, 1.49692846, 0.45088481, 0.75652727])
    
    # Lists to store parameters for each simulation run
    param_sa_lin = []
    param_sa_log = []
    param_sa_exp = []
    
    for j in range(40):
        step_size = 0.1
        # Run simulated annealing for Linear Cooling
        opt_param_sa_linear = simulated_annealing(guess, time_points, given_data, iterations, 5, "MSE", "Linear")
        # Run simulated annealing for Exponential Cooling Schedule
        opt_param_sa_exp = simulated_annealing(guess, time_points, given_data, iterations, 5, "MSE", "Exponential")
        # Run simulated annealing for Logarithmic Cooling
        opt_param_sa_log = simulated_annealing(guess, time_points, given_data, iterations, 5, "MSE", "Logarithmic")
        # Append optimized parameters to respective lists
        param_sa_lin.append(opt_param_sa_linear) 
        param_sa_log.append(opt_param_sa_log) 
        param_sa_exp.append(opt_param_sa_exp) 
        print("simulation", j)
    
    # Calculate mean of optimized parameters for each model
    optimised_param_lin = np.mean(np.array(param_sa_lin), axis =0)
    optimised_param_exp = np.mean(np.array(param_sa_exp), axis =0)
    optimised_param_log = np.mean(np.array(param_sa_log), axis =0)
    
    # Calculate MSE errors for each model using the mean parameters
    mse_error_lin = objective_function_MSE(optimised_param_lin, observed_x, observed_y, time_points)
    mse_error_exp = objective_function_MSE(optimised_param_exp, observed_x, observed_y, time_points)
    mse_error_log = objective_function_MSE(optimised_param_log, observed_x, observed_y, time_points)
    
    # Append the MSE errors to respective lists
    opt_mse_lin.append(mse_error_lin)
    opt_mse_exp.append(mse_error_exp)
    opt_mse_log.append(mse_error_log)
    
#confidence interval for MSE 
ci_mse_lin = 1.96 * np.std(opt_mse_lin)/np.sqrt(len(opt_mse_lin))
ci_mse_log= 1.96 * np.std(opt_mse_log)/np.sqrt(len(opt_mse_log))
ci_mse_exp= 1.96 * np.std(opt_mse_exp)/np.sqrt(len(opt_mse_exp))

# Plot MSE and its confidence interval for Linear
plt.plot(no_iter, opt_mse_lin, label='Linear', color='blue')
plt.fill_between(no_iter, opt_mse_lin - ci_mse_lin, opt_mse_lin + ci_mse_lin, color='blue', alpha=0.3)

# Plot MSE and its confidence interval for Logarithmic
plt.plot(no_iter, opt_mse_log, label='Logarithmic',  color='red')
plt.fill_between(no_iter, opt_mse_log - ci_mse_log, opt_mse_log + ci_mse_log, color='red', alpha=0.3)

# Plot MSE and its confidence interval for Exponential
plt.plot(no_iter, opt_mse_exp, label='Exponential',  color='green')
plt.fill_between(no_iter, opt_mse_exp - ci_mse_exp, opt_mse_exp + ci_mse_exp, color='green', alpha=0.3)

plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.savefig("Cooling_vs_MSE.svg")
plt.show()
