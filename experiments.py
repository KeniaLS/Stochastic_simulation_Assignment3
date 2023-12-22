from read_csv import read_data
from model import *
from optimisation_func import * 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
    
    
# Extract time and observed data
time_points, observed_x, observed_y = read_data()
given_data = [observed_x, observed_y]


opt_error_mae_sa = []
opt_error_mae_hill = []


no_iter = range(50, 2000, 100)

for iterations in no_iter:
    guess = np.array([0.5, 0.1, 0.1, 0.5])
    param_SA = []
    param_hill = []
    for j in range(20):
        step_size = 0.1
        opt_param_sa = simulated_annealing(guess, time_points, given_data, iterations, 100, "MAE", "Exp")
        opt_param_hill = hill_climbing(guess, time_points, given_data , iterations, step_size, "MAE")
        param_hill.append(opt_param_hill)
        param_SA.append(opt_param_sa) 
        print("simulation", j)
    
    optimised_param_sa = np.mean(np.array(param_SA), axis =0)
    optimised_param_h = np.mean(np.array(param_hill), axis =0)
    mae_error_sa = objective_function_MAE(optimised_param_sa, observed_x, observed_y, time_points)
    mae_error_hill = objective_function_MAE(optimised_param_h, observed_x, observed_y, time_points)
    opt_error_mae_sa.append(mae_error_sa)
    opt_error_mae_hill.append(mae_error_hill)


#confidence interval for MAE 
ci_mae_sa = 1.96 * np.std(opt_error_mae_sa)/np.sqrt(len(opt_error_mae_sa))
ci_mae_hill= 1.96 * np.std(opt_error_mae_hill)/np.sqrt(len(opt_error_mae_hill))
# Plot MAE and its confidence interval
plt.plot(no_iter, opt_error_mae_sa, label='SA')
plt.fill_between(no_iter, opt_error_mae_sa - ci_mae_sa, opt_error_mae_sa + ci_mae_sa, color='blue', alpha=0.3)
plt.plot(no_iter, opt_error_mae_hill, label='SA')
plt.fill_between(no_iter, opt_error_mae_hill - ci_mae_hill, opt_error_mae_hill + ci_mae_hill, color='blue', alpha=0.3)
plt.xlabel('Iterations')
plt.ylabel('MAE')
plt.legend()
plt.show()


opt_error_mse_sa = []
opt_error_mse_hill = []
no_iter = range(50, 2000, 100)
for iterations in no_iter:
    guess = np.array([0.5, 0.1, 0.1, 0.5])
    param_SA = []
    param_hill = []
    for j in range(20):
        step_size = 0.1
        opt_param_sa = simulated_annealing(guess, time_points, given_data, iterations, 100, "MSE", "Exp")
        opt_param_hill = hill_climbing(guess, time_points, given_data , iterations, step_size, "MSE")
        param_hill.append(opt_param_hill)
        param_SA.append(opt_param_sa) 
        print("simulation", j)
    
    optimised_param_sa = np.mean(np.array(param_SA), axis =0)
    optimised_param_h = np.mean(np.array(param_hill), axis =0)
    mse_error_sa = objective_function_MAE(optimised_param_sa, observed_x, observed_y, time_points)
    mse_error_hill = objective_function_MAE(optimised_param_h, observed_x, observed_y, time_points)
    opt_error_mse_sa.append(mse_error_sa)
    opt_error_mse_hill.append(mse_error_hill)
    
#confidence interval for MSE 
ci_mse_sa = 1.96 * np.std(opt_error_mse_sa)/np.sqrt(len(opt_error_mse_sa))
ci_mse_hill= 1.96 * np.std(opt_error_mse_hill)/np.sqrt(len(opt_error_mse_hill))
# Plot MAE and its confidence interval
plt.plot(no_iter, opt_error_mse_sa, label='SA')
plt.fill_between(no_iter, opt_error_mse_sa - ci_mse_sa, opt_error_mse_sa + ci_mse_sa, color='blue', alpha=0.3)
plt.plot(no_iter, opt_error_mse_hill, label='SA')
plt.fill_between(no_iter, opt_error_mse_hill - ci_mse_hill, opt_error_mse_hill + ci_mse_hill, color='red', alpha=0.3)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.legend()
plt.show()

opt_error_rmse_sa = []
opt_error_rmse_hill = []

for iterations in no_iter:
    guess = np.array([0.5, 0.1, 0.1, 0.5])
    param_SA = []
    param_hill = []
    for j in range(20):
        step_size = 0.1
        opt_param_sa = simulated_annealing(guess, time_points, given_data, iterations, 100, "RMSE", "Exp")
        opt_param_hill = hill_climbing(guess, time_points, given_data , iterations, step_size, "RMSE")
        param_hill.append(opt_param_hill)
        param_SA.append(opt_param_sa) 
        print("simulation", j)
    
    optimised_param_sa = np.mean(np.array(param_SA), axis =0)
    optimised_param_h = np.mean(np.array(param_hill), axis =0)
    mse_error_sa = objective_function_MAE(optimised_param_sa, observed_x, observed_y, time_points)
    mse_error_hill = objective_function_MAE(optimised_param_h, observed_x, observed_y, time_points)
    opt_error_rmse_sa.append(mse_error_sa)
    opt_error_rmse_hill.append(mse_error_hill)


#confidence interval for MSE 
ci_rmse_sa = 1.96 * np.std(opt_error_rmse_sa)/np.sqrt(len(opt_error_rmse_sa))
ci_rmse_hill= 1.96 * np.std(opt_error_rmse_hill)/np.sqrt(len(opt_error_rmse_hill))
# Plot MAE and its confidence interval
plt.plot(no_iter, opt_error_rmse_sa, label='SA')
plt.fill_between(no_iter, opt_error_rmse_sa - ci_rmse_sa, opt_error_rmse_sa + ci_rmse_sa, color='blue', alpha=0.3)
plt.plot(no_iter, opt_error_rmse_hill, label='SA')
plt.fill_between(no_iter, opt_error_rmse_hill - ci_rmse_hill, opt_error_rmse_hill + ci_rmse_hill, color='blue', alpha=0.3)
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.legend()
plt.show()



# Plot the original data and the model with optimized parameters
solution_sa = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(optimised_param_sa,))
model_predictions_x = solution_sa[:, 0]  # Adjust based on your model structure
model_predictions_y = solution_sa[:, 1]  
plt.scatter(time_points, observed_x, label='Observed X')
plt.plot(time_points, solution_sa[:, 0])
plt.scatter(time_points, observed_y, label='Observed Y')
plt.plot(time_points, solution_sa[:, 1])
plt.show()


# Plot the original data and the model with optimized parameters
solution_hill = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(optimised_param_h,))
model_predictions_x = solution_hill[:, 0]  # Adjust based on your model structure
model_predictions_y = solution_hill[:, 1]  
plt.scatter(time_points, observed_x, label='Observed X')
plt.plot(time_points, solution_hill[:, 0])
plt.scatter(time_points, observed_y, label='Observed Y')
plt.plot(time_points, solution_hill[:, 1])
plt.show()

print("SA", optimised_param_sa)
print("hill", optimised_param_h)

