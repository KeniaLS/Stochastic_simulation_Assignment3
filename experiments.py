from read_csv import read_data
from model import *
from optimisation_func import * 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

    
    
# Extract time and observed data
time_points, observed_x, observed_y = read_data()
given_data = [observed_x, observed_y]

opt_error_rmse = []
opt_error_mae_sa = []
opt_error_mae_hill = []
opt_error_mse = []

no_iter = range(50, 1000, 50)

for iterations in no_iter:
    guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])
    param_SA = []
    param_hill = []
    for j in range(20):
        step_size = 0.1
        opt_param_sa = simulated_annealing(guess, time_points, given_data, iterations, 2, "MAE", "Linear")
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
    
    
    

# rmse_error = objective_function_RMSE(optimised_param, observed_x, observed_y, time_points)
# mse_error = objective_function_MSE(optimised_param, observed_x, observed_y, time_points)
# opt_error_rmse.append(rmse_error)
# opt_error_mse.append(mse_error)
# ci_rmse = 1.96 * np.std(opt_error_rmse)/np.sqrt(len(opt_error_rmse))
# ci_mse = 1.96 * np.std(opt_error_mse)/np.sqrt(len(opt_error_mse))


#some confidence interval
ci_mae_sa = 1.96 * np.std(opt_error_mae_sa)/np.sqrt(len(opt_error_mae_sa))
ci_mae_hill= 1.96 * np.std(opt_error_mae_hill)/np.sqrt(len(opt_error_mae_hill))


# Plot MAE and its confidence interval
plt.plot(no_iter, opt_error_mae_sa, label='MAE')
plt.fill_between(no_iter, opt_error_mae_sa - ci_mae_sa, opt_error_mae_sa + ci_mae_sa, color='blue', alpha=0.3)

# Plot RMSE and its confidence interval
# plt.plot(no_iter, opt_error_rmse, label='RMSE', color='orange')
# plt.fill_between(no_iter, opt_error_rmse - ci_rmse, opt_error_rmse + ci_rmse, color='orange', alpha=0.3)

plt.plot(no_iter, opt_error_mae_hill, label='hill climbing')
plt.fill_between(no_iter, (opt_error_mae_hill - ci_mae_hill), (opt_error_mae_sa + ci_mae_hill), alpha=.1)
plt.xlabel('Time')
plt.ylabel('Population Size')
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