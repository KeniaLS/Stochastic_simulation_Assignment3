from read_csv import read_data
from model import *
from optimisation_func import * 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
    

# Extract time and observed data
time_points, observed_x, observed_y = read_data()
given_data = [observed_x, observed_y]

min_error = 5

mse_hill = []
sa_mse = []
sa_rmse = []
sa_mae = []

no_iter = range(10, 500, 50)
for iterations in no_iter:
    guess = np.array([2.58362893, 1.49692846, 0.28088481, 0.55652727])
    guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])
    param_sa_mse = []
    param_sa_mae = []
    param_sa_rmse = []
    for j in range(20):
        step_size = 0.1
        para_sa_MSE = simulated_annealing(guess, time_points, given_data, iterations, 5, "MSE", "Exp")
        para_sa_MAE = simulated_annealing(guess, time_points, given_data, iterations, 5, "MAE", "Exp")
        para_sa_RMSE = simulated_annealing(guess, time_points, given_data, iterations, 5, "RMSE", "Exp")
        param_sa_mse.append(para_sa_MSE)
        param_sa_rmse.append(para_sa_RMSE)
        param_sa_mae.append(para_sa_MAE)  
        
        print("simulation", j)
    
    opt_param_sa_mse = np.mean(np.array(param_sa_mse), axis =0)
    opt_param_sa_rmse = np.mean(np.array(param_sa_rmse), axis =0)
    opt_param_sa_mae = np.mean(np.array(param_sa_mae), axis =0)
    mse_error_sa = objective_function_MSE(opt_param_sa_mse, observed_x, observed_y, time_points)
    rmse_error_sa = objective_function_RMSE(opt_param_sa_rmse, observed_x, observed_y, time_points)
    mae_error_sa = objective_function_MAE(opt_param_sa_mae, observed_x, observed_y, time_points)
    if  mse_error_sa < min_error:
        min_error = mse_error_sa
        optimum_param = opt_param_sa_mse
        
    sa_mse.append(mse_error_sa)
    sa_rmse.append(rmse_error_sa)
    sa_mae.append(mae_error_sa)
    
    
#confidence interval for MSE 
ci_mse_sa = 1.96 * np.std(sa_mse)/np.sqrt(len(sa_mse))
ci_mae_sa = 1.96 * np.std(sa_mae)/np.sqrt(len(sa_mae))
ci_rmse_sa = 1.96 * np.std(sa_rmse)/np.sqrt(len(sa_rmse))
# Plot MAE and its confidence interval

color_mse = 'blue'
color_mae = 'orange'
color_rmse = 'green'

# Plot MSE
plt.plot(no_iter, sa_mse, label='MSE', color=color_mse)
plt.fill_between(no_iter, sa_mse - ci_mse_sa, sa_mse + ci_mse_sa, alpha=0.3, color=color_mse)

# Plot MAE
plt.plot(no_iter, sa_mae, label='MAE', color=color_mae)
plt.fill_between(no_iter, sa_mae - ci_mae_sa, sa_mae + ci_mae_sa, alpha=0.3, color=color_mae)

# Plot RMSE
plt.plot(no_iter, sa_rmse, label='RMSE', color=color_rmse)
plt.fill_between(no_iter, sa_rmse - ci_rmse_sa, sa_rmse + ci_rmse_sa, alpha=0.3, color=color_rmse)

plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.savefig("Error_vs_iterations.svg")
plt.show()

print("optimum_param")



# Plot the original data and the model with optimized parameters
solution_sa = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(optimum_param,))
model_predictions_x = solution_sa[:, 0]  # Adjust based on your model structure
model_predictions_y = solution_sa[:, 1]  
plt.scatter(time_points, observed_x, label='Observed X')
plt.plot(time_points, solution_sa[:, 0])
plt.scatter(time_points, observed_y, label='Observed Y')
plt.plot(time_points, solution_sa[:, 1])
plt.show()

opt_error_mse_sa = []
opt_error_mae_sa = []


no_iter = range(50, 2000, 100)
temp_range = range(1, 50, 1)

for temp in temp_range:
    iterations = 500
    print(temp)
    guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])
    param_SA_MSE = []
    param_SA_MAE = []
    for j in range(20):
        step_size = 0.1
        opt_param_sa_MSE = simulated_annealing(guess, time_points, given_data, iterations, temp, "MSE", "Logarithmic")
        opt_param_sa_MAE = simulated_annealing(guess, time_points, given_data, iterations, temp, "MAE", "Logarithmic")
        param_SA_MSE.append(opt_param_sa_MAE)
        param_SA_MAE.append(opt_param_sa_MSE) 
        print("simulation", j)
    
    optimised_param_sa_MSE = np.mean(np.array(param_SA_MSE), axis =0)
    optimised_param_sa_MAE = np.mean(np.array(param_SA_MAE), axis =0)
    mae_error_sa_MSE = objective_function_MAE(optimised_param_sa_MSE, observed_x, observed_y, time_points)
    mae_error_sa_MAE = objective_function_MAE(optimised_param_sa_MAE, observed_x, observed_y, time_points)
    opt_error_mse_sa.append(mae_error_sa_MSE)
    opt_error_mae_sa.append(mae_error_sa_MAE)


#confidence interval for MAE 
ci_mse_sa = 1.96 * np.std(opt_error_mse_sa)/np.sqrt(len(opt_error_mse_sa))
ci_mae_sa= 1.96 * np.std(opt_error_mae_sa)/np.sqrt(len(opt_error_mae_sa))
# Plot MAE and its confidence interval
plt.plot(temp_range, opt_error_mse_sa, label='MSE')
plt.fill_between(temp_range, opt_error_mse_sa - ci_mse_sa, opt_error_mse_sa + ci_mse_sa, color='blue', alpha=0.3)
plt.plot(temp_range, opt_error_mae_sa, label='MAE')
plt.fill_between(temp_range, opt_error_mae_sa - ci_mae_sa, opt_error_mae_sa + ci_mae_sa, color='orange', alpha=0.3)
plt.xlabel('Temperature')
plt.ylabel('Error')
plt.legend()
plt.show()