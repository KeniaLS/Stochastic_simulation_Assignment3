from read_csv import read_data
from model import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

 
    


def simulated_annealing(initial_guess, time_points, observed_data, iterations, step_size, obj_func, temp):
    
    x_value, y_value = observed_data
    # generate an initial point
    best = initial_guess
    
    if obj_func == "RMSE":
        best_eval = objective_function_RMSE(best, x_value, y_value, time_points)      # evaluate the initial point
    elif obj_func == "MSE":
        best_eval = objective_function_MSE(best, x_value, y_value, time_points)  
    elif obj_func == "MAE":
        best_eval = objective_function_MAE(best, x_value, y_value, time_points)  
        
    curr, curr_eval = best, best_eval                           # current working solution
    
    for i in range(iterations):
        
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
        
        #print('>%d f(%s) = %.5f' % (i, best, best_eval))                  # report progress
        
        diff = candidate_eval - curr_eval 
        t = temp / float(i + 1)                              # calculate temperature for current epoch
        metropolis = np.exp(-diff / t)                           # calculate metropolis acceptance criterion
        if diff < 0 or np.random.rand() < metropolis:         # check if we should keep the new point
            curr, curr_eval = candidate, candidate_eval        # store the new current point
    return best



# Extract time and observed data
time_points, observed_x, observed_y = read_data()
given_data = [observed_x, observed_y]
opt_param = []

guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])
for j in range(50):
    no_iterations = 1000
    step_size = 0.1
    param = simulated_annealing(guess, time_points, given_data, no_iterations, step_size, "MSE", 5)
    opt_param.append(param) 
    print("simulation", j)

optimised_param = np.mean(np.array(opt_param), axis =0)
mse_error = objective_function_RMSE(optimised_param, observed_x, observed_y, time_points)

#opt_error = np.mean(avg_error)
# Display the optimized parameters
print("Optimized Parameters for simulated annealing:", optimised_param, mse_error)

# Plot the original data and the model with optimized parameters
solution = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(optimised_param,))
model_predictions_x = solution[:, 0]  # Adjust based on your model structure
model_predictions_y = solution[:, 1] 


plt.scatter(time_points, observed_x, label='Observed X')
plt.plot(time_points, model_predictions_x, label='Model Predictions X')

plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.show()

plt.scatter(time_points, observed_y, label='Observed Y', color = "red")
plt.plot(time_points, model_predictions_y, "r", label='Model Predictions Y')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.show()