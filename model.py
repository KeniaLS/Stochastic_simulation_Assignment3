from read_csv import read_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint




def system_equation(var, t, parameters):
 
    x = var[0]
    y = var[1]
    alpha, beta, delta, gamma = parameters
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    
    return dx, dy


def objective_function_MSE(parameters, x_value, y_value, time_points):
    
    # Simulate the model with the given parameters
    initial_c = [x_value[0], y_value[1]]
    solution = odeint(system_equation, initial_c, time_points, args=(parameters,))
    
    #model predictions for x and y
    model_predictions_x = solution[:, 0]
    model_predictions_y = solution[:, 1]

    mse_x = np.mean(((model_predictions_x - x_value)**2).astype(np.float64))
    mse_y = np.mean(((model_predictions_y - y_value)**2).astype(np.float64))
    total_mse = mse_x + mse_y
    
    return total_mse


def objective_function_RMSE(parameters, x_value, y_value, time_points):
    
    # Simulate the model with the given parameters
    initial_c = [x_value[0], y_value[1]]
    
    solution = odeint(system_equation, initial_c, time_points, args=(parameters,))
    
    #model predictions for x and y
    model_predictions_x = solution[:, 0]
    model_predictions_y = solution[:, 1]

    rmse_x = np.sqrt(np.mean((model_predictions_x - x_value)**2).astype(np.float64))
    rmse_y = np.sqrt(np.mean((model_predictions_y - y_value)**2).astype(np.float64))
    total_RMSE = rmse_x + rmse_y
    
    return total_RMSE


def objective_function_MAE(parameters, x_value, y_value, time_points):
    
    # Simulate the model with the given parameters
    initial_c = [x_value[0], y_value[1]]
    
    solution = odeint(system_equation, initial_c, time_points, args=(parameters,))
    
    #model predictions for x and y
    model_predictions_x = solution[:, 0]
    model_predictions_y = solution[:, 1]

    mae_x = np.mean(np.abs(model_predictions_x - x_value))
    mae_y = np.mean(np.abs(model_predictions_y - y_value))
    total_MAE = mae_x + mae_y
    
    return total_MAE



def hill_climbing(initial_guess, time_points, observed_data, iterations, step_size, obj_func):
    
    x_observed, y_observed = observed_data
    
    current_solution = initial_guess
    current_objective = objective_function_MSE(current_solution, x_observed, y_observed, time_points)

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


# Extract time and observed data
time_points , observed_x, observed_y = read_data()

parameters = []
for _ in range(20):
    # Set initial parameter values and other parameters
    iterations = 1000  
    step_size = 0.1  
    guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])
    given_data = [observed_x, observed_y]
    optimized_parameters = hill_climbing(guess, time_points, given_data , iterations, step_size, "MAE")
    parameters.append(optimized_parameters)
    

optimized_parameters = np.mean(np.array(parameters), axis = 0)
# Display the optimized parameters
print("Optimized Parameters:", optimized_parameters)

# Plot the original data and the model with optimized parameters
solution = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(optimized_parameters,))
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


