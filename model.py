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



def logarithmic_cooling(initial_temperature, iteration):
    return initial_temperature / np.log(1 + iteration)

def linear_cooling(temp, cooling_rate):
    return temp - cooling_rate

def exponential_cooling(temp, cooling_rate):
    return temp * cooling_rate

    


