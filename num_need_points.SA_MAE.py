from read_csv import read_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from helper_functions import *





def system_equation(var, t, parameters):
 """
    Define the system of differential equations.

    Parameters:
    - var (array): State variables [x, y].
    - t (float): Time.
    - parameters (array): System parameters [alpha, beta, delta, gamma].

    Returns:
    - array: Derivatives [dx, dy].
    """
    x = var[0]
    y = var[1]
    alpha, beta, delta, gamma = parameters
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    
    return dx, dy


# Function to remove data points from a time series
def remove_data_points(data, indices_to_remove):
 """
    Remove data points from a time series.

    Parameters:
    - data (array): Time series data.
    - indices_to_remove (array): Indices of data points to be removed.

    Returns:
    - array: Time series data with specified points removed.
    """
    data_removed = np.copy(data)
    data_removed_per_index = np.delete(data_removed, indices_to_remove, axis=0)

    return data_removed_per_index


def random_choice(observed_x, observed_y, time_points, num_points):
 """
    Randomly select and remove data points from observed time series.

    Parameters:
    - observed_x (array): Observed x values.
    - observed_y (array): Observed y values.
    - time_points (array): Time points.
    - num_points (int): Number of points to be removed.

    Returns:
    - Tuple: Tuple containing updated x, y, and time_points.
    """
    rand_choice = np.random.choice(len(time_points), size=num_points)

    observed_x_removed = remove_data_points(observed_x, rand_choice)
    observed_y_removed = remove_data_points(observed_y, rand_choice)
    time_points_removed = remove_data_points(time_points, rand_choice)

    return observed_x_removed, observed_y_removed, time_points_removed


    

def objective_function_MAE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask):
 """
    Objective function for Mean Absolute Error (MAE) with masked x and y data.

    Parameters:
    - parameters (array): Model parameters [alpha, beta, delta, gamma].
    - x_data_mask (array): Masked observed x values.
    - y_data_mask (array): Masked observed y values.
    - t_data_mask (array): Masked time points.

    Returns:
    - Tuple: MAE for x and y.
    """

    initial_c = [x_data_mask[0], y_data_mask[0]]
    solution = odeint(system_equation, initial_c, t_data_mask, args=(parameters,))
    
    model_predictions_x_mask = solution[:, 0]
    model_predictions_y_mask = solution[:, 1]

    mae_x_mask = np.mean(np.abs(model_predictions_x_mask - x_data_mask))
    mae_y_mask = np.mean(np.abs(model_predictions_y_mask - y_data_mask))
    
    return mae_x_mask, mae_y_mask






def objective_function_MAE_x_mask(parameters, x_value, y_value, t_value, num_points): # Y value fixed
 """
    Objective function for Mean Absolute Error (MAE) with masked x data.

    Parameters:
    - parameters (array): Model parameters [alpha, beta, delta, gamma].
    - x_value (array): Observed x values.
    - y_value (array): Observed y values.
    - t_value (array): Time points.
    - num_points (int): Number of points to be removed.

    Returns:
    - Tuple: Total MAE, MAE for x, and MAE for y.
    """

    x_data_mask, y_data_mask, t_data_mask = random_choice(x_value, y_value, t_value, num_points)   


    # Simulate the model with the given parameters
    initial_c = [x_value[0], y_value[0]]
    solution = odeint(system_equation, initial_c, t_value, args=(parameters,))
    model_predictions_y = solution[:, 1]


    mae_x_mask, _ = objective_function_MAE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask)
    mae_y = np.mean(np.abs(model_predictions_y - y_value))
    total_mae_x_mask = mae_x_mask + mae_y
    return total_mae_x_mask, mae_x_mask, mae_y


def objective_function_MAE_y_mask(parameters, x_value, y_value, t_value, num_points): # X values are fixed
 """
    Objective function for Mean Absolute Error (MAE) with masked y data.

    Parameters:
    - parameters (array): Model parameters [alpha, beta, delta, gamma].
    - x_value (array): Observed x values.
    - y_value (array): Observed y values.
    - t_value (array): Time points.
    - num_points (int): Number of points to be removed.

    Returns:
    - Tuple: Total MAE, MAE for x, and MAE for y.
    """

    x_data_mask, y_data_mask, t_data_mask = random_choice(x_value, y_value, t_value, num_points) 


    initial_c = [x_value[0], y_value[0]]
    solution = odeint(system_equation, initial_c, t_value, args=(parameters,))

    model_predictions_x = solution[:, 0]

    _, mae_y_mask = objective_function_MAE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask)
    mae_x = np.mean(np.abs(model_predictions_x - x_value))
    total_mae_y_mask = mae_x + mae_y_mask

    return total_mae_y_mask, mae_x, mae_y_mask



def critical_points_analysis(observed_x, observed_y, time_points, parameters, fixed_time_series, num_points):
 """
    Analyze critical points using Mean Absolute Error (MAE) with fixed time series.

    Parameters:
    - observed_x (array): Observed x values.
    - observed_y (array): Observed y values.
    - time_points (array): Time points.
    - parameters (array): Model parameters [alpha, beta, delta, gamma].
    - fixed_time_series (str): Type of fixed time series ('x fixed', 'y fixed', 'none fixed').
    - num_points (int): Number of points to be removed.

    Returns:
    - float: Mean Absolute Error (MAE).
    """
       
    if fixed_time_series == "x fixed":
        mae, _, _ = objective_function_MAE_y_mask(parameters, observed_x, observed_y, time_points, num_points)
    
    elif fixed_time_series == "y fixed":
        mae, _, _ = objective_function_MAE_x_mask(parameters, observed_x, observed_y, time_points, i)
        
    elif fixed_time_series == "none fixed":
        _, _, mae_y_mask = objective_function_MAE_y_mask(parameters, observed_x, observed_y, time_points, num_points)
        _, mae_x_mask, _ = objective_function_MAE_x_mask(parameters, observed_x, observed_y, time_points, num_points)
        mae = mae_x_mask + mae_y_mask
        
    return mae


def simulated_annealing_critical_points(initial_guess, time_points, observed_data, iterations,
                                        temp, fixed_time_series, num_points, cooling_type):
                                         """
    Perform simulated annealing for critical points analysis.

    Parameters:
    - initial_guess (array): Initial guess for model parameters.
    - time_points (array): Time points.
    - observed_data (tuple): Tuple containing observed x and y values.
    - iterations (int): Number of iterations for simulated annealing.
    - temp (float): Initial temperature.
    - fixed_time_series (str): Type of fixed time series ('x fixed', 'y fixed', 'none fixed').
    - num_points (int): Number of points to be removed.
    - cooling_type (str): Type of cooling schedule ('Linear', 'Exp', 'Logarithmic').

    Returns:
    - array: Optimized parameters.
    """

    x_value, y_value = observed_data
    step_size = 0.1

    current_temperature = temp
    best = initial_guess
    best_eval = critical_points_analysis(x_value, y_value, time_points, best, fixed_time_series, num_points)

    curr, curr_eval = best, best_eval

    for i in range(iterations):
        candidate = curr + step_size * np.random.normal(size=len(curr))
            
        # Evaluate the objective function for the neighbor
        if fixed_time_series == "x fixed":
            candidate_eval = critical_points_analysis(observed_x, observed_y, time_points, candidate, fixed_time_series, num_points)
        elif fixed_time_series == "y fixed":
            candidate_eval = critical_points_analysis(observed_x, observed_y, time_points, candidate, fixed_time_series, num_points)
        elif fixed_time_series == "none fixed":
            candidate_eval = critical_points_analysis(observed_x, observed_y, time_points, candidate, fixed_time_series, num_points)

        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
        
        if cooling_type == "Linear":
            current_temperature = linear_cooling(current_temperature, 0.001)
        elif cooling_type == "Exp":
            current_temperature = exponential_cooling(current_temperature, 0.95)
        elif cooling_type == "Logarithmic":
            current_temperature = logarithmic_cooling(temp, i)

            
        diff = candidate_eval - curr_eval
    
        metropolis = np.exp(-diff / current_temperature)
        if diff < 0 or np.random.rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval
    return best

time_points, observed_x, observed_y = read_data()
given_data = [observed_x, observed_y]

points = len(time_points)
guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])




#Change to "x fixed" to fix x_data and variable y_data
#Change to "y fixed" to fix y_data and variable x_data
#Change to "none fixed" to not fix any data
fixed_time_series = 'none fixed'
cooling_type = 'Exp' # Other options: 'Linear' , 'Logarithmic'
temp = 5

mean_error = []
std_error = []
final_para = []

for i in range(points):
    mae_per_iter = []
    opt_param = []
    
    for _ in range(10):
        
        iterations = 50
        step_size = 0.1
        param = simulated_annealing_critical_points(guess, time_points, given_data, iterations, temp,
                                     fixed_time_series, i, cooling_type) # obtain optimized parameters after increasing number of points removed
        solution = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(param,))
        model_predictions_x = solution[:, 0]
        model_predictions_y = solution[:, 1]
        mae_x = np.mean(np.abs(model_predictions_x - observed_x))
        mae_y = np.mean(np.abs(model_predictions_y - observed_y))


        mae = mae_x + mae_y
        opt_param.append(param) 
        mae_per_iter.append(mae)

    final_para.append(np.mean(opt_param, axis=0))
    mean_error.append(np.mean(mae_per_iter))
    std_error.append(np.std(mae_per_iter))

final_para = np.array(final_para)
mean_error = np.array(mean_error)
std_error = np.array(std_error)




# plt.fill_between(range(1, points + 1), mean_error - std_error, mean_error + std_error, alpha=0.2, color="red")
# plt.plot(range(1, points + 1), mean_error, label=f"MAE: {fixed_time_series}")
# plt.xlabel("Number of points removed")
# plt.ylabel("Mean MAE")
# plt.title("Simulated Annealing")
# plt.legend()
# plt.show()

plt.errorbar(range(1, points + 1), mean_error, yerr=std_error, fmt='o-', label=f"MAE, Cooling type: {fixed_time_series, cooling_type}")
plt.xlabel("Number of points removed")
plt.ylabel("Mean MAE")
plt.title("Simulated Annealing")
plt.legend()
plt.show()





