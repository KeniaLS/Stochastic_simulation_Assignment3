from read_csv import read_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from helper_functions import *

from scipy.special import expit





def system_equation(var, t, parameters):
 
    x = var[0]
    y = var[1]
    alpha, beta, delta, gamma = parameters
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    
    return dx, dy


# Function to remove data points from a time series
def remove_data_points(data, indices_to_remove):
    data_removed = np.copy(data)
    data_removed_per_index = np.delete(data_removed, indices_to_remove, axis=0)

    return data_removed_per_index


def random_choice(observed_x, observed_y, time_points, num_points): #remove one point at a time

    rand_choice = np.random.choice(len(time_points), size=num_points)

    observed_x_removed = remove_data_points(observed_x, rand_choice)
    observed_y_removed = remove_data_points(observed_y, rand_choice)
    time_points_removed = remove_data_points(time_points, rand_choice)

    return observed_x_removed, observed_y_removed, time_points_removed


    

def objective_function_MSE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask):

    initial_c = [x_data_mask[0], y_data_mask[0]]
    solution = odeint(system_equation, initial_c, t_data_mask, args=(parameters,))
    
    #model predictions for x and y
    model_predictions_x_mask = solution[:, 0]
    model_predictions_y_mask = solution[:, 1]

    mse_x_mask = np.mean(((model_predictions_x_mask - x_data_mask)**2).astype(np.float64))
    mse_y_mask = np.mean(((model_predictions_y_mask - y_data_mask)**2).astype(np.float64))
    
    return mse_x_mask, mse_y_mask



def objective_function_MSE_no_mask(parameters, x_value, y_value, t_value, num_points):

    x_d, y_d, t_d = random_choice(x_value, y_value, t_value, num_points) 

    initial_c = [x_d[0], y_d[0]]
    solution = odeint(system_equation, initial_c, t_d, args=(parameters,))
    
    #model predictions for x and y
    model_predictions_x = solution[:, 0]
    model_predictions_y= solution[:, 1]

    mse_x_no_mask = np.mean(((model_predictions_x - x_d)**2).astype(np.float64))
    mse_y_no_mask = np.mean(((model_predictions_y - y_d)**2).astype(np.float64))
    
    return mse_x_no_mask, mse_y_no_mask


def objective_function_MSE_x_mask(parameters, x_value, y_value, t_value, num_points): # Y value fixed

    x_data_mask, y_data_mask, t_data_mask = random_choice(x_value, y_value, t_value, num_points)   


    # Simulate the model with the given parameters
    initial_c = [x_value[0], y_value[0]]
    solution = odeint(system_equation, initial_c, t_value, args=(parameters,))
    
    model_predictions_y = solution[:, 1]


    mse_x_mask, _ = objective_function_MSE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask)
    mse_y = np.mean(((model_predictions_y - y_value)**2).astype(np.float64))
    total_mse_x_mask = mse_x_mask + mse_y

    return total_mse_x_mask, mse_x_mask, mse_y


def objective_function_MSE_y_mask(parameters, x_value, y_value, t_value, num_points): # X values are fixed

    x_data_mask, y_data_mask, t_data_mask = random_choice(x_value, y_value, t_value, num_points) 


    initial_c = [x_value[0], y_value[0]]
    solution = odeint(system_equation, initial_c, t_value, args=(parameters,))

    model_predictions_x = solution[:, 0]

    _, mse_y_mask = objective_function_MSE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask)
    mse_x = np.mean(((model_predictions_x - x_value)**2).astype(np.float64))
    total_mse_y_mask = mse_x + mse_y_mask

    return total_mse_y_mask, mse_x, mse_y_mask




def critical_points_analysis(observed_x, observed_y, time_points, parameters, fixed_time_series, num_points):

       
    if fixed_time_series == "x fixed":
        mse, _, _ = objective_function_MSE_y_mask(parameters, observed_x, observed_y, time_points, num_points)
        mse_x_no_mask, mse_y_no_mask = objective_function_MSE_no_mask(parameters, observed_x, observed_y, time_points, num_points)

    elif fixed_time_series == "y fixed":
        mse, _, _ = objective_function_MSE_x_mask(parameters, observed_x, observed_y, time_points, i)
        mse_x_no_mask, mse_y_no_mask = objective_function_MSE_no_mask(parameters, observed_x, observed_y, time_points, num_points)
    
    elif fixed_time_series == "none fixed":
        _, _, mse_y_mask = objective_function_MSE_y_mask(parameters, observed_x, observed_y, time_points, num_points)
        _, mse_x_mask, _= objective_function_MSE_x_mask(parameters, observed_x, observed_y, time_points, num_points)
        mse = mse_x_mask + mse_y_mask
        mse_x_no_mask, mse_y_no_mask = objective_function_MSE_no_mask(parameters, observed_x, observed_y, time_points, num_points)
        
    
    return mse, mse_x_no_mask, mse_y_no_mask


def simulated_annealing_critical_points(initial_guess, time_points, observed_data, iterations, temp, 
                                         fixed_time_series, num_points, cooling_type):
    
    x_value, y_value = observed_data
    step_size = 0.1
    current_temperature = temp

    best = initial_guess
    best_eval, mse_x_no_mask, mse_y_no_mask = critical_points_analysis(x_value, y_value, time_points, best, fixed_time_series, num_points)

    curr, curr_eval = best, best_eval

    for i in range(iterations):
        candidate = curr + step_size * np.random.normal(size=len(curr))
            
        # Evaluate the objective function for the neighbor
        if fixed_time_series == "x fixed":
            candidate_eval, _,  _ = critical_points_analysis(observed_x, observed_y, time_points, candidate, fixed_time_series, num_points)
        elif fixed_time_series == "y fixed":
            candidate_eval, _, _ = critical_points_analysis(observed_x, observed_y, time_points, candidate, fixed_time_series, num_points)
        elif fixed_time_series == "none fixed":
            candidate_eval, _, _ = critical_points_analysis(observed_x, observed_y, time_points, candidate, fixed_time_series, num_points)

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
fixed_time_series = 'x fixed'
cooling_type = 'Exp'  # Other options: 'Linear' , 'Logarithmic'
temp = 5

mean_error = []
std_error = []
final_para = []

for i in range(points):
    mse_per_iter = []
    opt_param = []
    
    for _ in range(10):
        
        iterations = 50
        step_size = 0.1
        param = simulated_annealing_critical_points(guess, time_points, given_data, iterations, temp, 
                                     fixed_time_series, i, cooling_type) # obtain optimized parameters after increasing number of points removed
    
        
        solution = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(param,))
        model_predictions_x = solution[:, 0]
        model_predictions_y = solution[:, 1]
        mse_x = np.mean(((model_predictions_x - observed_x)**2).astype(np.float64))
        mse_y = np.mean(((model_predictions_y - observed_y)**2).astype(np.float64))

        mse = mse_x + mse_y
        opt_param.append(param) 
        mse_per_iter.append(mse)

    final_para.append(np.mean(opt_param, axis=0))
    mean_error.append(np.mean(mse_per_iter))
    std_error.append(np.std(mse_per_iter))

final_para = np.array(final_para)
mean_error = np.array(mean_error)
std_error = np.array(std_error)




# plt.fill_between(range(1, points + 1), mean_error - std_error, mean_error + std_error, alpha=0.2, color="red")
# plt.plot(range(1, points + 1), mean_error, label=f"MSE: {fixed_time_series}")
# plt.xlabel("Number of points removed")
# plt.ylabel("Mean MSE")
# plt.title("Simulated Annealing")
# plt.legend()
# plt.show()


plt.errorbar(range(1, points + 1), mean_error, yerr=std_error, fmt='o-', label=f"MSE, Cooling type: {fixed_time_series, cooling_type}")
plt.xlabel("Number of points removed")
plt.ylabel("Mean MSE")
plt.title("Simulated Annealing")
plt.legend()
plt.show()




