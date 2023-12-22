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


    

def objective_function_MAE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask):

    initial_c = [x_data_mask[0], y_data_mask[0]]
    solution = odeint(system_equation, initial_c, t_data_mask, args=(parameters,))
    
    #model predictions for x and y
    model_predictions_x_mask = solution[:, 0]
    model_predictions_y_mask = solution[:, 1]

    mae_x_mask = np.mean(np.abs(model_predictions_x_mask - x_data_mask))
    mae_y_mask = np.mean(np.abs(model_predictions_y_mask - y_data_mask))
    
    return mae_x_mask, mae_y_mask



def objective_function_MAE_no_mask(parameters, x_value, y_value, t_value, num_points):

    x_d, y_d, t_d = random_choice(x_value, y_value, t_value, num_points) 

    initial_c = [x_d[0], y_d[0]]
    solution = odeint(system_equation, initial_c, t_d, args=(parameters,))
    
    #model predictions for x and y
    model_predictions_x = solution[:, 0]
    model_predictions_y= solution[:, 1]

    mae_x_no_mask = np.mean(np.abs(model_predictions_x - x_d))
    mae_y_no_mask = np.mean(np.abs(model_predictions_y - y_d))
    
    return mae_x_no_mask, mae_y_no_mask


def objective_function_MAE_x_mask(parameters, x_value, y_value, t_value, num_points): # Y value fixed

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

    x_data_mask, y_data_mask, t_data_mask = random_choice(x_value, y_value, t_value, num_points) 


    initial_c = [x_value[0], y_value[0]]
    solution = odeint(system_equation, initial_c, t_value, args=(parameters,))

    model_predictions_x = solution[:, 0]

    _, mae_y_mask = objective_function_MAE_x_y_mask(parameters, x_data_mask, y_data_mask, t_data_mask)
    mae_x = np.mean(np.abs(model_predictions_x - x_value))
    total_mae_y_mask = mae_x + mae_y_mask

    return total_mae_y_mask, mae_x, mae_y_mask




def critical_points_analysis(observed_x, observed_y, time_points, parameters, fixed_time_series, num_points):

       
    if fixed_time_series == "x fixed":
        mae, _, _ = objective_function_MAE_y_mask(parameters, observed_x, observed_y, time_points, num_points)
        mae_x_no_mask, mae_y_no_mask = objective_function_MAE_no_mask(parameters, observed_x, observed_y, time_points, num_points)

    elif fixed_time_series == "y fixed":
        mae, _, _ = objective_function_MAE_x_mask(parameters, observed_x, observed_y, time_points, num_points)
        mae_x_no_mask, mae_y_no_mask = objective_function_MAE_no_mask(parameters, observed_x, observed_y, time_points, num_points)
    
    elif fixed_time_series == "none fixed":
        _, _, mae_y_mask = objective_function_MAE_y_mask(parameters, observed_x, observed_y, time_points, num_points)
        _, mae_x_mask, _= objective_function_MAE_x_mask(parameters, observed_x, observed_y, time_points, num_points)
        mae = mae_x_mask + mae_y_mask
        mae_x_no_mask, mae_y_no_mask = objective_function_MAE_no_mask(parameters, observed_x, observed_y, time_points, num_points)
        
    
    return mae, mae_x_no_mask, mae_y_no_mask




def hill_climbing(initial_guess, time_points, observed_data, iterations, step_size, fixed_time_series, num_points):
    
    
    current_solution = initial_guess
    current_objective_MAE, _, _ = critical_points_analysis(observed_x, observed_y, time_points, current_solution, fixed_time_series, num_points)
        
    for _ in range(iterations):
        new_solution = current_solution + np.random.uniform(-step_size, step_size, size=len(current_solution))
            
        # Evaluate the objective function for the neighbor
        if fixed_time_series == "x fixed":
            neighbor_objective, _,  _ = critical_points_analysis(observed_x, observed_y, time_points, new_solution, fixed_time_series, num_points)
        elif fixed_time_series == "y fixed":
            neighbor_objective, _, _ = critical_points_analysis(observed_x, observed_y, time_points, new_solution, fixed_time_series, num_points)
        elif fixed_time_series == "none fixed":
            neighbor_objective, _, _ = critical_points_analysis(observed_x, observed_y, time_points, new_solution, fixed_time_series, num_points)

        if neighbor_objective < current_objective_MAE:
            current_objective_MAE = neighbor_objective
            current_solution = new_solution
            
    current_solution = new_solution
    current_objective_MAE = neighbor_objective
    return current_solution

   


time_points, observed_x, observed_y = read_data()
points = len(time_points)
  
guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])

mean_error = []
std_error = []

#Change to "x fixed" to fix x_data and variable y_data
#Change to "y fixed" to fix y_data and variable x_data
#Change to "none fixed" to not fix any data
time_series = "x fixed"

for i in range(points):
    mae_per_iter = []
    
    for _ in range(20):
        iterations = 100
        step_size = 0.1
        given_data = [observed_x, observed_y]
        optimized_parameters = hill_climbing(guess, time_points, given_data, iterations, step_size, time_series, i) # obtain optimized parameters after increasing number of points removed
        
        solution = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(optimized_parameters,))
        model_predictions_x = solution[:, 0]
        model_predictions_y = solution[:, 1]
        mae_x = np.mean(np.abs(model_predictions_x - observed_x))
        mae_y = np.mean(np.abs(model_predictions_y - observed_y))
        total_mae = mae_x + mae_y
        mae_per_iter.append(total_mae)

    mean_error.append(np.mean(mae_per_iter))
    std_error.append(np.std(mae_per_iter))

mean_error = np.array(mean_error)
std_error = np.array(std_error)

# plt.fill_between(range(1, points + 1), mean_error - std_error, mean_error + std_error, alpha=0.2, color="red")
# plt.plot(range(1, points + 1), mean_error, label=f"MAE: {time_series}")
# plt.xlabel("Number of points removed")
# plt.ylabel("Mean MAE")
# plt.title('Hill Climbing')
# plt.legend()
# plt.show()


plt.errorbar(range(1, points + 1), mean_error, yerr=std_error, fmt='o-', label=f"MAE: {time_series}", color='red')
plt.xlabel("Number of points removed")
plt.ylabel("Mean MAE")
plt.title("Hill Climbing")
plt.legend()
plt.show()










