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
        mse, _, _ = objective_function_MSE_x_mask(parameters, observed_x, observed_y, time_points, num_points)
        mse_x_no_mask, mse_y_no_mask = objective_function_MSE_no_mask(parameters, observed_x, observed_y, time_points, num_points)
    
    elif fixed_time_series == "none fixed":
        _, _, mse_y_mask = objective_function_MSE_y_mask(parameters, observed_x, observed_y, time_points, num_points)
        _, mse_x_mask, _= objective_function_MSE_x_mask(parameters, observed_x, observed_y, time_points, num_points)
        mse = mse_x_mask + mse_y_mask
        mse_x_no_mask, mse_y_no_mask = objective_function_MSE_no_mask(parameters, observed_x, observed_y, time_points, num_points)
        
    
    return mse, mse_x_no_mask, mse_y_no_mask




def hill_climbing(initial_guess, time_points, observed_data, iterations, step_size, fixed_time_series, num_points):
    
    
    current_solution = initial_guess
    current_objective_MSE, _, _ = critical_points_analysis(observed_x, observed_y, time_points, current_solution, fixed_time_series, num_points)
        
    for _ in range(iterations):
        new_solution = current_solution + np.random.uniform(-step_size, step_size, size=len(current_solution))
            
        # Evaluate the objective function for the neighbor
        if fixed_time_series == "x fixed":
            neighbor_objective, _,  _ = critical_points_analysis(observed_x, observed_y, time_points, new_solution, fixed_time_series, num_points)
        elif fixed_time_series == "y fixed":
            neighbor_objective, _, _ = critical_points_analysis(observed_x, observed_y, time_points, new_solution, fixed_time_series, num_points)
        elif fixed_time_series == "none fixed":
            neighbor_objective, _, _ = critical_points_analysis(observed_x, observed_y, time_points, new_solution, fixed_time_series, num_points)

        if neighbor_objective < current_objective_MSE:
            current_objective_MSE = neighbor_objective
            current_solution = new_solution
            
    current_solution = new_solution
    current_objective_MSE = neighbor_objective
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
    mse_per_iter = []
    
    for _ in range(100):
        iterations = 1500
        step_size = 0.1
        given_data = [observed_x, observed_y]
        optimized_parameters = hill_climbing(guess, time_points, given_data, iterations, step_size, time_series, i) # obtain optimized parameters after increasing number of points removed
        
        solution = odeint(system_equation, [observed_x[0], observed_y[0]], time_points, args=(optimized_parameters,))
        model_predictions_x = solution[:, 0]
        model_predictions_y = solution[:, 1]
        mse_x = np.mean(((model_predictions_x - observed_x)**2).astype(np.float64))
        mse_y = np.mean(((model_predictions_y - observed_y)**2).astype(np.float64))
        total_mse = mse_x + mse_y
        mse_per_iter.append(total_mse)

    mean_error.append(np.mean(mse_per_iter))
    std_error.append(np.std(mse_per_iter))

mean_error = np.array(mean_error)
std_error = np.array(std_error)

plt.fill_between(range(1, points + 1), mean_error - std_error, mean_error + std_error, alpha=0.2, color="red")
plt.plot(range(1, points + 1), mean_error, label=f"MSE: {time_series}")
plt.xlabel("Number of points removed")
plt.ylabel("Mean MSE")
plt.title('Hill Climbing')
plt.legend()
plt.show()









# num_points = len(time_points)
# initial_guess = np.array([2.07697341, 1.31609389, 0.44702719, 0.97037843])
# # Analyze critical points for x_data fixed
# mse_x_critical, mse_final_no_mask_ = critical_points_analysis(observed_x, observed_y, time_points, initial_guess, fixed_time_series="x")

# # Analyze critical points for y_data fixed
# mse_y_critical, mse_final_no_mask_ = critical_points_analysis(observed_x, observed_y, time_points, initial_guess, fixed_time_series="y")

# # Analyze critical points for both x_data and y_data not fixed
# mse_both_critical, mse_final_no_mask_ = critical_points_analysis(observed_x, observed_y, time_points, initial_guess, fixed_time_series="none")

# #show mean and standard deviation of mse for each number of points removed
# mse_x_critical_mean = np.mean(mse_x_critical, axis=1)
# mse_x_critical_std = np.std(mse_x_critical, axis=1)

# mse_y_critical_mean = np.mean(mse_y_critical, axis=1)
# mse_y_critical_std = np.std(mse_y_critical, axis=1)

# mse_both_critical_mean = np.mean(mse_both_critical, axis=1)
# mse_both_critical_std = np.std(mse_both_critical, axis=1)


# x_y_no_mask_mean = np.mean(mse_final_no_mask_, axis=1)
# x_y_no_mask_std = np.std(mse_final_no_mask_, axis=1)



# plt.fill_between(range(1, num_points + 1), mse_x_critical_mean - mse_x_critical_std, mse_x_critical_mean + mse_x_critical_std, alpha=0.2, color="red")
# plt.fill_between(range(1, num_points + 1), x_y_no_mask_mean - x_y_no_mask_std, x_y_no_mask_mean + x_y_no_mask_std, alpha=0.2, color="blue")
# plt.plot(range(1, num_points + 1), mse_x_critical_mean, label="MSE_x fixed")
# plt.plot(range(1, num_points + 1), x_y_no_mask_mean, label="MSE_x-y fixed")
# plt.xlabel("Number of points removed")
# plt.ylabel("MSE")
# plt.legend()
# plt.show()


# plt.fill_between(range(1, num_points + 1), mse_y_critical_mean - mse_y_critical_std, mse_y_critical_mean + mse_y_critical_std, alpha=0.2, color="red")
# plt.fill_between(range(1, num_points + 1), x_y_no_mask_mean - x_y_no_mask_std, x_y_no_mask_mean + x_y_no_mask_std, alpha=0.2, color="blue")
# plt.plot(range(1, num_points + 1), mse_y_critical_mean, label="MSE_y fixed")
# plt.plot(range(1, num_points + 1), x_y_no_mask_mean, label="MSE_x-y fixed")
# plt.xlabel("Number of points removed")
# plt.ylabel("MSE")
# plt.legend()
# plt.show()

# plt.fill_between(range(1, num_points + 1), mse_both_critical_mean - mse_both_critical_std, mse_both_critical_mean + mse_both_critical_std, alpha=0.2, color="red")
# plt.fill_between(range(1, num_points + 1), x_y_no_mask_mean - x_y_no_mask_std, x_y_no_mask_mean + x_y_no_mask_std, alpha=0.2, color="blue")
# plt.plot(range(1, num_points + 1), mse_x_critical_mean, label="MSE_x-y both not fixed")
# plt.plot(range(1, num_points + 1), x_y_no_mask_mean, label="MSE_x-y fixed")
# plt.xlabel("Number of points removed")
# plt.ylabel("MSE")
# plt.legend()
# plt.show()


 
# # The goal is to find a critical point where removing more data points does not significantly affect the model's performance. Thus
# # a quantitative analysis (such as the derivative of the MSE curve)can be useful. If the MSE continues 
# # to decrease without stabilization, it might indicate that there's room for further investigation into the model, data, or problem formulation.


# # We take the derivative of the mean MSE curve to look for points where the derivative is close to zero, indicating stabilization

# derivative_mse_both_critical = np.gradient(mse_both_critical_mean)
# derivative_mse_x_critical = np.gradient(mse_x_critical_mean)
# derivative_mse_y_critical = np.gradient(mse_y_critical_mean)


# # Plot the mean MSE and its derivative

# plt.plot(range(1, num_points + 1), derivative_mse_both_critical, label="Derivative")
# plt.plot(range(1, num_points + 1), derivative_mse_x_critical, label="Derivative")
# plt.plot(range(1, num_points + 1), derivative_mse_y_critical, label="Derivative")
# plt.xlabel("Number of points removed")
# plt.ylabel("MSE / Derivative")
# plt.legend()
# plt.show()











