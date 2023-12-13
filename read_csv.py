import csv
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint




def read_data():
    
    # Define the file path
    csv_file_path = 'predator_prey_data.csv'

    # Empty lists to store data
    t_data = []
    x_data = []
    y_data = []

    # Read data from CSV file
    with open(csv_file_path, 'r') as csvfile:
        
        # Create a CSV reader
        csvreader = csv.reader(csvfile)
        
        # Skip the header row if it exists
        next(csvreader, None)
        
        # Read data row by row
        for row in csvreader:
            # Assuming column 1 is at index 0 and column 2 is at index 1
            t_data.append(float(row[1]))
            x_data.append(float(row[2]))
            y_data.append(float(row[3]))
    return t_data, x_data, y_data

def plot_given_data():
    t, x , y = read_data()
    plt.scatter(t,x, label="Predator")
    plt.scatter(t,y, label="Prey")
    plt.xlabel("Time")
    plt.ylabel("Population Size")
    plt.legend()
    plt.tight_layout()


