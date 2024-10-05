import torch
import os
import sys
import numpy as np
import casadi as ca
import pickle
import matplotlib.pyplot as plt


# Add the folder containing your modules to the Python path
path_core = "..\\core"
path_casadi_converter = "..\\Casadi_modules"

module_path_core = os.path.abspath(path_core)
module_path_casadi_converter = os.path.abspath(path_casadi_converter)

if module_path_core not in sys.path:
    sys.path.append(module_path_core)

if module_path_casadi_converter not in sys.path:
    sys.path.append(module_path_casadi_converter)

from Density_model import DensityNN, density_get  

# Load the saved model parameters (state dict)
model_density = DensityNN()
model_density.load_state_dict(torch.load('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\density_nn_model_W10.pth',weights_only=True))

# Extracting weights and biases
weights = {}
for name, param in model_density.named_parameters():
    weights[name] = param.detach().numpy()

# Define CasADi softplus and ReLU activation functions
def softplus_casadi(x):
    return ca.log(1 + ca.exp(x))

def relu_casadi(x):
    return ca.fmax(x, 0)

# Define the CasADi model to mimic the PyTorch model
def casadi_density_nn(input_vector, weights):
    x = input_vector
    W_fc1 = weights['fc1.weight']
    b_fc1 = weights['fc1.bias']
    x = relu_casadi(ca.mtimes(W_fc1, x) + b_fc1)

    W_fc2 = weights['fc2.weight']
    b_fc2 = weights['fc2.bias']
    x = relu_casadi(ca.mtimes(W_fc2, x) + b_fc2)

    W_fc3 = weights['fc3.weight']
    b_fc3 = weights['fc3.bias']
    x = relu_casadi(ca.mtimes(W_fc3, x) + b_fc3)

    W_fc4 = weights['fc4.weight']
    b_fc4 = weights['fc4.bias']
    x = ca.mtimes(W_fc4, x) + b_fc4

    # Apply softplus to ensure non-negative outputs
    x = softplus_casadi(x)
    return x

# Example usage: Create CasADi input vector and run the model
input_vector = ca.MX.sym('input_vector', 3, 1)  # Input shape (3, 1)
output_vector = casadi_density_nn(input_vector, weights)

# Create a CasADi function to compute the model's output
density_nn_function = ca.Function('density_nn', [input_vector], [output_vector])

# Load the MinMaxScaler for input data
with open('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\scaler_W10.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the MinMaxScaler for target data
with open("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\target_scaler_W10.pkl", 'rb') as f:
    target_scaler = pickle.load(f)

# Extract the min and max values from the scalers
scaler_min = np.array(scaler.data_min_)
scaler_max = np.array(scaler.data_max_)
target_scaler_min = np.array(target_scaler.data_min_)
target_scaler_max = np.array(target_scaler.data_max_)

# Function to normalize inputs using MinMaxScaler
def apply_scaler_casadi(input_vector, scaler_min, scaler_max):
    return (input_vector - scaler_min) / (scaler_max - scaler_min)

# Function to inverse-transform the outputs back to original scale
def inverse_transform_casadi(output_vector, target_scaler_min, target_scaler_max):
    return output_vector * (target_scaler_max - target_scaler_min) + target_scaler_min

# Normalize input
scaled_input = apply_scaler_casadi(input_vector, ca.MX(scaler_min), ca.MX(scaler_max))

# Neural network output using CasADi
nn_output = casadi_density_nn(scaled_input, weights)

# Apply inverse transform to the neural network output
scaled_output = inverse_transform_casadi(nn_output, ca.MX(target_scaler_min), ca.MX(target_scaler_max))

# Define the CasADi function to represent the entire process
density_get_casadi = ca.Function('density_get_casadi', [input_vector], [scaled_output])

if __name__ == "__main__":

    # Example input values (altitude, u, inclination)
    test_inputs = [
        [200, 0, 0.707],
        [220, -1, 0.8],
        [230, -1, 0.8],
        [250, 0.1, 0.5],
        [300, -0.2, 0.9],
        [350, 0.3, 0.7],
        [400, -0.5, 0.8],
        [400, 0.5, 0.6],
        [450, -0.7, 0.9],
        [450, 1, 0.1] 
    ]

    pytorch_density = []
    casadi_density = []

    pytorch_molar_mass = []
    casadi_molar_mass = []

    pytorch_temperature = []
    casadi_temperature = []

    # Compare outputs for each input combination
    for test_input in test_inputs:
        # PyTorch result
        density, molar_mass, temperature = density_get(test_input[0], test_input[1], test_input[2], model_density, scaler, target_scaler)
        pytorch_density.append(density)
        pytorch_molar_mass.append(molar_mass)
        pytorch_temperature.append(temperature)
        
        # CasADi result
        test_input_casadi = ca.DM(np.array(test_input).reshape(3, 1))
        casadi_output = density_get_casadi(test_input_casadi).full().flatten()
        casadi_density.append(casadi_output[0])
        casadi_molar_mass.append(casadi_output[1])
        casadi_temperature.append(casadi_output[2])

    # Convert input altitudes for plotting
    altitudes = [input_[0] for input_ in test_inputs]

    # Plotting the results
    plt.figure(figsize=(12, 8))

    # Plot for Density
    plt.subplot(3, 1, 1)
    plt.plot(altitudes, pytorch_density, label='PyTorch Density', marker='o')
    plt.plot(altitudes, casadi_density, label='CasADi Density', marker='x')
    plt.title('Density Comparison')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Density (kg/m^3)')
    plt.legend()

    # Plot for Molar Mass
    plt.subplot(3, 1, 2)
    plt.plot(altitudes, pytorch_molar_mass, label='PyTorch Molar Mass', marker='o')
    plt.plot(altitudes, casadi_molar_mass, label='CasADi Molar Mass', marker='x')
    plt.title('Molar Mass Comparison')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Molar Mass (g/mol)')
    plt.legend()

    # Plot for Temperature
    plt.subplot(3, 1, 3)
    plt.plot(altitudes, pytorch_temperature, label='PyTorch Temperature', marker='o')
    plt.plot(altitudes, casadi_temperature, label='CasADi Temperature', marker='x')
    plt.title('Temperature Comparison')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Temperature (K)')
    plt.legend()

    plt.tight_layout()
    plt.show()
