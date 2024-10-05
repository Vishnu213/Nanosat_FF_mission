import torch

import os
import sys
import numpy as np
import casadi as ca
import pickle

# Add the folder containing your modules to the Python path
path_core = "..\\core"
path_casadi_converter = "..\\Casadi_modules"

module_path_core = os.path.abspath(path_core)
module_path_casadi_converter = os.path.abspath(path_casadi_converter)

if module_path_core not in sys.path:
    sys.path.append(module_path_core)

if module_path_casadi_converter not in sys.path:
    sys.path.append(module_path_casadi_converter)

from Density_model import DensityNN

# Load the saved model parameters (state dict)
model_density = DensityNN()
model_density.load_state_dict(torch.load('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\density_nn_model_W10.pth'))

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

# Example input values (altitude, u, inclination)
test_input = np.array([[350], [0], [0.707]])  # Shape (3, 1)

# Apply the CasADi function
test_input_casadi = ca.DM(test_input)
casadi_output = density_get_casadi(test_input_casadi)

# PyTorch equivalent model output
nn_input_scaled = scaler.transform(test_input.T)
nn_input_tensor = torch.tensor(nn_input_scaled, dtype=torch.float32)
with torch.no_grad():
    pytorch_output = model_density(nn_input_tensor).numpy()[0]

# Print outputs from both models
print("PyTorch model output:", pytorch_output)
print("CasADi model output:", casadi_output.full())
