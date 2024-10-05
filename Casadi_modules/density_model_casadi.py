import casadi as ca
from L4CasADi import NeuralNetwork  # Assuming L4CasADi provides a NeuralNetwork class
import numpy as np
import pickle


# Placeholder for scaler and target scaler transforms (these need to be represented symbolically in CasADi)
def apply_scaler_casadi(input_vector, scaler_min, scaler_max):
    return (input_vector - scaler_min) / (scaler_max - scaler_min)

def inverse_transform_casadi(output_vector, target_scaler_min, target_scaler_max):
    return output_vector * (target_scaler_max - target_scaler_min) + target_scaler_min

# Define a symbolic neural network using CasADi
class DensityNNCasadi:
    def __init__(self, input_size=3, output_size=3):
        # Load the converted CasADi neural network from the PyTorch model using L4CasADi
        self.nn_model = NeuralNetwork("density_nn_model_W10.pth", input_size, output_size)

    def forward(self, x):
        return self.nn_model(x)

# Symbolic density_get function in CasADi
def density_get_casadi(altitude, u, i, model, scaler_min, scaler_max, target_scaler_min, target_scaler_max):
    # Create symbolic variables
    altitude_sym = ca.SX.sym('altitude_sym')
    u_sym = ca.SX.sym('u_sym')
    i_sym = ca.SX.sym('i_sym')

    # Input symbolic vector
    nn_input_raw = ca.vertcat(altitude_sym, u_sym, i_sym)
    
    # Apply the scaling transformation
    nn_input_scaled = apply_scaler_casadi(nn_input_raw, scaler_min, scaler_max)
    
    # Predict density, molar mass, and temperature using the CasADi neural network
    nn_output_scaled = model.forward(nn_input_scaled)
    
    # Apply the inverse scaling transformation
    nn_output = inverse_transform_casadi(nn_output_scaled, target_scaler_min, target_scaler_max)
    
    # Return the symbolic results for density, molar mass, and temperature
    density = nn_output[0]
    molar_mass = nn_output[1]
    temperature = nn_output[2]
    
    return density, molar_mass, temperature


# Load the MinMaxScaler for input data
with open('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\scaler_W10.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the MinMaxScaler for target data
with open("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\target_scaler_W10.pkl", 'rb') as f:
    target_scaler = pickle.load(f)

# Extract min and max values for input scaler
scaler_min = scaler.data_min_
scaler_max = scaler.data_max_

# Extract min and max values for target scaler
target_scaler_min = target_scaler.data_min_
target_scaler_max = target_scaler.data_max_

# Convert to numpy arrays
scaler_min = np.array(scaler_min)
scaler_max = np.array(scaler_max)
target_scaler_min = np.array(target_scaler_min)
target_scaler_max = np.array(target_scaler_max)

# Create the CasADi neural network model
model_density_casadi = DensityNNCasadi()

# Example usage with symbolic inputs
density, molar_mass, temperature = density_get_casadi(200, 0, 0.707, model_density_casadi, scaler_min, scaler_max, target_scaler_min, target_scaler_max)
