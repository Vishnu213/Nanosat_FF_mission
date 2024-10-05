import pytest
import numpy as np
import torch
import os
import sys

from casadi import SX, Function
from L4CasADi import NeuralNetwork  # Assuming L4CasADi provides this for the CasADi neural network



# Add the folder containing your modules to the Python path
path_core = "..\\core"
path_casadi_converter = "..\\Casadi_modules"

# Get absolute paths
module_path_core = os.path.abspath(path_core)
module_path_casadi_converter = os.path.abspath(path_casadi_converter)

# Check if the paths are not already in sys.path and add them
if module_path_core not in sys.path:
    sys.path.append(module_path_core)

if module_path_casadi_converter not in sys.path:
    sys.path.append(module_path_casadi_converter)

from Density_model import DensityNN, density_get  # Assuming this is where the model functions are
from density_model_casadi import DensityNN, density_get, density_get_casadi, DensityNNCasadi  # Assuming this is where the model functions are

# Load the original PyTorch model and scalers
model_density = DensityNN()
model_density.load_state_dict(torch.load('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\density_nn_model_W10.pth'))
model_density.eval()

with open('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\scaler_W10.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\target_scaler_W10.pkl", 'rb') as f:
    target_scaler = pickle.load(f)

# Example input data for testing
altitude_test = 200
u_test = 0
i_test = 0.707

# Original PyTorch-based function test
def test_density_get_original():
    density, molar_mass, temperature = density_get(altitude_test, u_test, i_test, model_density, scaler, target_scaler)
    assert isinstance(density, float)
    assert isinstance(molar_mass, float)
    assert isinstance(temperature, float)
    print(f"Original model: Density={density}, Molar mass={molar_mass}, Temperature={temperature}")

# Testing the CasADi-based model
def test_density_get_casadi():
    # Prepare symbolic input for CasADi
    altitude_sym = SX.sym('altitude_sym')
    u_sym = SX.sym('u_sym')
    i_sym = SX.sym('i_sym')

    # Define symbolic function using density_get_casadi
    model_density_casadi = DensityNNCasadi()  # Create CasADi-based neural network
    density, molar_mass, temperature = density_get_casadi(altitude_sym, u_sym, i_sym, model_density_casadi, scaler.data_min_, scaler.data_max_, target_scaler.data_min_, target_scaler.data_max_)

    # Create CasADi function for numerical testing
    density_func = Function('density_func', [altitude_sym, u_sym, i_sym], [density, molar_mass, temperature])

    # Evaluate the CasADi function at test inputs
    density_casadi_result = density_func(altitude_test, u_test, i_test)

    density_casadi = density_casadi_result[0]
    molar_mass_casadi = density_casadi_result[1]
    temperature_casadi = density_casadi_result[2]

    # Check that outputs are numeric
    assert isinstance(density_casadi, np.ndarray)
    assert isinstance(molar_mass_casadi, np.ndarray)
    assert isinstance(temperature_casadi, np.ndarray)

    print(f"CasADi model: Density={density_casadi}, Molar mass={molar_mass_casadi}, Temperature={temperature_casadi}")

# Compare the two models
def test_density_model_comparison():
    # Get results from original PyTorch-based model
    density_orig, molar_mass_orig, temperature_orig = density_get(altitude_test, u_test, i_test, model_density, scaler, target_scaler)

    # Get results from CasADi-based model
    altitude_sym = SX.sym('altitude_sym')
    u_sym = SX.sym('u_sym')
    i_sym = SX.sym('i_sym')

    model_density_casadi = DensityNNCasadi()
    density, molar_mass, temperature = density_get_casadi(altitude_sym, u_sym, i_sym, model_density_casadi, scaler.data_min_, scaler.data_max_, target_scaler.data_min_, target_scaler.data_max_)
    density_func = Function('density_func', [altitude_sym, u_sym, i_sym], [density, molar_mass, temperature])
    density_casadi_result = density_func(altitude_test, u_test, i_test)
    
    density_casadi = density_casadi_result[0]
    molar_mass_casadi = density_casadi_result[1]
    temperature_casadi = density_casadi_result[2]

    # Compare the results within a tolerance
    assert np.isclose(density_orig, density_casadi, atol=1e-5), f"Density mismatch: {density_orig} vs {density_casadi}"
    assert np.isclose(molar_mass_orig, molar_mass_casadi, atol=1e-5), f"Molar mass mismatch: {molar_mass_orig} vs {molar_mass_casadi}"
    assert np.isclose(temperature_orig, temperature_casadi, atol=1e-5), f"Temperature mismatch: {temperature_orig} vs {temperature_casadi}"

    print("Original model and CasADi model results match within tolerance.")
