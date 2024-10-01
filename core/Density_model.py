"""
Nanosat Formation Flying Project

Density model for the nanosatellites in the formation flying project
The model is based on the NRLMSISE-00 model- fitted with a neural network model for the density, molar mass, and temperature of the atmosphere

Author:
    Vishnuvardhan Shakthibala
    
"""
## Copy the following lines of code 
# FROM HERE
import numpy
from scipy import integrate
import matplotlib.pyplot as plt
import os
import sys
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# from pyproj import Transformer  # For ECI to lat, lon, alt conversion (ECEF)
from nrlmsise00 import msise_model

import numpy as np
from nrlmsise00 import msise_model
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pymap3d as pm
import pickle
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta
import math

# from pyproj import Transformer  # For ECI to lat, lon, alt conversion (ECEF)



import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class DensityNN(nn.Module):
    def __init__(self):
        super(DensityNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)  # Output layer for 3 outputs: density, molar mass, temperature

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.softplus(x)  # Ensure non-negative outputs for density, molar mass, and temperature

def density_get(altitude, u,i, model, scaler, target_scaler):

    # Initialize lists to store data for 100 days
        # Normalize inputs before passing them to the neural network
    nn_input_raw = np.array([[altitude, u, i]])
    nn_input_scaled = scaler.transform(nn_input_raw)
    nn_input_tensor = torch.tensor(nn_input_scaled, dtype=torch.float32)

    # Use the neural network to predict density, molar mass, and temperature
    with torch.no_grad():
        nn_output_scaled = model(nn_input_tensor).numpy()[0]
        nn_output = target_scaler.inverse_transform([nn_output_scaled])



        # # Print the comparison for the current date
        # print(f"\nDate: {current_date.strftime('%Y-%m-%d')}")
        # print(f"  NRLMSISE-00 Density: {density_ref:.4e} kg/m³")
        # print(f"  NN Predicted Density: {nn_output[0][0]:.4e} kg/m³")
        # print(f"  NRLMSISE-00 Molar Mass: {molar_mass_ref:.4f} g/mol")
        # print(f"  NN Predicted Molar Mass: {nn_output[0][1]:.4f} g/mol")
        # print(f"  NRLMSISE-00 Temperature: {temp_ref:.2f} K")
        # print(f"  NN Predicted Temperature: {nn_output[0][2]:.2f} K")

    # Return all relevant data for further processing or visualization
    return nn_output[0][0], nn_output[0][1], nn_output[0][2]

# Create a new instance of the model
model_density = DensityNN()
# Load the saved model parameters (state dict)

model_density.load_state_dict(torch.load('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\density_nn_model_W10.pth', weights_only=True))

# Set the model to evaluation mode if you are going to use it for inference
model_density.eval()

# Load the MinMaxScaler for input data
with open('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\scaler_W10.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the MinMaxScaler for target data
with open("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\target_scaler_W10.pkl", 'rb') as f:
    target_scaler = pickle.load(f)


# a=density_get(200,0,0.707, model, scaler, target_scaler)
# print(a)