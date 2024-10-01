"""
Nanosat Formation Flying Project

Testing the core libraries : Testing the absolute near non singular orbital elements dynamics

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
## ADD the packages here if you think it is needed and update it in this file.

## Import our libraries here
Library= os.path.join(os.path.dirname(os.path.abspath(__file__)),"../core")
sys.path.insert(0, Library)

from TwoBP import car2kep, kep2car, twobp_cart, gauss_eqn, Event_COE, theta2M, guess_nonsingular, M2theta, Param2NROE, guess_nonsingular_Bmat, lagrage_J2_diff, absolute_NSROE_dynamics ,NSROE2car




def compare_density_nn_nrlmsise_over_days(alt, u,i, model, scaler, target_scaler):
    """
    Compares the density, molar mass, and temperature from NRLMSISE-00 and Neural Network over multiple days.

    Parameters:
    - altitude (float): Altitude in kilometers.
    - argument_of_latitude (float): Argument of latitude in radians.
    - inclination (float): Inclination in radians.
    - model (nn.Module): Loaded PyTorch neural network model for density prediction.
    - scaler (MinMaxScaler): Scaler for normalizing the input data.
    - target_scaler (MinMaxScaler): Scaler for normalizing the target data (density, molar mass, temperature).
    
    Returns:
    - dates (list): List of datetime objects for the comparison dates.
    - density_ref_100_days (list): List of reference densities from NRLMSISE-00.
    - molar_mass_ref_100_days (list): List of reference molar masses from NRLMSISE-00.
    - temp_ref_100_days (list): List of reference temperatures from NRLMSISE-00.
    - density_nn_100_days (list): List of predicted densities from the neural network.
    - molar_mass_nn_100_days (list): List of predicted molar masses from the neural network.
    - temp_nn_100_days (list): List of predicted temperatures from the neural network.
    """
    # Initialize lists to store data for 100 days
        # Normalize inputs before passing them to the neural network
    nn_input_raw = np.array([[alt, u, i]])
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

# Neural network class (simplified)
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


# Function to get density from NRLMSISE-00
def get_density_from_nrlmsise00(alt, lat, lon, date):
    doy = date.timetuple().tm_yday
    sec = date.hour * 3600 + date.minute * 60 + date.second
    densities, t = msise_model(
        time=date, 
        alt=alt, 
        lat=lat,  
        lon=lon,  
        f107a=150.0,  
        f107=150.0,  
        ap=4.0,  
    )

    # Molar masses of different gases (in g/mol)
    molar_masses = {
    'He': 4.0026,  # Helium
    'O': 16.0,     # Oxygen atom
    'N2': 28.0134, # Nitrogen molecule
    'O2': 32.0,    # Oxygen molecule
    'Ar': 39.948,  # Argon
    'H': 1.008,    # Hydrogen atom
    }


    # Densities (in g/cm³)
    density_He = densities[0]
    density_O = densities[1]
    density_N2 = densities[2]
    density_O2 = densities[3]
    density_Ar = densities[4]
    density_H = densities[6]

    # Calculate total number density
    total_density = density_He + density_O + density_N2 + density_O2 + density_Ar + density_H

    # Calculate mean molar mass in kg/mol
    mean_molar_mass = (density_He * molar_masses['He'] + density_O * molar_masses['O'] + density_N2 * molar_masses['N2'] +
                        density_O2 * molar_masses['O2'] + density_Ar * molar_masses['Ar'] + density_H * molar_masses['H']) / total_density


    # Temperature at the altitude (in K)
    temp_alt =t[1]  # Temperature at the given altitude (not exospheric temperature)

    return densities[5], mean_molar_mass , temp_alt

def ecef_to_latlon(x, y, z):
    # Longitude calculation
    lon = np.arctan2(y, x)  # In radians

    # Latitude calculation (spherical Earth approximation)
    hyp = np.sqrt(x**2 + y**2)  # Hypotenuse in the xy-plane
    lat = np.arctan2(z, hyp)  # In radians

    # Convert to degrees
    lon = np.degrees(lon)
    lat = np.degrees(lat)

    return lat, lon

# Parameters that is of interest to the problem

data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
deg2rad = numpy.pi / 180

# CHECK Formation Establishment and Reconfiguration Using
# Differential Elements in J2-Perturbed Orbits and SENGUPTA
# Chaser spacecraft initial conditions
# orbital elements - non singular



# Deputy spacecraft relative orbital  elements/ LVLH initial conditions
# NOE_chief = numpy.array([a,lambda_0,i,q1,q2,omega])
NOE_chief = numpy.array([6600.1366,0,90*deg2rad,0.005,0,270.828*deg2rad]) # numpy.array([6803.1366,0,97.04,0.005,0,270.828])
## MAKE SURE TO FOLLOW RIGHT orbital elements order


# Design parameters for the formation - Sengupta and Vadali 2007 Relative Motion and the Geometry of Formations in Keplerian Elliptic Orbits

rho_1 = 500 # [m]  - radial separation 
rho_2 = 200 # [m]  - along-track separation
rho_3 = 300 # [m]  - cross-track separation
alpha = 0  # [rad] - angle between the radial and along-track separation
beta = numpy.pi/2 # [rad] - angle between the radial and cross-track separation
vd = -10 # Drift per revolutions m/resolution

parameters=numpy.array([rho_1,rho_2,rho_3,alpha,beta,vd])

# Initial relative orbital elements
RNOE_0=Param2NROE(NOE_chief, parameters,data)

print("RELATIVE ORBITAL ELEMTNS INITIAL", RNOE_0)
print("CHIEF INTIIAL ORBITAL ELEMENTS", NOE_chief)


# feed it into dynamical system to get the output
yy_o=NOE_chief
# test for gauess equation
mu=data["Primary"][0]
Torb = 2*numpy.pi*numpy.sqrt(NOE_chief[0]**3/mu)    # [s]    Orbital period
n_revol_T =1*24*60*60/Torb
n_revolution= 10 #n_revol_T
T_total=n_revolution*Torb

t_span=[0,T_total]
teval=numpy.linspace(0, T_total, 2000)
# K=numpy.array([k1,k2])

sol=integrate.solve_ivp(absolute_NSROE_dynamics, t_span, yy_o,t_eval=teval,
                        method='RK45',args=(data,),rtol=1e-13, atol=1e-10)


# Calculate rr_s and vv_s as before, but now add date tracking
initial_date = datetime(2024, 1, 1, 0, 0, 0)  # Replace with your actual start date
time_step = T_total / len(teval)  # Calculate the time step

rr_s = numpy.zeros((3, len(sol.y[0])))
vv_s = numpy.zeros((3, len(sol.y[0])))
lat_lon_alt_data = numpy.zeros((len(sol.y[0]), 3))  # Store lat, lon, alt for each step


# Convert from NROE to Carterian co-ordinates. 



rr_s=numpy.zeros((3,len(sol.y[0])))
vv_s=numpy.zeros((3,len(sol.y[0])))

density_nn_100_days = []
molar_mass_nn_100_days = []
temp_nn_100_days = []

den_ref = []
molar_mass_ref = []
temp_ref = []
u_temp = []



# Create a new instance of the model
model = DensityNN()

# Load the saved model parameters (state dict)
model.load_state_dict(torch.load('density_nn_model_W10.pth', weights_only=True))

# Set the model to evaluation mode if you are going to use it for inference
model.eval()

# Load the MinMaxScaler for input data
with open('scaler_W10.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the MinMaxScaler for target data
with open('target_scaler_W10.pkl', 'rb') as f:
    target_scaler = pickle.load(f)



for i in range(0,len(sol.y[0])):
    # if sol.y[5][i]>2*numpy.pi:
    #     sol.y[5][i]= 
    if sol.y[1][i]>2:
        print("lambda",sol.y[1][i])
    rr_s[:,i],vv_s[:,i]=NSROE2car(numpy.array([sol.y[0][i],sol.y[1][i],sol.y[2][i],
                                               sol.y[3][i],sol.y[4][i],sol.y[5][i]]),data)
    current_date = initial_date + timedelta(seconds=i * time_step)
    rr_ecef = numpy.zeros(3)


    # assigning the state variables
    a = sol.y[0][i]
    l = sol.y[1][i]
    inc = sol.y[2][i]
    q1 = sol.y[3][i]
    q2 = sol.y[4][i]
    OM = sol.y[5][i]



    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    eta = 1- q1**2 - q2**2
    p=term1
    rp=a*(1-e)
    n = numpy.sqrt(mu/(a**3))

    if e==0:  
        u = l
        r = (a * eta**2) / (1 + (q1 * numpy.cos(u)) + (q2 * numpy.sin(u)))
    else:
        omega_peri = numpy.arccos(q1 / e)
        mean_anamoly = l - omega_peri
        theta_tuple = M2theta(mean_anamoly, e, 1e-8)
        theta = theta_tuple[0]
        u = theta + omega_peri
        r = (a * eta**2) / (1 + (q1 * numpy.cos(u)) + (q2 * numpy.sin(u)))
    # print(rr_s[:,i])
    # print("current time",current_date)

    rr_ecef[0],rr_ecef[1],rr_ecef[2] = pm.eci2ecef(rr_s[0,i],rr_s[1,i],rr_s[2,i], current_date)
    # print("================", np.linalg.norm(rr_ecef),np.linalg.norm(rr_s[:,i]))
    # print(rr_s[:,i],vv_s[:,i])
    # print(sol.y[0][i],sol.y[1][i],sol.y[2][i], sol.y[3][i],sol.y[4][i],sol.y[5][i])

    # Get the latitude, longitude, and altitude
    # lat_lon_alt = pm.ecef2geodetic(rr_ecef[0], rr_ecef[1], rr_ecef[2])
    # lat = lat_lon_alt[0]
    # lon = lat_lon_alt[1]
    # alt = lat_lon_alt[2]
    lat, lon = ecef_to_latlon(rr_ecef[0], rr_ecef[1], rr_ecef[2])
    r_rad =numpy.linalg.norm(rr_s[:,i]) - data["Primary"][1]
    r_r = r - data["Primary"][1]
    err = r_rad - r_r


    # Get the density at the current location
    density, mean_molar_mass, temp = get_density_from_nrlmsise00(r_rad, lat, lon, current_date)
    # print("Density",density, "radius",r_rad, "error",err,"rr",r_r)
    # print("Molar Mass",mean_molar_mass)
    # print("Temperature",temp)
    den_ref.append(density)
    molar_mass_ref.append(mean_molar_mass)
    temp_ref.append(temp)
    u_temp.append(u)




    
    density_nn, molar_mass_nn, temp_nn = compare_density_nn_nrlmsise_over_days(r_rad, u,inc, model, scaler, target_scaler)
    #print(density_nn, molar_mass_nn, temp_nn)
    density_nn_100_days.append(density_nn)
    # print(density_nn_100_days)
    molar_mass_nn_100_days.append(molar_mass_nn)
    temp_nn_100_days.append(temp_nn)



# Example list with 10 elements
arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Slicing from the 6th element (index 5) to the last element
sliced_arr = arr[5:]

print(f"Original array: {arr}")
print(f"Sliced array (last 5 elements starting from index 5): {sliced_arr}")
# for i, density in enumerate(density_nn_100_days):
#     print(f"Index {i}: Shape {np.shape(density_nn_100_days)}")
print(len(density_nn_100_days))
print(len(den_ref))
sub_teval= teval
# Check the lengths before slicing
print(f"Length of teval: {len(teval)}")  # Should be 2000
print(f"Length of den_ref: {len(den_ref)}")  # Should be 1000
print(f"Length of density_nn_100_days: {len(density_nn_100_days)}")  # Should be 1000

# Slice teval to match the length of den_ref and density_nn_100_days
teval_sliced = teval[1000:2000]  # This will now have 1000 elements
print(f"Length of teval_sliced: {len(sub_teval)}")  # Should now be 1000



# Plotting the density comparison
plt.figure(figsize=(10, 6))
plt.plot(sub_teval, den_ref, label='NRLMSISE-00 Reference Density', color='b', marker='o')
plt.plot(sub_teval, density_nn_100_days, label='Neural Network Predicted Density', color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Density (kg/m³)')
plt.title('Density Comparison over 10 Days')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()




# Plotting the NRLMSISE-00 Reference Density
plt.figure(figsize=(10, 6))
plt.plot(teval, den_ref, label='NRLMSISE-00 Reference Density', color='b', marker='o')
plt.xlabel('Date')
plt.ylabel('Density (kg/m³)')
plt.title('NRLMSISE-00 Reference Density over 10 Days')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()


# Plotting the Neural Network Predicted Density
plt.figure(figsize=(10, 6))
plt.plot(teval, density_nn_100_days, label='Neural Network Predicted Density', color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Density (kg/m³)')
plt.title('Neural Network Predicted Density over 10 Days')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

    # h = COE[0]
    # e =COE[1]
    # i =COE[2]
    # OM = COE[3]
    # om =COE[4]
    # TA =COE[5]

    


# Spherical earth
# Setting up Spherical Earth to Plot
N = 50
phi = numpy.linspace(0, 2 * numpy.pi, N)
theta = numpy.linspace(0, numpy.pi, N)
theta, phi = numpy.meshgrid(theta, phi)

r_Earth = 6378.14  # Average radius of Earth [km]
X_Earth = r_Earth * numpy.cos(phi) * numpy.sin(theta)
Y_Earth = r_Earth * numpy.sin(phi) * numpy.sin(theta)
Z_Earth = r_Earth * numpy.cos(theta)

# draw the unit vectors of the ECI frame on the 3d plot of earth



# Plotting Earth and Orbit
fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)
# x-axis
ax.quiver(0, 0, 0, 1e4, 0, 0, color='r', label='X-axis')
# y-axis
ax.quiver(0, 0, 0, 0, 1e4, 0, color='g', label='Y-axis')
# z-axis
ax.quiver(0, 0, 0, 0, 0, 1e4, color='b', label='Z-axis')
# plotting
ax.plot3D(rr_s[0],rr_s[1],rr_s[2] , 'black', linewidth=2, alpha=1)

ax.set_title('two body trajectory')




fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, rr_s[0])
axs[0].set_title('x')

# Plot data on the second subplot
axs[1].plot(teval, rr_s[1])
axs[1].set_title('y')

axs[2].plot(teval, rr_s[2])
axs[2].set_title('z')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[0])
axs[0].set_title('semi major axis')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[1])
axs[1].set_title('mean true latitude')

axs[2].plot(teval, sol.y[2])
axs[2].set_title('inclination')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[3])
axs[0].set_title('q1')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[4])
axs[1].set_title('q2')

axs[2].plot(teval, sol.y[5])
axs[2].set_title('right ascenstion of ascending node')

plt.show()

