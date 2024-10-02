"""
Nanosat Formation Flying Project

Relative dynamics of two nanosatellites are defined here with J2 perturbation. Taken from paper: 
A planning tool for optimal three-dimensional formation flight maneuvers of satellites in VLEO using aerodynamic lift and drag via yaw angle deviations  
Traub, C., Fasoulas, S., and Herdrich, G. (2022). 

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec



## ADD the packages here if you think it is needed and update it in this file.

## Import our libraries here
Library= os.path.join(os.path.dirname(os.path.abspath(__file__)),"../core")
sys.path.insert(0, Library)

from TwoBP import (
    car2kep, 
    kep2car, 
    twobp_cart, 
    gauss_eqn, 
    Event_COE, 
    theta2M, 
    M2theta, 
    Param2NROE, 
    guess_nonsingular_Bmat, 
    lagrage_J2_diff, 
    NSROE2car,
    NSROE2LVLH,
    NSROE2LVLH_2)

from dynamics import Dynamics_N, yaw_dynamics_N, yaw_dynamics, absolute_NSROE_dynamics, Dynamics



# Parameters that is of interest to the problem

data = {
    "Primary": [3.98600433e5,6378.16,7.2921150e-5],
    "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients

    # Satellites data including chief and deputies
    "satellites": {
        "chief": {
            "mass": 300,         # Mass in kg
            "area": 2,           # Cross-sectional area in m^2
            "C_D": 0.9,          # Drag coefficient
        },
        "deputy_1": {
            "mass": 250,
            "area": 1.8,
            "C_D": 0.85,
        },
        "deputy_2": {
            "mass": 270,
            "area": 1.9,
            "C_D": 0.88,
        }
    },
    "N_deputies": 2,  # Number of deputies
    "sat": [1.2, 1.2,1.2],  # Moment of inertia for each satellite

}


deg2rad = numpy.pi / 180


# Deputy spacecraft relative orbital  elements/ LVLH initial conditions
# NOE_chief = numpy.array([a,lambda_0,i,q1,q2,omega])
NOE_chief = numpy.array([6500,0.1,63.45*deg2rad,0.005,0,270.828*deg2rad]) # numpy.array([6803.1366,0,97.04,0.005,0,270.828])
## MAKE SURE TO FOLLOW RIGHT orbital elements order
 

    # assigning the state variables
a =NOE_chief[0]
l =NOE_chief[1]
i =NOE_chief[2]
q1 =NOE_chief[3]
q2 =NOE_chief[4]
OM =NOE_chief[5]
mu = data["Primary"][0]


data["Init"] = [NOE_chief[4],NOE_chief[3], 0]


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




# Number of deputies (can be increased dynamically)
N_deputies = 2  # Set this to the number of deputies (can be increased)

# Initialize relative orbital elements for each deputy
deputy_params = []

# Initialize yaw states (1 yaw angle per spacecraft: chief + deputies)
yaw_c_d = numpy.zeros(N_deputies + 1)  # 1 yaw state per spacecraft (chief + deputies)

# For each deputy, create initial conditions
for i in range(N_deputies):
    rho_1 = 0  # [m] - radial separation 
    rho_3 = 0  # [m] - cross-track separation
    alpha = 0  # [rad] - angle between the radial and along-track separation
    beta = 0  # [rad] - angle between the radial and cross-track separation
    vd = 0  # Drift per revolution
    
    if i == 0:
        d = 0.2 # [m] - along track separation (changing for each deputy)
    else:
        d = - 0.2  # [m] - along track separation (changing for each deputy)
    
    rho_2 = (2*(eta**2) * d) /(3-eta**2) # [m]  - along-track separation
    print("RHO_2",rho_2)
    # Relative parameters for each deputy
    parameters = numpy.array([rho_1, rho_2, rho_3, alpha, beta, vd])
    deputy_params.append(parameters)
    

# Define the yaw angles for each spacecraft (chief + N deputies)
for i in range(N_deputies + 1):
    yaw_c_d[i] = 0  # Initialize yaw angle (can be customized for each spacecraft)

# Initialize the state vectors for each deputy relative to the chief
yy_o = [Param2NROE(NOE_chief, param, data) for param in deputy_params]
yy_o_flattened = numpy.concatenate([yy.flatten() for yy in yy_o])

# Combine with chief elements and yaw control for each deputy
yy_total = numpy.concatenate((yy_o_flattened, NOE_chief, yaw_c_d))

# Define simulation parameters
mu = data["Primary"][0]
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # Orbital period
n_revol_T = 24 * 60 * 60 / Torb  # Number of revolutions per day
n_revolution = 0.005 * 365 * n_revol_T  # Simulation for 2 years
T_total = 1#n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total, 2000)

# Add the number of deputies to data for passing to Dynamics
data["N_deputies"] = N_deputies

# Run the simulation using integrate.solve_ivp
sol = integrate.solve_ivp(
    Dynamics_N, t_span, yy_total, t_eval=teval,
    method='DOP853', args=(data,), rtol=1e-10, atol=1e-10
)

# Convert NROE to Cartesian coordinates for all deputies
rr_s = numpy.zeros((3, len(sol.y[0]), N_deputies))
vv_s = numpy.zeros((3, len(sol.y[0]), N_deputies))

for i in range(len(sol.y[0])):
    for d in range(N_deputies):
        start_idx = d * 6
        yy1 = sol.y[start_idx:start_idx + 6, i]
        yy2 = sol.y[N_deputies * 6:N_deputies * 6 + 6, i]
        rr_s[:, i, d] = NSROE2LVLH(yy1, yy2, data)



# for d in range(N_deputies):
#     plt.figure()
#     plt.plot(rr_s[0, :, d], rr_s[1, :, d], label=f'Deputy {d+1}')
#     plt.xlabel('x (km)')
#     plt.ylabel('y (km)')
#     plt.legend()
#     plt.title(f'Relative motion of Deputy {d+1} in LVLH frame')
#     plt.show()

# Define colors for the deputies
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create a figure and specify layout using GridSpec
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 3, width_ratios=[2.5, 1, 1])  # Control layout: LVLH takes more space

# Larger main 3D plot
ax1 = fig.add_subplot(gs[:, 0], projection='3d')  # Main 3D plot occupies two rows

# Create 3D axes for the zoomed-in view and reference frame
zoom_ax = fig.add_subplot(gs[0, 1], projection='3d')  # Top-right for zoomed-in 3D view
ref_ax = fig.add_subplot(gs[1, 1], projection='3d')   # Bottom-right for 3D reference frame


# Plot trajectories in the main LVLH 3D plot
for d in range(N_deputies):
    ax1.plot(rr_s[0, :, d], rr_s[1, :, d], rr_s[2, :, d], color=colors[d], label=f'Deputy {d+1}')

# Add Chief Satellite (assumed to be at the origin for simplicity)
ax1.scatter(0, 0, 0, color='r', label='Chief', s=50)

arrow_length = 0.1  # Adjust this factor to change the relative size of arrows
x_axis = numpy.array([arrow_length, 0, 0])
y_axis = numpy.array([0, arrow_length, 0])
z_axis = numpy.array([0, 0, arrow_length])

# x-axis
x_quiver = ax1.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1)
# y-axis
y_quiver = ax1.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1)
# z-axis
z_quiver = ax1.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1)



# Set the limits to 1 km for each axis
x_limits = [-0.5, 1.5]  # 1 km range centered at 0
y_limits = [-0.5, 1.5]  # 1 km range centered at 0
z_limits = [-0.5, 1.5]  # 1 km range centered at 0

# Apply the fixed limits (1 km range) to the interactive plot
ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)
ax1.set_zlim(z_limits)

# Set labels and title for the main 3D plot
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Y (km)')
ax1.set_zlabel('Z (km)')
ax1.set_title('3D View of LVLH Frame with Deputies')
ax1.legend()

# Function to update zoomed view based on interaction
def update_zoomed_view(event):
    if event.inaxes == ax1:
        elev, azim = ax1.elev, ax1.azim

        # Update the zoomed-in plot for each deputy
        zoom_ax.clear()
        for d in range(N_deputies):
            zoom_ax.plot(rr_s[0, :, d], rr_s[1, :, d], rr_s[2, :, d], color=colors[d], label=f'Deputy {d+1}')
            zoom_ax.scatter(rr_s[0, -1, d], rr_s[1, -1, d], rr_s[2, -1, d], color=colors[d], s=50)  # End marker

        zoom_ax.set_title('Zoomed-in 3D View of Deputies')
        zoom_ax.set_xlim([-0.05, 0.05])  # Adjust as needed for zoom
        zoom_ax.set_ylim([-0.05, 0.05])
        zoom_ax.set_zlim([-0.05, 0.05])
        zoom_ax.set_xlabel('X (km)')
        zoom_ax.set_ylabel('Y (km)')
        zoom_ax.set_zlabel('Z (km)')
        zoom_ax.legend()
        zoom_ax.grid(True)

        # Update the reference frame at the right (in 3D)
        ref_ax.clear()
        ref_ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X-axis')
        ref_ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y-axis')
        ref_ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z-axis')
        ref_ax.set_title("Reference Frame (3D)")
        ref_ax.set_xlim([-1, 1])
        ref_ax.set_ylim([-1, 1])
        ref_ax.set_zlim([-0.5, 0.5])
        ref_ax.grid(True)

        # Set view angles of reference frame based on the main 3D plot
        ref_ax.view_init(elev=elev, azim=azim)

        # Redraw the figure
        fig.canvas.draw_idle()

# Connect zoom function to figure
fig.canvas.mpl_connect('motion_notify_event', update_zoomed_view)

# Plot all figures simultaneously
plt.show()
