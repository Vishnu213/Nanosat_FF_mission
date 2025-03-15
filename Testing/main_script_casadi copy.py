import casadi as ca
import numpy
import matplotlib.pyplot as plt
import time
import os
import sys
import pickle
import pytest



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


from test_converted_functions import test_Dynamics_casadi
# Load the CasADi versions of the functions you've converted
# from converted_functions import Dynamics_casadi, NSROE2LVLH_casadi
# from converted_functions import loaded_polynomials
from constrains import con_chief_deputy_angle
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import pickle

# Import the CasADi versions of the functions you've converted
from converted_functions_original import Dynamics_casadi, NSROE2LVLH_casadi, con_chief_deputy_angle_casadi
from TwoBP import Param2NROE, M2theta


deg2rad = np.pi / 180
# Parameters (same as the Python version)
# Parameters (same as th
# e Python version)
param = {
    "Primary": [3.98600433e5, 6378.16, 7.2921150e-5],
    "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients
    "satellites": {
        "chief": {
            "mass": 5,
            "area": 0.05,
            "C_D": 0.9
        },
        "deputy_1": {
            "mass": 5,
            "area": 0.05,
            "C_D": 0.85
        }
    },
    "N_deputies": 2,
    "sat": [0.0412, 0.0412, 1.2],  # Moments of inertia
    "T_MAX": 23e-6,  # Maximum torque (Nm)
    "PHI_DOT": [0.1, 0.1],  # Limits for yaw rate (rad/s)
    "PHI": [-ca.pi / 2, ca.pi / 2],  # Limits for yaw angle (rad)
    "T_period": 2000.0  # Period of the sine wave
}

# Define CasADi symbolic variables
param_matrix = ca.MX.zeros(11, 3)  # 11 rows, 3 columns

# # Assign values row by row
# param_matrix[0, :] = [398600.433, 6378.16, 7.292115e-05]  # Primary
# param_matrix[1, :] = [0.001082626925638815, 0, 0]  # J2 effects
# param_matrix[2, :] = [5, 0.05,0.9]  # Chief Satellite
# param_matrix[3, :] = [5, 0.05, 0.85]  # Deputy Satellite
# param_matrix[4, :] = [2, 0, 0]  # Number of deputies
# param_matrix[5, :] = [0.0412, 0.0412, 1.2]  # Satellite design parameters
# param_matrix[6, :] = [2.3e-05, 0, 0]  # Max thrust
# param_matrix[7, :] = [0.1, 0.1, 0]  # PHI_DOT
# param_matrix[8, :] = [-ca.pi / 2, ca.pi / 2, 0]  # PHI constraints
# param_matrix[9, :] = [5580.5159, 0, 0]  # Orbital period
# param_matrix[10, :] = [0.004999999999999999, 0.008660254037844387, 0]  # Initial conditions



print("Parameters initialized.")

deg2rad = numpy.pi / 180

# CHECK Formation Establishment and Reconfiguration Using
# Differential Elements in J2-Perturbed Orbits and SENGUPTA
# Chaser spacecraft initial conditions
# orbital elements - non singular




e_init = 0.01
omega_init = 30
q1_init = e_init * numpy.cos(omega_init * deg2rad)
q2_init = e_init * numpy.sin(omega_init * deg2rad)
print("Q1_init",q1_init)
print("Q2_init",q2_init)


# Deputy spacecraft relative orbital elements/ LVLH initial conditions
NOE_chief = numpy.array([6800,0.1,90*deg2rad,q1_init,q2_init,10*deg2rad])
print("Chief initial orbital elements set.")

# Assigning the state variables
a = NOE_chief[0]
l = NOE_chief[1]
i = NOE_chief[2]
q1 = NOE_chief[3]
q2 = NOE_chief[4]
OM = NOE_chief[5]
mu = param["Primary"][0]

e = numpy.sqrt(q1**2 + q2**2)
h = numpy.sqrt(mu*a*(1-e**2))
term1 = (h**2)/(mu)
eta = ca.sqrt(1 - q1**2 - q2**2)
p = term1
rp = a*(1-e)
ra = a*(1+e)


n = numpy.sqrt(mu/(a**3))
print("State variables calculated.")
print("rp", rp)
print("ra", ra)
print("e---", e)
print("a---", (rp + ra) / 2)

if rp < 200+param["Primary"][1]:
    print("Satellite is inside the Earth's radius")
    exit()
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


delta_omega_sunsynchronous = (360 / (365.2421897*24*60*60)) * deg2rad
term1 = (-2 * (a**(7/2)) * delta_omega_sunsynchronous * (1-e**2)**2 )/(3*(param["Primary"][1]**2)*param["J"][0]*numpy.sqrt(mu))
inclination_sunsynchronous = numpy.arccos(term1)
i = inclination_sunsynchronous * 180 / numpy.pi

# setting suncrynchronous orbit inclination
NOE_chief[2] = inclination_sunsynchronous

print("Inclination of the sunsynchronous orbit",i)
print("ROE", NOE_chief)



# Design parameters for the formation - Sengupta and Vadali 2007 Relative Motion and the Geometry of Formations in Keplerian Elliptic Orbits

rho_1 = 0 # [m]  - radial separation
rho_3 =0 # [m]  - cross-track separation
alpha = 0*45* deg2rad#180 * deg2rad  # [rad] - angle between the radial and along-track separation
beta = alpha + 90 * deg2rad # [rad] - angle between the radial and cross-track separation
vd = 0.000 #-10 # Drift per revolutions m/resolution
d= -0.5# [m] - along track separation
rho_2 = (2*(eta**2) * d) /(3-eta**2) # [m]  - along-track separation
#rho_2 = (e*(3+2*eta**2) * d) /(3-eta**2)*rho_1 * np.cos(alpha) # [m]  - along-track separation for bounded symmnetic deputy motion in along track direction

print("RHO_2",rho_2)
print(d/1+e, d/1-e,  d*(1/(2*(eta**2)) /(3-eta**2)))
parameters=numpy.array([rho_1,rho_2,rho_3,alpha,beta,vd])

print("Formation parameters",parameters)



# Initial relative orbital elements
RNOE_0=Param2NROE(NOE_chief, parameters,param)


# ## J2 Invariant condition

# epsi_J2 = param["J"][0] 
# a_0 = NOE_chief[0]
# i_0 = NOE_chief[2]
# eta_0 = np.sqrt(1 - NOE_chief[3]**2 - NOE_chief[4]**2)
# a_non = a_0/ param["Primary"][1]
# L_0_non = np.sqrt(a_non)
# print("a_0",a_0)
# print("i_0",i_0)
# print("eta_0",eta_0)
# print("a_non",a_non)
# print("L_0_non",L_0_non)


# term_1 = -(epsi_J2)/(4*(L_0_non**4)*(eta_0**5))
# term_2 = (4+3*eta_0)
# term_3 = 1 + 5*np.cos(i_0)**2

# print("term_1",term_1)
# print("term_2",term_2)
# print("term_3",term_3)

# D = term_1 * term_2 * term_3
# print("D",D)

# del_eta = -(eta_0/4)*np.tan(i_0)*RNOE_0[2] 
# print("del_eta",del_eta)

# del_a = 2* D * a_0 * del_eta * param["Primary"][1]
# print("del_a",del_a)


# print("From the design parameters",RNOE_0)
# RNOE_0[0] = del_a


## Passive safety condition fomr C_traub
# RNOE_0[0]=0
# RNOE_0[2]=-RNOE_0[5]*numpy.cos(NOE_chief[2])

print("J2 Invariant condition",RNOE_0)


# Deputy spacecraft initial conditions
# assigning the state variables
a = NOE_chief[0]
l = NOE_chief[1]
i = NOE_chief[2]
q1 = NOE_chief[3]
q2 = NOE_chief[4]
OM = NOE_chief[5]

delta_a = RNOE_0[0]
delta_lambda0 = RNOE_0[1]
delta_i = RNOE_0[2]
delta_q1 = RNOE_0[3]
delta_q2 = RNOE_0[4]
delta_Omega = RNOE_0[5]


# Compute deputy orbital elements
a_d = a + delta_a               # Deputy semi-major axis
l_d = l + delta_lambda0         # Deputy mean longitude
i_d = i + delta_i               # Deputy inclination
q1_d = q1 + delta_q1            # Deputy eccentricity term q1
q2_d = q2 + delta_q2            # Deputy eccentricity term q2
OM_d = OM + delta_Omega         # Deputy RAAN

print("NS orbital elements computed.")
print("Relative orbital elements -> ",RNOE_0)
print("Chief orbital elements -> ",NOE_chief)
print("Deputy orbital elements -> ",a_d,l_d,i_d,q1_d,q2_d,OM_d)
# angle of attack for the deputy spacecraft
yaw_1 = 0*deg2rad  # [rad] - angle of attack = 0 assumption that V_sat = V_rel
yaw_2 = 90*deg2rad  # [rad] - angle of attack = 0
# 12 -> chief yaw angle
# 13 -> deputy yaw angle
# 14 -> deputy 1 yaw angle
# 15 -> deputy 2 yaw angle
yaw_c_d=numpy.array([yaw_1,yaw_2,0,0])
print("yaw angles",yaw_c_d)
print("Relative orbital elements",RNOE_0)
print("Chief orbital elements",NOE_chief)
print("RELATIVE ORBITAL ELEMTNS INITIAL", RNOE_0)
print("CHIEF INTIIAL ORBITAL ELEMENTS", NOE_chief)

 
# statement matrix [RNOE_0,NOE_chief,yaw_c_d]
# [6x1,6x1,4x1]
yy_o=numpy.concatenate((RNOE_0,NOE_chief,yaw_c_d))




# TIME #################################################
N_points = 10000
mu = param["Primary"][0]
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.05 * 365 * 24 * 60 * 60 / Torb
n_revolution = 25# n_revol_T
T_total = n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total, N_points)

print("Time span set.")
print("number of revolutions", n_revolution)
print("Total time", T_total)
# Simulate
uu_val = np.zeros((2, 1))  # Control inputs

param["Init"] = [NOE_chief[4], NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0
param["T_period"] = Torb
print(len(yy_o))

#### FILL THE PARAM MATRIX

# Fill the matrix using the dictionary
param_matrix[0, :] = param["Primary"]  # Primary body (mu, radius, angular velocity)
param_matrix[1, :] = param["J"]  # J2 effects
param_matrix[2, :] = [param["satellites"]["chief"]["mass"], 
                      param["satellites"]["chief"]["area"], 
                      param["satellites"]["chief"]["C_D"]]  # Chief Satellite
param_matrix[3, :] = [param["satellites"]["deputy_1"]["mass"], 
                      param["satellites"]["deputy_1"]["area"], 
                      param["satellites"]["deputy_1"]["C_D"]]  # Deputy Satellite
param_matrix[4, :] = [param["N_deputies"], 0, 0]  # Number of deputies
param_matrix[5, :] = param["sat"]  # Satellite design parameters
param_matrix[6, :] = [param["T_MAX"], 0, 0]  # Max thrust
param_matrix[7, :] = param["PHI_DOT"] + [0]  # PHI_DOT
param_matrix[8, :] = param["PHI"] + [0]  # PHI constraints
param_matrix[9, :] = [param["T_period"], 0, 0]  # Orbital period
param_matrix[10, :] = param["Init"]  # Initial conditions


# Define the symbolic variables for CasADi integration
t_sym = ca.MX.sym('t',1)  # Time
yy = ca.MX.sym('yy', len(yy_o))  # 16 state variables (6 NSROE for deputy, 6 NSROE for chief, 4 for yaw dynamics)
uu = ca.MX.sym('uu', len(uu_val))   # Control inputs for yaw dynamics 2 inputs

ADD_para = {'a': 1,
            'b': 2}

param_matrix_sym = ca.MX.sym('param_matrix', 11, 3)

dynamics_casadi_sym = Dynamics_casadi(t_sym, yy, uu, param_matrix_sym)

# Ensure Dynamics_casadi is returning symbolic output
assert isinstance(dynamics_casadi_sym, ca.MX), "Output of Dynamics_casadi is not symbolic!"
print("Dynamics_casadi output is symbolic.")
# Define the CasADi function for dynamics
dynamics_function = ca.Function('dynamics', [t_sym, yy, uu, param_matrix_sym], [dynamics_casadi_sym])

# # Define the ODE with just state `x` and control `p`
# ode = {'x': yy, 'p': ca.vertcat(uu,t), 'ode': dynamics_function(t, yy, uu, param)}

print(print, yy_o, uu_val)
# # dynamics_output = dynamics_function(0, yy_o, uu_val)
# print("!!!!!!!!!!!!!!!!Dynamics output:", dynamics_output.full())

# Setup the integrator
integrator = ca.integrator('integrator', 'cvodes', {
                           'x' : yy,
                            'p' : uu,
                            'ode' : dynamics_function}
                           , {'grid': teval, 'output_t0': True})

# Use the integrator with initial states and inputs
result = integrator(x0=yy_o, p=ca.vertcat(uu_val,0))

# Extract results
integrated_states = result['xf'].full()  # Integrated states
print("Integrated states:", integrated_states)
# Print the integrated states
print("Integrated states shape:", integrated_states.shape)

time_grid = np.linspace(0, T_total, len(integrated_states[0]))

# save the numpy array
# numpy.save('solution_x_100_30.npy', integrated_states)
# numpy.save('time_grid_100_30.npy', time_grid)


# Generate the LVLH Frame positions from CasADi results
rr_s = np.zeros((3, len(teval)))
angle_con_array=numpy.zeros((len(teval)))
# For each time step, compute the position
for i in range(len(teval)):
    yy1 = integrated_states[0:6,i]  # Deputy NSROE
    yy2 = integrated_states[6:12,i]  # Chief NSROE
    rr_s[:, i] = NSROE2LVLH_casadi(yy1, yy2, param).full().flatten()
    angle_con=con_chief_deputy_angle_casadi(integrated_states[:,i],param)
    # print("angle_con",)    
    angle_con_array[i] = ca.evalf(angle_con)
    # print("rr_s:", rr_s[:, i])

# Set the limits to 1 km for each axis
x_limits = [-0.5, 1.5]  # 1 km range centered at 0
y_limits = [-0.5, 1.5]  # 1 km range centered at 0
z_limits = [-0.5, 1.5]  # 1 km range centered at 0

# Create a figure with two subplots: one for the interactive 3D plot and one for the dynamic frame
fig = plt.figure(figsize=(12, 6))

# Interactive 3D plot
ax1 = fig.add_subplot(121, projection='3d')

# Plot the trajectory in 3D space
line, = ax1.plot3D(rr_s[0], rr_s[1], rr_s[2], 'black', linewidth=2, alpha=1, label='Deputy 1')

# Draw reference frame arrows (LVLH) on the interactive plot
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

# Apply the fixed limits (1 km range) to the interactive plot
ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)
ax1.set_zlim(z_limits)

# Set axis labels with km units and title
ax1.set_xlabel('x (km)')
ax1.set_ylabel('y (km)')
ax1.set_zlabel('z (km)')
ax1.set_title('LVLH frame - Deput Spacecraft (Interactive)')
ax1.legend(loc='best')

# Dynamic frame plot (linked to the interactive plot)
ax2 = fig.add_subplot(122, projection='3d')

# Static reference frame, which will update based on ax1's view
x_quiver2 = ax2.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1)
y_quiver2 = ax2.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1)
z_quiver2 = ax2.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1)

# Set the limits to zoom into 200 meters for each axis (adjust as needed)
x_zoom_limits = [-0.1, 0.1]  # 200 meters range centered at 0
y_zoom_limits = [-0.1, 0.1]  # 200 meters range centered at 0
z_zoom_limits = [-0.1, 0.1]  # 200 meters range centered at 0


# Set axis labels for the dynamic frame plot
ax2.set_xlabel('x (km)')
ax2.set_ylabel('y (km)')
ax2.set_zlabel('z (km)')
ax2.set_title('Dynamic LVLH Frame (Zoomed View)')

# Function to update the dynamic frame based on the interactive plot's view
def update_dynamic_frame(event):
    # Get the current view angle of ax1
    elev = ax1.elev
    azim = ax1.azim

    # Set the same view for ax2 (the dynamic frame)
    ax2.view_init(elev=elev, azim=azim)

    # Redraw the figure to reflect the changes
    fig.canvas.draw_idle()

# Connect the update function to the interactive plot (ax1)
ax1.figure.canvas.mpl_connect('motion_notify_event', update_dynamic_frame)

# Show the zoomed plot
plt.show() 

############# Relative orbital Dynamics ####################
fig, axs = plt.subplots(3, 1)

# Plot param on the first subplot
axs[0].plot(teval, rr_s[0])
axs[0].set_title('x')

# Plot param on the second subplot
axs[1].plot(teval, rr_s[1])
axs[1].set_title('y')

axs[2].plot(teval, rr_s[2])
axs[2].set_title('z')


fig, axs = plt.subplots(3, 1)

# Plot param on the first subplot
axs[0].plot(teval, integrated_states[0])
axs[0].set_title('semi major axis')

# Plot param on the second subplot
axs[1].plot(teval, integrated_states[1])
axs[1].set_title('mean true latitude')

axs[2].plot(teval, integrated_states[2])
axs[2].set_title('inclination')
 

fig, axs = plt.subplots(3, 1)

# Plot param on the first subplot
axs[0].plot(teval, integrated_states[3])
axs[0].set_title('q1')

# Plot param on the second subplot
axs[1].plot(teval, integrated_states[4])
axs[1].set_title('q2')

axs[2].plot(teval, integrated_states[5])
axs[2].set_title('right ascenstion of ascending node')



x = rr_s[0]
y = rr_s[1]
z = rr_s[2]
# Plot x and y
plt.figure(5)
plt.plot(x, y, label='x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Plot of x vs y')


# Plot z and y
plt.figure(6)
plt.plot(z, y, label='z vs y', color='g')
plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.title('Plot of z vs y')


# Plot x and z
plt.figure(7)
plt.plot(x, z, label='x vs z', color='r')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.title('Plot of x vs z')





fig, axs = plt.subplots(3, 1)

# Plot param on the first subplot
axs[0].plot(teval, integrated_states[12])
axs[0].set_title('Chief yaw angle')

# Plot param on the second subplot
axs[1].plot(teval, integrated_states[13])
axs[1].set_title('Deputy 1 yaw angle')

axs[2].plot(teval, angle_con_array)
axs[2].set_title('Constrains angle')

plt.show()