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
import numpy as np
import numpy
from scipy import integrate
import matplotlib.pyplot as plt
import os
import sys
import math
import time

from scipy.interpolate import interp1d
import pickle
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
    NSROE2LVLH_2,
       )

from dynamics import Dynamics_N, yaw_dynamics_N, yaw_dynamics, absolute_NSROE_dynamics, Dynamics, uu_log, uu_deputy
from constrains import con_chief_deputy_angle, con_chief_deputy_vec_numeric

from integrators import integrate_system


class StateVectorInterpolator:
    def __init__(self, teval, solution_x):
        self.interpolating_functions = [interp1d(teval, solution_x[i, :], kind='linear', fill_value="extrapolate") for i in range(solution_x.shape[0])]

    def __call__(self, t):
        return np.array([f(t) for f in self.interpolating_functions])


# Parameters that is of interest to the problem

data = {
    "Primary": [3.98600433e5,6378.16,7.2921150e-5],
    "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients

    # Satellites data including chief and deputies
    "satellites": {
        "chief": {
            "mass": 5,         # Mass in kg
            "area": 0.05,           # Cross-sectional area in m^2
            "C_D": 0.9,          # Drag coefficient
        },
        "deputy_1": {
            "mass": 5,
            "area": 0.05,
            "C_D": 0.85,
        }
    },
    "N_deputies": 2,  # Number of deputies
    "sat": [0.0412, 0.0412,0.0412],  # Moment of inertia for each satellite
    "T_period": 2000.0,  # Period of the sine wave
}

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
mu = data["Primary"][0]

e = numpy.sqrt(q1**2 + q2**2)
h = numpy.sqrt(mu*a*(1-e**2))
term1 = (h**2)/(mu)
eta = np.sqrt(1 - q1**2 - q2**2)
p = term1
rp = a*(1-e)
ra = a*(1+e)


n = numpy.sqrt(mu/(a**3))
print("State variables calculated.")
print("rp", rp)
print("ra", ra)
print("e---", e)
print("a---", (rp + ra) / 2)

if rp < 200+data["Primary"][1]:
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
term1 = (-2 * (a**(7/2)) * delta_omega_sunsynchronous * (1-e**2)**2 )/(3*(data["Primary"][1]**2)*data["J"][0]*numpy.sqrt(mu))
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
RNOE_0=Param2NROE(NOE_chief, parameters,data)


# ## J2 Invariant condition

# epsi_J2 = data["J"][0] 
# a_0 = NOE_chief[0]
# i_0 = NOE_chief[2]
# eta_0 = np.sqrt(1 - NOE_chief[3]**2 - NOE_chief[4]**2)
# a_non = a_0/ data["Primary"][1]
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

# del_a = 2* D * a_0 * del_eta * data["Primary"][1]
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
yaw_2 = 0*deg2rad  # [rad] - angle of attack = 0
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
# [6x1,6x1,2x1]
yy_o=numpy.concatenate((RNOE_0,NOE_chief,yaw_c_d))


# test for gauess equation
mu=data["Primary"][0]
Torb = 2*numpy.pi*numpy.sqrt(NOE_chief[0]**3/mu)    # [s]    Orbital period
n_revol_T = 365*24*60*60/Torb
n_revolution= 100 #n_revol_T #n_revol_T
T_total=n_revolution*Torb
print("Orbital period",Torb, "Number of orbits",n_revol_T)


t_span=[0,T_total]
teval=numpy.linspace(0, T_total, 100000)
# K=numpy.array([k1,k2])

data["Init"] = [NOE_chief[4],NOE_chief[3], 0]
data["T_period"] = Torb
uu = numpy.zeros((2,1)) # input torque to the dynamics model - it is fed inside the yaw dynamics.

print("Number of Period",n_revolution)
print("Orbital Period",Torb)
print("Time of Integration",T_total)
print("integration time step",teval[1]-teval[0])
print("Number of data points",len(teval))
print("Integration starting....")

# Start the timer
start_time = time.time()

sol=integrate.solve_ivp(Dynamics, t_span, yy_o,t_eval=teval,
                        method='RK45',args=(data,uu), rtol=1e-13, atol=1e-11,dense_output=True)

# sol=integrate.solve_ivp(Dynamics, t_span, yy_o,t_eval=teval,
#                         method='RK45',args=(data,uu), max_step=0.01, atol = 1, rtol = 1,dense_output=True)

# Check solver status
if sol.status != 0:
    print(f"Solver stopped early with status {sol.status}: {sol.message}")
    exit()

# h=0.1*(T_total)/(len(teval)) 
# print("Time step......h : ",h)
# t_values, sol_y = integrate_system("RK5", Dynamics,teval, yy_o,h, data, uu)
# # End the timer
end_time = time.time()

# Calculate the time taken
execution_time = end_time - start_time

# teval = t_values

sol_y = sol.y

teval = sol.t


# teval = t_values

print("Integration done....")
print("time",teval.shape,teval[0],teval[-1])
print("sol_y",sol_y.shape)
state_vector_function = StateVectorInterpolator(teval, sol_y)

print("Interpolating function created successfully.")


# # Save the function using pickle
with open('state_vector_function.pkl', 'wb') as f:
    pickle.dump(state_vector_function, f)

print("Interpolating function saved successfully.")

# Print the execution time
print(f"Time taken for integration: {execution_time:.4f} seconds")

print("Integration done....")

# Convert from NROE to Carterian co-ordinates.
rr_s=numpy.zeros((3,len(sol_y[0])))
vv_s=numpy.zeros((3,len(sol_y[0])))
angle_con_array=numpy.zeros((len(sol_y[0])))


mid_index = len(teval) // 2

for i in range(0,len(sol_y[0])):
    # if sol_y[5][i]>2*numpy.pi:
    #     sol_y[5][i]=

    # rr_s[:,i],vv_s[:,i]=NSROE2car(numpy.array([sol_y[0][i],sol_y[1][i],sol_y[2][i],
    #                                            sol_y[3][i],sol_y[4][i],sol_y[5][i]]),data)

    yy1=sol_y[0:6,i]
    yy2=sol_y[6:12,i]
    # if yy2[1]>2000:
    #     print("ANOMALY",yy2[1])
    # print("yy1",yy1)
    # print("yy2",yy2)
    rr_s[:,i]=NSROE2LVLH(yy1,yy2,data)
    angle_con=con_chief_deputy_vec_numeric(sol_y[:,i],data)
    angle_con_array[i] = angle_con

    # print("############# constains angle", angle_con)

    # h = COE[0]
    # e =COE[1]
    # i =COE[2]
    # OM = COE[3]
    # om =COE[4]
    # TA =COE[5]


print("mean position in x",numpy.mean(rr_s[0]))
print("mean position in y",numpy.mean(rr_s[1]))
print("mean position in z",numpy.mean(angle_con_array))
if np.isnan(sol_y).any():
    print("NAN values in the solution")
    exit()
else:
    print("No NAN values in the solution")

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
# ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)
# x-axis
# Add the unit vectors of the LVLH frame
# Define a constant length for the arrows
# Define a constant arrow length relative to the axis ranges
# arrow_length = 0.01  # Adjust this factor to change the relative size of arrows
# a=max(rr_s[0])
# b=max(rr_s[1])
# c= max(rr_s[2])

# d = max([a,b,c])
# # Normalize the vectors based on the axis scales
# x_axis = numpy.array([arrow_length * max(rr_s[0])/d, 0, 0])
# y_axis = numpy.array([0, arrow_length * max(rr_s[1])/d, 0])
# z_axis = numpy.array([0, 0, arrow_length * max(rr_s[2])/d])
# # add xlim and ylim
# ax.set_xlim(-d, d)
# ax.set_ylim(-d, d)

# # x-axis
# ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1)
# # y-axis
# ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1)
# # z-axis
# ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1)
# ax.plot3D(rr_s[0],rr_s[1],rr_s[2] , 'black', linewidth=2, alpha=1)
# ax.set_title('LVLH frame - Deput Spacecraft')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# The original rr_s array is already defined in your code as the spacecraft trajectory

# Set the limits to 1 km for each axis
x_limits = [-0.5, 1.5]  # 1 km range centered at 0
y_limits = [-0.5, 1.5]  # 1 km range centered at 0
z_limits = [-0.5, 1.5]  # 1 km range centered at 0

# Create a figure with two subplots: one for the interactive 3D plot and one for the dynamic frame
fig = plt.figure(figsize=(12, 6))

# Interactive 3D plot
ax1 = fig.add_subplot(121, projection='3d')

# Plot the trajectory in 3D space
line, = ax1.plot3D(rr_s[0], rr_s[1], rr_s[2], 'black', linewidth=2, alpha=1)

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


# LVLH frame
# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D trajectory
ax.plot(rr_s[0], rr_s[1], rr_s[2], 'black', linewidth=2, alpha=1)


# Plot the initial, midpoint, and final points
ax.plot(rr_s[0, 0], rr_s[1, 0], rr_s[2, 0], 'ko', linewidth=2, alpha=1, label='Start Point')  # Starting point
ax.plot(rr_s[0, mid_index], rr_s[1, mid_index], rr_s[2, mid_index], 'ro', markersize=8, label='Midpoint')  # Midpoint
ax.plot(rr_s[0, -1], rr_s[1, -1], rr_s[2, -1], 'bo', linewidth=2, alpha=1, label='End Point')  # End point

# Plot the origin
ax.plot([0], [0], [0], 'ro', linewidth=2, alpha=1)

# Set plot labels and title
ax.set_title('LVLH frame - Deputy Spacecraft')
ax.set_xlabel('Radial (m)')
ax.set_ylabel('Along track (m)')
ax.set_zlabel('Cross track (m)')
ax.legend()


# Show the zoomed plot
plt.show()

############# Relative orbital Dynamics ####################
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
axs[0].plot(teval, sol_y[0])
axs[0].set_title('semi major axis')

# Plot data on the second subplot
axs[1].plot(teval, sol_y[1])
axs[1].set_title('mean true latitude')

axs[2].plot(teval, sol_y[2])
axs[2].set_title('inclination')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol_y[3])
axs[0].set_title('q1')

# Plot data on the second subplot
axs[1].plot(teval, sol_y[4])
axs[1].set_title('q2')

axs[2].plot(teval, sol_y[5])
axs[2].set_title('right ascenstion of ascending node')



radial = rr_s[0]
along_track = rr_s[1]
cross_track = rr_s[2]

# Plot radial and along_track
plt.figure(5)
plt.plot(radial, along_track, label='Along track vs Radial')
plt.xlabel('Radial (m)')
plt.ylabel('Along track (m)')
plt.legend()
plt.title('Plot of Along track vs Radial')

# Plot cross_track and along_track
plt.figure(6)
plt.plot(cross_track, along_track, label='Along track vs Cross track', color='g')
plt.xlabel('Cross track (m)')
plt.ylabel('Along track (m)')
plt.legend()
plt.title('Plot of Along track vs Cross track')

# Plot radial and cross_track
plt.figure(7)
plt.plot(radial, cross_track, label='Cross track vs Radial', color='r')
plt.xlabel('Radial (m)')
plt.ylabel('Cross track (m)')
plt.legend()
plt.title('Plot of Cross track vs Radial')




fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol_y[12])
axs[0].set_title('Chief yaw angle')

# Plot data on the second subplot
axs[1].plot(teval, sol_y[13])
axs[1].set_title('Deputy 1 yaw angle')

axs[2].plot(teval, angle_con_array)
axs[2].set_title('Constrains angle')


# After integration
uu_log1 = np.array(uu_log)  # Convert to numpy array
uu_log2 =np.array(uu_deputy)  # Convert to numpy array

u_chief = uu_log1[:, 0]  # Chief input torque
u_deputy = uu_log1[:, 1]  # Deputy input torque
u_deputy1 = uu_log1[:, 2]  # Deputy

# Check the shape of uu_log (it should be Nx2, where N is the number of time steps)
print(f"uu_log shape: {uu_log1.shape}")

# # Plotting the results for both components of uu
# plt.figure()
# plt.plot(uu_log1[:, 0], label='uu component 1')  # First column
# plt.plot(uu_log1[:, 1], label='uu component 2')  # Second column
# plt.xlabel('Time (s)')
# plt.ylabel('Input torque (uu)')
# plt.title('Evolution of uu over time')
# plt.legend()

print("uu_log1",np.linalg.norm(uu_log1[:, 0]))
# Plotting the results for both components of uu
plt.figure()
plt.plot(np.linalg.norm(uu_log1[:,0], axis=1), label='Chief force')  # First column
plt.plot(np.linalg.norm(uu_log1[:, 1], axis=1), label='deputy force')  # First column
plt.plot(np.linalg.norm(uu_log1[:, 2], axis=1), label='diff force')  # First column
plt.xlabel('Time (s)')
plt.ylabel('Force in N (uu)')
plt.title('Evolution of uu over time')
plt.legend()

plt.show()




