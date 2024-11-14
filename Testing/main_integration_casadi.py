import casadi as ca
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
import sys

# Add the folder containing your modules to the Python path
path_core = "..\\core"
path_casadi_converter = "..\\Casadi_modules"

# Get absolute paths and add them to sys.path
module_path_core = os.path.abspath(path_core)
module_path_casadi_converter = os.path.abspath(path_casadi_converter)

if module_path_core not in sys.path:
    sys.path.append(module_path_core)

if module_path_casadi_converter not in sys.path:
    sys.path.append(module_path_casadi_converter)

# Load the CasADi versions of the functions you've converted
<<<<<<< HEAD
from converted_functions_original import Dynamics_casadi, NSROE2LVLH_casadi, con_chief_deputy_angle_casadi, Dynamics_with_PID_casadi
=======
from converted_functions_original import Dynamics_casadi, NSROE2LVLH_casadi, con_chief_deputy_vec
>>>>>>> main_dev_testing
from TwoBP import Param2NROE, M2theta, NSROE2LVLH
from dynamics import Dynamics , uu_log  
from constrains import con_chief_deputy_angle, con_chief_deputy_vec_numeric

### CasADi Path Constraints Function
def path_constraints_casadi(t,xk, uk, param, lower, upper):
    """
    CasADi function to compute path constraints such as:
    - Relative distance constraint: lower <= r <= upper
    - Collision avoidance: r > d_koz
    - Yaw rate constraints
    - Control input constraints
    - Yaw angle (phi) constraints
    Args:
    - xk: The current state variables (14x1).
    - uk: The current control input (2x1).
    - param: A dictionary of parameters for the problem.
    - lower: Precomputed lower bound for relative distance (numeric).
    - upper: Precomputed upper bound for relative distance (numeric).
    Returns:
    - A CasADi vector of path constraint violations.
    """
    # Extract state variables for chief and deputy
    deputy_state = xk[0:6]
    chief_state = xk[6:12]
    phi_chief = xk[12]
    phi_deputy = xk[13]

    # Compute relative distance in LVLH frame (CasADi)
    r = ca.norm_2(NSROE2LVLH_casadi(deputy_state, chief_state, param))

    # Relative distance constraint: lower <= r <= upper
    constraint_r_lower = lower - r  # r >= lower  --> lower - r <= 0
    constraint_r_upper = r - upper  # r <= upper  --> r - upper <= 0

    # Collision avoidance constraint: r > d_koz
    d_koz = 0.3  # Distance threshold for collision avoidance
    collision_avoidance = d_koz - r  # d_koz < r  --> d_koz - r <= 0

    # angle constraint
    phi_deputy = con_chief_deputy_angle_casadi(xk, param)

    # Yaw rate constraints
    lb_phi_dot = param["PHI_DOT"][0]
    ub_phi_dot = param["PHI_DOT"][1]
    
    # Placeholder dynamics to get yaw rates from state variables
    yaw_rates = Dynamics_casadi(t, xk, param, uk)

    phi_dot_chief = yaw_rates[12]
    phi_dot_deputy = yaw_rates[13]

    # Yaw rate constraints for chief and deputy
    yaw_rate_constraints = ca.vertcat(
        lb_phi_dot - phi_dot_chief,  # Lower bound for chief yaw rate
        phi_dot_chief - ub_phi_dot,  # Upper bound for chief yaw rate
        lb_phi_dot - phi_dot_deputy,  # Lower bound for deputy yaw rate
        phi_dot_deputy - ub_phi_dot   # Upper bound for deputy yaw rate
    )

    # Control input constraints (torque limits)
    T_max = param["T_MAX"]
    control_constraints = ca.vertcat(
        -T_max - uk[0], uk[0] - T_max,  # Control limits for chief
        -T_max - uk[1], uk[1] - T_max   # Control limits for deputy
    )

    # Yaw angle (phi) constraints
    lb_phi = param["PHI"][0]
    ub_phi = param["PHI"][1]
    phi_constraints = ca.vertcat(
        lb_phi - phi_chief,  # Lower bound for chief yaw angle
        phi_chief - ub_phi,  # Upper bound for chief yaw angle
        lb_phi - phi_deputy,  # Lower bound for deputy yaw angle
        phi_deputy - ub_phi   # Upper bound for deputy yaw angle
    )

    # Combine all path constraints into one vector
    path_constraints = ca.vertcat(
        phi_deputy,    # Enforce angle constraint
        # constraint_r_lower,    # Enforce lower bound on relative distance
        # constraint_r_upper,    # Enforce upper bound on relative distance
        # collision_avoidance,   # Enforce collision avoidance
        # yaw_rate_constraints,  # Enforce yaw rate constraints
        # control_constraints,   # Enforce control input constraints
        # phi_constraints        # Enforce yaw angle constraints
    )
    print("path_constraints_CASADI",path_constraints.shape)
    
    return path_constraints


### Setup CasADi Integrator (DAE system) ###

# Define the CasADi DAE system (dynamics + algebraic constraints)
def dae_system(t, yy, param, uu, lower, upper):
    """
    Define the system of differential and algebraic equations (DAE) combining
    dynamics and path constraints.
    """
    # Dynamics
    dynamics = Dynamics_casadi(t, yy, param, uu)
    
    # Path constraints
    # path_constraints_val = path_constraints_casadi(t,t,yy, uu, param, lower, upper)
    path_constraints_val = con_chief_deputy_angle_casadi(yy, param)
    
    return dynamics#ca.vertcat(dynamics, path_constraints_val)


def path_constraints_numeric(t, xk, uk, param, lower, upper):
    """
    A numerical function to compute path constraints such as:
    - Relative distance constraint: lower <= r <= upper
    - Collision avoidance: r > d_koz
    - Yaw rate constraints
    - Control input constraints
    - Yaw angle (phi) constraints
    Args:
    - t: Time (not used here, but included for compatibility).
    - xk: The current state variables (14x1).
    - uk: The current control input (2x1).
    - param: A dictionary of parameters for the problem.
    - lower: Precomputed lower bound for relative distance (numeric).
    - upper: Precomputed upper bound for relative distance (numeric).
    Returns:
    - A NumPy array of path constraint violations.
    """
    # Extract state variables for chief and deputy
    deputy_state = xk[0:6]
    chief_state = xk[6:12]
    phi_chief = xk[12]
    phi_deputy = xk[13]

    # Compute relative distance in LVLH frame (using a numerical function for NSROE2LVLH)
    r = np.linalg.norm(NSROE2LVLH(deputy_state, chief_state, param))  # Replace with numerical NSROE2LVLH

    # Relative distance constraint: lower <= r <= upper
    constraint_r_lower = lower - r  # r >= lower  --> lower - r <= 0
    constraint_r_upper = r - upper  # r <= upper  --> r - upper <= 0

    # Collision avoidance constraint: r > d_koz
    d_koz = 0.3  # Distance threshold for collision avoidance
    collision_avoidance = d_koz - r  # d_koz < r  --> d_koz - r <= 0

    phi_deputy = con_chief_deputy_angle(xk, param)

    # Yaw rate constraints
    lb_phi_dot = param["PHI_DOT"][0]
    ub_phi_dot = param["PHI_DOT"][1]

    # Placeholder dynamics to get yaw rates from state variables (use a numeric version of Dynamics)
    yaw_rates = Dynamics(t, xk, param, uk)  # Replace with numeric dynamics

    phi_dot_chief = yaw_rates[12]
    phi_dot_deputy = yaw_rates[13]

    # Yaw rate constraints for chief and deputy
    yaw_rate_constraints = np.array([
        lb_phi_dot - phi_dot_chief,  # Lower bound for chief yaw rate
        phi_dot_chief - ub_phi_dot,  # Upper bound for chief yaw rate
        lb_phi_dot - phi_dot_deputy,  # Lower bound for deputy yaw rate
        phi_dot_deputy - ub_phi_dot   # Upper bound for deputy yaw rate
    ])

    # Control input constraints (torque limits)
    T_max = param["T_MAX"]
    control_constraints = np.array([
        -T_max - uk[0], uk[0] - T_max,  # Control limits for chief
        -T_max - uk[1], uk[1] - T_max   # Control limits for deputy
    ])

    # Yaw angle (phi) constraints
    lb_phi = param["PHI"][0]
    ub_phi = param["PHI"][1]
    phi_constraints = np.array([
        lb_phi - phi_chief,  # Lower bound for chief yaw angle
        phi_chief - ub_phi,  # Upper bound for chief yaw angle
        lb_phi - phi_deputy,  # Lower bound for deputy yaw angle
        phi_deputy - ub_phi   # Upper bound for deputy yaw angle
    ])

    # Combine all path constraints into one array
    path_constraints = np.concatenate([
        [phi_deputy],            # Enforce angle constraint
        # [constraint_r_lower],    # Enforce lower bound on relative distance
        # [constraint_r_upper],    # Enforce upper bound on relative distance
        # [collision_avoidance],   # Enforce collision avoidance
        # yaw_rate_constraints,    # Enforce yaw rate constraints
        # control_constraints,     # Enforce control input constraints
        # phi_constraints          # Enforce yaw angle constraints
    ])
    
    print("path_constraints NUMERIC",path_constraints.shape)

    return path_constraints

deg2rad = np.pi / 180
# Parameters (same as the Python version)
param = {
    "Primary": [3.98600433e5, 6378.16, 7.2921150e-5],
    "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients
    "satellites": {
        "chief": {
            "mass": 300,
            "area": 2,
            "C_D": 0.9
        },
        "deputy_1": {
            "mass": 250,
            "area": 1.8,
            "C_D": 0.85
        }
    },
    "N_deputies": 2,
    "sat": [0.0412, 0.0412, 1.2],  # Moments of inertia
    "T_MAX": 23e-6,  # Maximum torque (Nm)
    "PHI_DOT": [0.0, 0.1],  # Limits for yaw rate (rad/s)
    "PHI": [-np.pi / 2, np.pi / 2],  # Limits for yaw angle (rad)
    "K_p": 5 * 1e-8,  # Proportional gain
    "K_d": 1 * 1e-9, # Derivative gain
    "K_i": 0.1* 1e-10,  # Integral gain
}

print("Parameters initialized.")

deg2rad = numpy.pi / 180

# Deputy spacecraft relative orbital elements/ LVLH initial conditions
NOE_chief = numpy.array([6600,0.1,63.45*deg2rad,0.001,0.003,270.828*deg2rad])

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
eta = 1 - q1**2 - q2**2
p = term1
rp = a*(1-e)
ra = a*(1+e)

n = numpy.sqrt(mu/(a**3))
print("State variables calculated.")
print("rp", rp)
print("ra", ra)
print("e---", e)
print("a---", (rp + ra) / 2)

if rp < param["Primary"][1]:
    print("Satellite is inside the Earth's radius")
    exit()

if e == 0:  
    u = l
    r = (a * eta**2) / (1 + (q1 * numpy.cos(u)) + (q2 * numpy.sin(u)))
else:
    omega_peri = numpy.arccos(q1 / e)
    mean_anamoly = l - omega_peri
    theta_tuple = M2theta(mean_anamoly, e, 1e-8)

    theta = theta_tuple[0]
    u = theta + omega_peri
    r = (a * eta**2) / (1 + (q1 * numpy.cos(u)) + (q2 * numpy.sin(u)))

print("State variables assigned.")

# Design parameters for the formation
rho_1 = 0  # [m]  - radial separation 
rho_3 = 0  # [m]  - cross-track separation
alpha = 0  # [rad] - angle between the radial and along-track separation
beta = np.pi/2  # [rad] - angle between the radial and cross-track separation
vd = 0  # Drift per revolutions m/resolution
d = 10  # [m] - along track separation
rho_2 = (2*(eta**2) * d) /(3-eta**2)  # [m]  - along-track separation
print("RHO_2", rho_2)
print(d/1+e, d/1-e, d*(1/(2*(eta**2)) /(3-eta**2)))
parameters = numpy.array([rho_1, rho_2, rho_3, alpha, beta, vd])

print("Formation parameters set:", parameters)

# Initial relative orbital elements
RNOE_0 = Param2NROE(NOE_chief, parameters, param)
print("Initial relative orbital elements calculated.")

# Angle of attack for the deputy spacecraft
<<<<<<< HEAD
yaw_1 = 0.45  # [rad] - angle of attack = 0 assumption that V_sat = V_rel
yaw_2_val = con_chief_deputy_angle(numpy.concatenate((RNOE_0, NOE_chief,np.zeros(2))), param)
yaw_2 = 0.2
=======
yaw_1 = 45* deg2rad  # [rad] - angle of attack = 0 assumption that V_sat = V_rel
yaw_2_val = -0 * deg2rad#con_chief_deputy_angle(numpy.concatenate((RNOE_0, NOE_chief,np.zeros(2))), param)
yaw_2 = yaw_2_val
>>>>>>> main_dev_testing
print("Yaw angles calculated, yaw_2:", yaw_2)
# 12 -> chief yaw angle
# 13 -> deputy yaw angle
# 14 -> deputy 1 yaw angle
# 15 -> deputy 2 yaw angle
yaw_c_d=numpy.array([yaw_1,yaw_2,0,0])

PID_state = numpy.array([yaw_2_val-yaw_2, 0, 0])  # Initial PID state

print("RELATIVE ORBITAL ELEMENTS INITIAL", RNOE_0)
print("CHIEF INITIAL ORBITAL ELEMENTS", NOE_chief)

# Statement matrix [RNOE_0, NOE_chief, yaw_c_d]
yy_o = numpy.concatenate((RNOE_0, NOE_chief, yaw_c_d, PID_state))

# TIME #################################################
N_points = 10000
mu = param["Primary"][0]
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.05 * 365 * 24 * 60 * 60 / Torb
n_revolution = 10000# n_revol_T
T_total = n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total, N_points)

print("Time span set.")
print("number of revolutions", n_revolution)
print("Total time", T_total)
# Simulate
uu_o = np.zeros((2, 1))  # Control inputs

param["Init"] = [NOE_chief[4], NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0
param["T_period"] = Torb
print("Simulation parameters set.")

# Setup tolerances and other required variables
tol1 = 0.2  # Define appropriate values
tol2 = 0.2
d_koz = 0.3  # Define collision avoidance distance


### Numeric Function to Compute `r0` (Non-CasADi)
def compute_r0_numeric(deputy_state, chief_state, param):
    """
    A numeric function to compute the relative distance r0 between the deputy and the chief.
    Args:
    - deputy_state: The state vector of the deputy (6x1).
    - chief_state: The state vector of the chief (6x1).
    - param: A dictionary of parameters for the problem.
    Returns:
    - r0: The relative distance between the deputy and the chief.
    """
    relative_distance = np.linalg.norm(NSROE2LVLH(deputy_state, chief_state, param))
    return relative_distance


# Compute r0 numerically using the initial state
r0_numeric = compute_r0_numeric(yy_o[0:6], yy_o[6:12], param)
lower = r0_numeric - tol1
upper = r0_numeric + tol2

print(f"Computed r0 (numeric): {r0_numeric}")
print(f"Lower bound: {lower}, Upper bound: {upper}")




# Define the state, control, and algebraic variables for the DAE system
<<<<<<< HEAD
yy = ca.MX.sym('yy', 17)  # State vector
=======
yy = ca.MX.sym('yy', 16)  # State vector
>>>>>>> main_dev_testing
uu = ca.MX.sym('uu', 2)   # Control input vector
t = ca.MX.sym('t', 1)     # Time variable

print("tsdsdssssssssssssssssss",teval[0])
<<<<<<< HEAD
alg = ca.MX.sym('alg', 4) #ca.MX.sym('alg', path_constraints_casadi(t,yy, uu, param, lower, upper).size()[0])  # Algebraic variables
print("ALGEBRAIC VARIABLES",alg.size()[0])
=======
alg = ca.MX.sym('alg',1)  # Algebraic variables
>>>>>>> main_dev_testing
t = ca.MX.sym('t', 1)     # Time variable
# Set up the DAE system in CasADi
dae = {
    'x': yy,  # State variables
    #'z': alg,  # Algebraic variables (for path constraints)
<<<<<<< HEAD
    'p': uu,  # Control inputs
    'ode':  Dynamics_with_PID_casadi(t, yy, param, uu)  # Dynamics equations
    #'alg':  Dynamics_with_PID_casadi(t, yy, param, uu)[1]#path_constraints_casadi(t,yy, uu, param, lower, upper)  # Algebraic path constraints
=======
    'p': ca.vertcat(t, uu),  # Control inputs
    'ode': Dynamics_casadi(t, yy, param, uu),  # Dynamics equations
    #'alg': con_chief_deputy_vec(yy, param) - yy[13,-1]  # Algebraic path constraints
>>>>>>> main_dev_testing
}

# Define integration options
integrator_options = {
    'tf': T_total,              # Final time of the integration
    'grid': np.linspace(0, T_total, N_points),  # Time grid with 100 intervals
    'reltol': 1e-8,               # Relative tolerance
    'abstol': 1e-10,               # Absolute tolerance
    'output_t0': True           # Include initial time in the output
}

# Set up the integrator with IDAS
integrator = ca.integrator('integrator', 'idas', dae, integrator_options)

# Initial conditions and control input
yy0 = yy_o  # Initial state
<<<<<<< HEAD
alg0 =con_chief_deputy_angle(yy_o, param)#path_constraints_numeric(teval[0] ,yy_o, np.zeros(2), param, lower, upper)  # Initial algebraic variables

=======
alg0 = con_chief_deputy_vec_numeric(yy_o, param)-yy_o[13]#path_constraints_numeric(teval[0] ,yy_o, np.zeros(2), param, lower, upper)  # Initial algebraic variables
>>>>>>> main_dev_testing
#print("Initial algebraic variables:", alg0)

u0 = np.zeros(2)  # Initial control inputs

# Perform the integration
res = integrator(x0=yy0,  p=ca.vertcat(teval[0], u0)) # , z0=alg0,
print("Integration complete.")
# print("Results:", res)

# # Extract the results
# final_state = res['xf']
# final_alg = res['zf']

# print("Final state:", final_state)
# print("Final algebraic constraint values:", final_alg)

# print("Results extracted.", res.keys())
# Extract the state trajectory (all x values over the time grid)
solution_x = res['xf']  # This will give you the state values for each time step in the grid
time_grid = np.linspace(0, T_total, N_points)
extra_output = res['out']['extra']

print("Extra output (phi_dot):", extra_output)
exit()
# Reshape the results if necessary (usually solution_x will be an array of all states over time)
solution_x = np.array(solution_x.full()).reshape((16, N_points))  # Reshape to get [states x time points]


# Assume the integrator returns solution trajectories as 'solution_x' and 'time_grid' for plotting
# If not, adapt it based on how 'res' returns the time evolution of 'x' (state)
time_grid = np.linspace(0, T_total, len(solution_x[0]))


sol_y = solution_x
teval = time_grid
# save the numpy array
# numpy.save('solution_x_100_30.npy', solution_x)
# numpy.save('time_grid_100_30.npy', time_grid)

# print("Final state:", final_state)
# print("Final algebraic constraint values:", final_alg)
<<<<<<< HEAD

np.save("solution_x.npy", solution_x)
np.save("time_grid.npy", time_grid)
np.save("control_input.npy", u0)
exit()
=======
# exit()
>>>>>>> main_dev_testing
### Plotting the Results ###

print("Integration done....")

# Convert from NROE to Carterian co-ordinates.
rr_s=numpy.zeros((3,len(sol_y[0])))
vv_s=numpy.zeros((3,len(sol_y[0])))
angle_con_array=numpy.zeros((len(sol_y[0])))




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
    rr_s[:,i]=NSROE2LVLH(yy1,yy2,param)
    angle_con=con_chief_deputy_vec_numeric(sol_y[:,i],param)
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

<<<<<<< HEAD
plt.show()
# Show the zoomed plot
=======

# LVLH frame
# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D trajectory
ax.plot(rr_s[0], rr_s[1], rr_s[2], 'black', linewidth=2, alpha=1)

# Plot the origin
ax.plot([0], [0], [0], 'ro', linewidth=2, alpha=1)

# Set plot labels and title
ax.set_title('LVLH frame - Deputy Spacecraft')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# Show the zoomed plot
plt.show()

>>>>>>> main_dev_testing
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


<<<<<<< HEAD
# Plot semi-major axis, mean true latitude, inclination
fig, axs = plt.subplots(3, 1)
axs[0].plot(time_grid, solution_x[0, :], label='semi-major axis')
axs[1].plot(time_grid, solution_x[1, :], label='mean true latitude')
axs[2].plot(time_grid, solution_x[2, :], label='inclination')
axs[0].set_title('Semi-major axis')
axs[1].set_title('Mean true latitude')
axs[2].set_title('Inclination')

plt.tight_layout()

=======
fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol_y[0])
axs[0].set_title('semi major axis')

# Plot data on the second subplot
axs[1].plot(teval, sol_y[1])
axs[1].set_title('mean true latitude')

axs[2].plot(teval, sol_y[2])
axs[2].set_title('inclination')

>>>>>>> main_dev_testing

fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol_y[3])
axs[0].set_title('q1')

# Plot data on the second subplot
axs[1].plot(teval, sol_y[4])
axs[1].set_title('q2')

<<<<<<< HEAD
plt.tight_layout()
=======
axs[2].plot(teval, sol_y[5])
axs[2].set_title('right ascenstion of ascending node')


>>>>>>> main_dev_testing

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
<<<<<<< HEAD
plt.figure()
=======
plt.figure(6)
>>>>>>> main_dev_testing
plt.plot(z, y, label='z vs y', color='g')
plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.title('Plot of z vs y')


# Plot x and z
<<<<<<< HEAD
plt.figure()
=======
plt.figure(7)
>>>>>>> main_dev_testing
plt.plot(x, z, label='x vs z', color='r')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.title('Plot of x vs z')
<<<<<<< HEAD
=======




>>>>>>> main_dev_testing

fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol_y[12])
axs[0].set_title('Chief yaw angle')

# Plot data on the second subplot
axs[1].plot(teval, sol_y[13])
axs[1].set_title('Deputy 1 yaw angle')

<<<<<<< HEAD
axs[2].plot(time_grid, angle_con_array, label='Constraints angle')
axs[2].set_title('Constraints angle')
=======
axs[2].plot(teval, angle_con_array)
axs[2].set_title('Constrains angle')


# # # After integration
# # uu_log1 = np.array(uu_log)  # Convert to numpy array

# # # Check the shape of uu_log (it should be Nx2, where N is the number of time steps)
# # print(f"uu_log shape: {uu_log1.shape}")

# # # Plotting the results for both components of uu
# # plt.figure()
# # plt.plot(uu_log1[:, 0], label='uu component 1')  # First column
# # plt.plot(uu_log1[:, 1], label='uu component 2')  # Second column
# # plt.xlabel('Time (s)')
# # plt.ylabel('Input torque (uu)')
# # plt.title('Evolution of uu over time')
# # plt.legend()
# # plt.show()

>>>>>>> main_dev_testing
plt.show()




