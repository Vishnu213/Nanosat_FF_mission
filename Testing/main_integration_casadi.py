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
from converted_functions_original import Dynamics_casadi, NSROE2LVLH_casadi
from TwoBP import Param2NROE, M2theta, NSROE2LVLH
from dynamics import Dynamics
from constrains import con_chief_deputy_angle

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
        constraint_r_lower,    # Enforce lower bound on relative distance
        constraint_r_upper,    # Enforce upper bound on relative distance
        collision_avoidance,   # Enforce collision avoidance
        yaw_rate_constraints,  # Enforce yaw rate constraints
        control_constraints,   # Enforce control input constraints
        phi_constraints        # Enforce yaw angle constraints
    )
    print("path_constraints",path_constraints.shape)
    
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
    path_constraints_val = path_constraints_casadi(t,t,yy, uu, param, lower, upper)

    return ca.vertcat(dynamics, path_constraints_val)


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
        [constraint_r_lower],    # Enforce lower bound on relative distance
        [constraint_r_upper],    # Enforce upper bound on relative distance
        [collision_avoidance],   # Enforce collision avoidance
        yaw_rate_constraints,    # Enforce yaw rate constraints
        control_constraints,     # Enforce control input constraints
        phi_constraints          # Enforce yaw angle constraints
    ])
    


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
    "sat": [1.2, 1.2, 1.2],  # Moments of inertia
    "T_MAX": 23e-6,  # Maximum torque (Nm)
    "PHI_DOT": [0.0, 0.1],  # Limits for yaw rate (rad/s)
    "PHI": [-np.pi / 2, np.pi / 2]  # Limits for yaw angle (rad)
}

print("Parameters initialized.")

deg2rad = numpy.pi / 180

# Deputy spacecraft relative orbital elements/ LVLH initial conditions
NOE_chief = numpy.array([6500,0.1,63.45*deg2rad,0.5,0.2,270.828*deg2rad])
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
eta = 1 - q1**2 - q2**2
p = term1
rp = a*(1-e)
n = numpy.sqrt(mu/(a**3))

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
beta = 0  # [rad] - angle between the radial and cross-track separation
vd = 0  # Drift per revolutions m/resolution
d = -1  # [m] - along track separation
rho_2 = (2*(eta**2) * d) /(3-eta**2)  # [m]  - along-track separation
print("RHO_2", rho_2)
print(d/1+e, d/1-e, d*(1/(2*(eta**2)) /(3-eta**2)))
parameters = numpy.array([rho_1, rho_2, rho_3, alpha, beta, vd])

print("Formation parameters set:", parameters)

# Initial relative orbital elements
RNOE_0 = Param2NROE(NOE_chief, parameters, param)
print("Initial relative orbital elements calculated.")

# Angle of attack for the deputy spacecraft
yaw_1 = 0.12  # [rad] - angle of attack = 0 assumption that V_sat = V_rel
yaw_2_val = con_chief_deputy_angle(numpy.concatenate((RNOE_0, NOE_chief,np.zeros(2))), param)
yaw_2 = yaw_2_val
print("Yaw angles calculated, yaw_2:", yaw_2)
yaw_c_d = numpy.array([yaw_1, yaw_2])

print("RELATIVE ORBITAL ELEMENTS INITIAL", RNOE_0)
print("CHIEF INITIAL ORBITAL ELEMENTS", NOE_chief)

# Statement matrix [RNOE_0, NOE_chief, yaw_c_d]
yy_o = numpy.concatenate((RNOE_0, NOE_chief, yaw_c_d))

# TIME #################################################
N_points = 100000
mu = param["Primary"][0]
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.05 * 365 * 24 * 60 * 60 / Torb
n_revolution = n_revol_T
T_total = n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total, N_points)

print("Time span set.")
print("number of revolutions", n_revolution)
print("Total time", T_total)
# Simulate
uu_o = np.zeros((2, 1))  # Control inputs

param["Init"] = [NOE_chief[4], NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0

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
yy = ca.MX.sym('yy', 14)  # State vector
uu = ca.MX.sym('uu', 2)   # Control input vector
t = ca.MX.sym('t', 1)     # Time variable

print("tsdsdssssssssssssssssss",teval[0])
alg = ca.MX.sym('alg', path_constraints_casadi(t,yy, uu, param, lower, upper).size()[0])  # Algebraic variables
t = ca.MX.sym('t', 1)     # Time variable
# Set up the DAE system in CasADi
dae = {
    'x': yy,  # State variables
    #'z': alg,  # Algebraic variables (for path constraints)
    'p': uu,  # Control inputs
    'ode': Dynamics_casadi(t, yy, param, uu),  # Dynamics equations
    #'alg': path_constraints_casadi(t,yy, uu, param, lower, upper)  # Algebraic path constraints
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
alg0 = path_constraints_numeric(teval[0] ,yy_o, np.zeros(2), param, lower, upper)  # Initial algebraic variables
#print("Initial algebraic variables:", alg0)

u0 = np.zeros(2)  # Initial control inputs

# Perform the integration
res = integrator(x0=yy0, p=u0) # , z0=alg0,
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

# Reshape the results if necessary (usually solution_x will be an array of all states over time)
solution_x = np.array(solution_x.full()).reshape((14, N_points))  # Reshape to get [states x time points]


# Assume the integrator returns solution trajectories as 'solution_x' and 'time_grid' for plotting
# If not, adapt it based on how 'res' returns the time evolution of 'x' (state)
time_grid = np.linspace(0, T_total, len(solution_x[0]))

# print("Final state:", final_state)
# print("Final algebraic constraint values:", final_alg)

### Plotting the Results ###

# Plot semi-major axis of the chief
plt.figure()
plt.plot(time_grid, solution_x[6, :], label='Chief semi-major axis')
plt.xlabel('Time (s)')
plt.ylabel('Semi-major axis')
plt.title('Semi-major axis over time')
plt.legend()

# Plot yaw angles
plt.figure()
plt.plot(time_grid, solution_x[12, :], label='Chief yaw angle')
plt.plot(time_grid, solution_x[13, :], label='Deputy yaw angle')
plt.xlabel('Time (s)')
plt.ylabel('Yaw angle')
plt.title('Yaw angles over time')
plt.legend()

print("Plotting results.")

rr_s = np.zeros((3, len(time_grid)))
angle_con_array = np.zeros((len(time_grid)))

# For each time step, compute the position
for i in range(len(solution_x[12, :])):
    yy1 = solution_x[0:6, i]  # Deputy NSROE
    yy2 = solution_x[6:12, i]  # Chief NSROE
    rr_s[:, i] = NSROE2LVLH(yy1, yy2, param)
    angle_con_array[i] = con_chief_deputy_angle(solution_x[:, i], param)

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
# Plot the constraints angle over time
fig, axs = plt.subplots(3, 1)
axs[0].plot(time_grid, solution_x[0, :])
axs[0].set_title('x')

axs[1].plot(time_grid, solution_x[1, :])
axs[1].set_title('y')

axs[2].plot(time_grid, solution_x[2, :])
axs[2].set_title('z')

# Plot semi-major axis, mean true latitude, inclination
fig, axs = plt.subplots(3, 1)
axs[0].plot(time_grid, solution_x[0, :], label='semi-major axis')
axs[1].plot(time_grid, solution_x[1, :], label='mean true latitude')
axs[2].plot(time_grid, solution_x[2, :], label='inclination')

axs[0].set_title('Semi-major axis')
axs[1].set_title('Mean true latitude')
axs[2].set_title('Inclination')

plt.tight_layout()
plt.show()

# Plot q1, q2, right ascension of ascending node over time
fig, axs = plt.subplots(3, 1)
axs[0].plot(time_grid, solution_x[3, :], label='q1')
axs[1].plot(time_grid, solution_x[4, :], label='q2')
axs[2].plot(time_grid, solution_x[5, :], label='RAAN')

axs[0].set_title('q1')
axs[1].set_title('q2')
axs[2].set_title('Right Ascension of Ascending Node')

plt.tight_layout()
plt.show()

# Plot yaw angles and constraints angle over time
fig, axs = plt.subplots(3, 1)
axs[0].plot(time_grid, solution_x[12, :], label='Chief yaw angle')
axs[0].set_title('Chief yaw angle')

axs[1].plot(time_grid, solution_x[13, :], label='Deputy 1 yaw angle')
axs[1].set_title('Deputy 1 yaw angle')

axs[2].plot(time_grid, angle_con_array, label='Constraints angle')
axs[2].set_title('Constraints angle')

plt.tight_layout()
plt.show()

print("Plots generated.")