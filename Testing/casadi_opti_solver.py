import casadi as ca
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
import sys

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

# Load the CasADi versions of the functions you've converted
from converted_functions import Dynamics_casadi, NSROE2LVLH_casadi, con_chief_deputy_angle_casadi

from TwoBP import Param2NROE, M2theta

print("Modules loaded successfully.")

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

    # New fields
    "T_MAX": 23e-6,  # Nm
    "PHI_DOT": 0.1* (np.pi / 180),  # rad/s
    "PHI": 90 * (np.pi / 180)  # Convert 90 degrees to radians
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
yaw_2 = 0.08  # [rad] - angle of attack = 0
yaw_c_d = numpy.array([yaw_1, yaw_2])

print("RELATIVE ORBITAL ELEMENTS INITIAL", RNOE_0)
print("CHIEF INITIAL ORBITAL ELEMENTS", NOE_chief)

# Statement matrix [RNOE_0, NOE_chief, yaw_c_d]
yy_o = numpy.concatenate((RNOE_0, NOE_chief, yaw_c_d))

# Test for Gauss equation
mu = param["Primary"][0]
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.0005 * 365 * 24 * 60 * 60 / Torb
n_revolution = 0.005  # n_revol_T
T_total = n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total, 1000)

# Simulate
uu_o = np.zeros((2, 1))  # Control inputs

param["Init"] = [NOE_chief[4], NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0

print("Simulation parameters set.")

# Setup tolerances and other required variables
tol1 = 0.2  # Define appropriate values
tol2 = 0.2
d_koz = 0.3  # Define collision avoidance distance

# Define your Opti problem
opti = ca.Opti()

# Problem parameters
N = 100  # Number of control intervals
T_total = 2 * np.pi * np.sqrt(6500**3 / 3.98600433e5)  # Orbital period
T = T_total
dt = T / N

# Define state variables (14 state variables: 6 for NSROE deputy, 6 for NSROE chief, 2 for yaw angles)
X = opti.variable(14, N + 1)

# Define control variables (2 control variables for yaw dynamics)
U = opti.variable(2, N)

# Time variable (not needed explicitly in Opti formulation, but present for dynamics)
t = ca.MX.sym('t')

# Objective: Minimize the semi-major axis of the chief (X[6] is the semi-major axis of the chief)
opti.minimize(-X[6, -1])

# Define tolerance values
tol1 = 0.2
tol2 = 0.2
d_koz = 0.3  # Collision avoidance distance

# Define the RK5 integration method
def rk5_step(f, t, x, u, dt, param):
    k1 = f(t, x, param, u)
    k2 = f(t + 0.25 * dt, x + 0.25 * dt * k1, param, u)
    k3 = f(t + 0.375 * dt, x + 0.375 * dt * k2, param, u)
    k4 = f(t + 0.923076923 * dt, x + 0.923076923 * dt * k3, param, u)
    k5 = f(t + dt, x + dt * k4, param, u)
    
    # Compute the weighted average of the slopes
    x_next = x + dt * (0.1185185185 * k1 + 0.5189863548 * k2 + 0.50613149 * k3 + 0.018963184 * k4 + 0.2374078411 * k5)
    
    return x_next

# Collocation constraints using RK5 for the state and dynamics
for k in range(N):
    xk = X[:, k]       # State at step k
    uk = U[:, k]       # Control at step k
    x_next = X[:, k+1]  # State at step k+1
    
    # Compute next state using RK5
    x_next_pred = rk5_step(Dynamics_casadi, t, xk, uk, dt, param)
    
    # Add the collocation constraint (RK5-based prediction should match the next state)
    opti.subject_to(x_next == x_next_pred)

    # Calculate the relative position `r` between chief and deputy in the LVLH frame
    r = ca.norm_2(NSROE2LVLH_casadi(xk[0:6], xk[6:12], param))

    # # Constraint 1: Lower and upper bounds on `r`
    # opti.subject_to(tol2 - r <= 0)
    # opti.subject_to(r - tol1 <= 0)

    # # Constraint 2: d_koz < r (collision avoidance)
    # opti.subject_to(d_koz - r <= 0)

    # Constraint 3 & 4: phi_dot range constraints for chief and deputy
    opti.subject_to(param["PHI_DOT"] - X[12, k] <= 0)  # Chief phi_dot
    opti.subject_to(X[12, k] - param["PHI_DOT"] <= 0)
    
    opti.subject_to(param["PHI_DOT"] - X[13, k] <= 0)  # Deputy phi_dot
    opti.subject_to(X[13, k] - param["PHI_DOT"] <= 0)

    # Constraint 5 & 6: Yaw angle range constraints for chief and deputy
    opti.subject_to(param["PHI"] - X[12, k] <= 0)  # Chief yaw angle
    opti.subject_to(X[12, k] - param["PHI"] <= 0)

    opti.subject_to(param["PHI"] - X[13, k] <= 0)  # Deputy yaw angle
    opti.subject_to(X[13, k] - param["PHI"] <= 0)

    # Constraint 7 & 8: Control input torque constraints
    opti.subject_to(param["T_MAX"] - U[0, k] <= 0)  # Chief torque
    opti.subject_to(U[0, k] - param["T_MAX"] <= 0)

    opti.subject_to(param["T_MAX"] - U[1, k] <= 0)  # Deputy torque
    opti.subject_to(U[1, k] - param["T_MAX"] <= 0)

    # Constraint 9: Dynamics constraint
    x_dot = Dynamics_casadi(t, xk, param, uk)
    opti.subject_to(ca.vertcat(x_dot - x_next_pred) == 0)

    # # Constraint 10: Relative angle constraint between chief and deputy
    # phi_deputy = con_chief_deputy_angle_casadi(xk, param)
    # opti.subject_to(phi_deputy <= 0)

# Initial condition constraints
opti.subject_to(X[:, 0] == yy_o)  # Initial state should match the given initial condition

# Set initial guesses for the optimization problem
opti.set_initial(X, np.tile(yy_o, (N + 1, 1)).T)
opti.set_initial(U, np.zeros((2, N))) 

# Solver options
opti.solver('ipopt', {
    'expand': True,
    'ipopt.print_level': 4,  # Detailed print level (higher values show more information)
    'ipopt.max_iter': 1000,  # Max iterations
    'ipopt.tol': 1e-6,  # Convergence tolerance
    'print_time': True,  # Print computation time
    'ipopt.sb': 'yes',  # Show solver's internal message
})

# Solve the problem
try:
    sol = opti.solve()

    # Check solver status
    if opti.stats()['success']:
        print("Solver was successful.")
    else:
        print("Solver failed to find a solution.")

    # # Extract solution
    # solution_x = sol.value(X)
    # solution_u = sol.value(U)

    # print("Solution extracted.")

except RuntimeError as e:
    print("Solver encountered an error:", str(e))
    print("Solver encountered an error:", str(e))
    # Debugging - retrieve the latest values of the variables at the point of failure
    X_value = opti.debug.value(X)
    U_value = opti.debug.value(U)
    print("State variables at failure:", X_value)
    print("Control variables at failure:", U_value)

# Ensure solution_x and solution_u are defined
if solution_x is not None and solution_u is not None:
    # Plotting the results
    time_grid = np.linspace(0, T, N + 1)

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

    rr_s = np.zeros((3, len(teval)))
    angle_con_array = numpy.zeros((len(teval)))
    # For each time step, compute the position
    for i in range(len(solution_x[12, :])):
        yy1 = solution_x[0:6, i]  # Deputy NSROE
        yy2 = solution_x[6:12, i]  # Chief NSROE
        rr_s[:, i] = NSROE2LVLH_casadi(yy1, yy2, param).full().flatten()

    # Plot the LVLH frame trajectory in 3D
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot3D(rr_s[0, :], rr_s[1, :], rr_s[2, :], 'black', linewidth=2, alpha=1, label='Deputy 1')
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')
    ax1.set_zlabel('z (km)')
    ax1.set_title('LVLH frame - Deputy Spacecraft (Interactive)')
    ax1.legend(loc='best')

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

    # Plot yaw angles
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
else:
    print("No solution to plot.")