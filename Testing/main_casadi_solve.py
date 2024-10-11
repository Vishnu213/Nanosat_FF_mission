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
from converted_functions_original import Dynamics_casadi, NSROE2LVLH_casadi, con_chief_deputy_angle_casadi

from TwoBP import Param2NROE, M2theta

print("Modules loaded successfully.")

# Define RK5 integration method
def rk5_step(f, t, x, u, dt, param):
    k1 = f(t, x, param, u)
    k2 = f(t + 0.25 * dt, x + 0.25 * dt * k1, param, u)
    k3 = f(t + 0.375 * dt, x + 0.375 * dt * k2, param, u)
    k4 = f(t + 0.923076923 * dt, x + 0.923076923 * dt * k3, param, u)
    k5 = f(t + dt, x + dt * k4, param, u)
    x_next = x + dt * (0.1185185185 * k1 + 0.5189863548 * k2 + 0.50613149 * k3 + 0.018963184 * k4 + 0.2374078411 * k5)
    return x_next


def RK4_step(f, t, y, h, *args):
    """
    Runge-Kutta 4th-order method for one step of integration.
    Args:
    - f: The function that defines the system dynamics f(t, y)
    - t: Current time
    - y: Current state vector
    - h: Time step
    - args: Additional arguments to be passed to f
    """
    k1 = h * f(t, y, *args)
    k2 = h * f(t + h/2, y + k1/2, *args)
    k3 = h * f(t + h/2, y + k2/2, *args)
    k4 = h * f(t + h, y + k3, *args)
    
    # Update y
    y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_next


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
NOE_chief = numpy.array([6500,0.1,45*deg2rad,0.5,0.2,270.828*deg2rad])
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
N_points = 1000
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.05 * 365 * 24 * 60 * 60 / Torb
n_revolution = 5  # n_revol_T
T_total = n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total,N_points)

# Simulate
uu_o = np.zeros((2, 1))  # Control inputs

param["Init"] = [NOE_chief[4], NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0

print("Simulation parameters set.")

# Setup tolerances and other required variables
tol1 = 0.2  # Define appropriate values
tol2 = 0.2
d_koz = 0.3  # Define collision avoidance distance

# Time discretization parameters
N = N_points  # Number of control intervals
T = T_total  # Total time in seconds (e.g., 1 hour)
dt = T / N
teval = np.linspace(0, T_total, N + 1)

# State and control variables
yy = ca.MX.sym('yy', len(yy_o))  # 14 state variables (6 NSROE for deputy, 6 NSROE for chief, 2 for yaw)
uu = ca.MX.sym('uu', 2)  # Control inputs for yaw dynamics
t = ca.MX.sym('t')  # Time variable

# Dynamics function in CasADi
dynamics_casadi_sym = Dynamics_casadi(t, yy, param, uu)

print("Dynamics function defined.")

# Define the NLP problem
X = ca.MX.sym('X', 14, N + 1)  # State trajectory
U = ca.MX.sym('U', 2, N)  # Control trajectory

# Objective function: minimize -yy[6] (minimizing the semi-major axis of chief)


objective = -X[6, -1]  # Minimize the semi-major axis of the chief

epsilon = 1e-8  # Small constant

# Original objective function
objective = -X[6, -1]  # Minimize the semi-major axis of the chief

# Add Mayer cost term (sum of squares of control terms multiplied by epsilon)
mayer_cost = epsilon * ca.sumsqr(U)

# Updated objective function
objective += mayer_cost


print("Objective function defined.")

# Collocation constraints and dynamics
g = []  # List of constraints
rr_s = np.zeros((3, N + 1))  # To store LVLH frame positions
angle_con_array = np.zeros(N + 1)  # To store constraint angles
for k in range(N):
    # State at collocation points
    xk = X[:, k]
    uk = U[:, k]

    # Dynamics at next point
    x_next = X[:, k + 1]

    # Compute the collocation using explicit Euler
    # xk_dot = Dynamics_casadi(t, xk, param, uk)
    x_next_pred = rk5_step(Dynamics_casadi, t, xk, uk, dt, param)
    
    # x_next_pred = xk + dt * xk_dot


    # Enforce the collocation constraint
    g.append(x_next - x_next_pred)

    # Enforce the path constraints at each collocation point
    r = NSROE2LVLH_casadi(xk[0:6], xk[6:12], param)
    constraint1_lower = 0.2 - r
    constraint1_upper = r - 0.2
    constraint2 = 0.3 - r
    phi_deputy = con_chief_deputy_angle_casadi(xk, param)
    constraint7 = ca.vertcat(-param["T_MAX"], uk[0] - param["T_MAX"])
    constraint8 = ca.vertcat(-param["T_MAX"], uk[1] - param["T_MAX"])

    g.append(ca.vertcat(
        constraint1_lower, constraint1_upper,
        # constraint2,
        constraint7, constraint8,
        # phi_deputy
    ))

    # Compute LVLH frame positions


print("Collocation constraints and dynamics defined.")

# Flatten the list of constraints
g = ca.vertcat(*g)

# Set up the decision variables for the optimization problem
decision_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

# Define bounds on decision variables and constraints
lbg = -np.inf * np.ones(g.size()[0])  # Lower bounds on constraints
ubg = np.zeros(g.size()[0])  # Upper bounds on constraints

# Set up the bounds on states and controls
lbx = -np.inf * np.ones(decision_vars.size()[0])
ubx = np.inf * np.ones(decision_vars.size()[0])

# Update control bounds based on constraints
for k in range(N):
    lbx[14 * (N + 1) + 2 * k] = -param["T_MAX"]  # Lower bound on control for chief torque
    ubx[14 * (N + 1) + 2 * k] = param["T_MAX"]  # Upper bound on control for chief torque
    lbx[14 * (N + 1) + 2 * k + 1] = -param["T_MAX"]  # Lower bound on control for deputy torque
    ubx[14 * (N + 1) + 2 * k + 1] = param["T_MAX"]  # Upper bound on control for deputy torque

print("Bounds on decision variables and constraints set.")


yy_ref = np.load("solution_x_1000.npy")
# Set initial guess
x_guess = yy_ref
u_guess = np.tile(uu_o, (N, 1)).T

# Flatten the initial guess for optimization
initial_guess = ca.vertcat(ca.reshape(x_guess, -1, 1), ca.reshape(u_guess, -1, 1))

print("Initial guess set.")

# Set up the NLP solver
nlp = {
    'x': decision_vars,
    'f': objective,
    'g': g
}
opts = {
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-5,
    'expand': True,
    'ipopt.print_level': 3,
    'print_time': False
    # 'jacobian_approximation': 'finite-difference'  # Use finite-difference for Jacobian
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

print("NLP solver set up.")

# Solve the problem
solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

# Check solver status
if solver.stats()['success']:
    print("Solver was successful.")
else:
    print("Solver failed.")

# Extract solution
solution_x = solution['x'][:14 * (N + 1)].full().reshape((14, N + 1))
solution_u = solution['x'][14 * (N + 1):].full().reshape((2, N))

print("Solution extracted.")

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

print("Plotting results.")

rr_s = np.zeros((3, len(teval)))
angle_con_array = numpy.zeros((len(teval)))
# For each time step, compute the position
for i in range(len(solution_x[12, :])):
    yy1 = solution_x[0:6, i]  # Deputy NSROE
    yy2 = solution_x[6:12, i]  # Chief NSROE
    rr_s[:, i] = NSROE2LVLH_casadi(yy1, yy2, param).full().flatten()
    angle_con_array[i] = con_chief_deputy_angle_casadi(solution_x[0:12,i], param).full().flatten()
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