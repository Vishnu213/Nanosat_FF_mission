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

# Time discretization parameters
N = 100  # Number of control intervals
T = T_total  # Total time in seconds (e.g., 1 hour)
dt = T / N
teval = np.linspace(0, T_total, N + 1)

# State and control variables
yy = ca.MX.sym('yy', len(yy_o))  # 14 state variables (6 NSROE for deputy, 6 NSROE for chief, 2 for yaw)
uu = ca.MX.sym('uu', 2)  # Control inputs for yaw dynamics
t = ca.MX.sym('t')  # Time variable


# Bounds on yaw angles, yaw rates, and control torque from ACADOS
lb_phi = param["PHI"][0]  # Lower bound for yaw angle
ub_phi = param["PHI"][1]  # Upper bound for yaw angle
lb_phi_dot = param["PHI_DOT"][0]  # Lower bound for yaw rate
ub_phi_dot = param["PHI_DOT"][1]  # Upper bound for yaw rate
T_max = param["T_MAX"]  # Control torque limit


# Dynamics function in CasADi
dynamics_casadi_sym = Dynamics_casadi(t, yy, param, uu)

print("Dynamics function defined.")

# Dynamics and constraints
r0 = np.linalg.norm(NSROE2LVLH_casadi(yy_o[0:6], yy_o[6:12], param))
lower = r0 - tol1
upper = r0 + tol2

# Define the NLP problem
X = ca.MX.sym('X', 14, N + 1)  # State trajectory
U = ca.MX.sym('U', 2, N)  # Control trajectory

# Objective function: minimize -yy[6] (minimizing the semi-major axis of chief)


objective = -X[6, -1]  # Minimize the semi-major axis of the chief

print("Objective function defined.")

# Constraints and bounds from ACADOS
g = []
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    
    # Collocation dynamics
    xk_next = X[:, k + 1]
    f_dyn = Dynamics_casadi(t, xk, param, uk)
    g.append(xk_next - (xk + dt * f_dyn))

    # Relative distance constraint
    r = NSROE2LVLH_casadi(xk[0:6], xk[6:12], param)
    g.append(ca.vertcat(lower - r, r - upper))
    
    # Collision avoidance constraint
    g.append(d_koz - r)
    
    # Yaw rate constraints
    phi_dot = f_dyn[12:14]
    g.append(ca.vertcat(phi_dot[0] - lb_phi_dot, ub_phi_dot - phi_dot[0]))
    g.append(ca.vertcat(phi_dot[1] - lb_phi_dot, ub_phi_dot - phi_dot[1]))
    
    # Control torque constraints
    g.append(ca.vertcat(uk[0] - T_max, uk[1] - T_max))
    # Dynamics constraints (ensure dynamics are respected)
    constraint9 = ca.vertcat(f_dyn - Dynamics_casadi(t, xk, param, uk))
    g.append(constraint9)


print("constrain 9 ",constraint9)



# Flatten constraints
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

# Set initial guess
x_guess = np.tile(yy_o, (N + 1, 1)).T
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
    'print_time': False,
    'ipopt.derivative_test': 'first-order',  # Enable derivative checking
    'ipopt.derivative_test_tol': 1e-3  # Tolerance for derivative test
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