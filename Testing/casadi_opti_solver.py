import casadi as ca
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
import sys
import time
from scipy.interpolate import interp1d
import pickle

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
from converted_functions_original import Dynamics_casadi, NSROE2LVLH_casadi, con_chief_deputy_angle_casadi, con_chief_deputy_vec

from TwoBP import Param2NROE, M2theta, NSROE2LVLH
from constrains import con_chief_deputy_angle

print("Modules loaded successfully.")

deg2rad = np.pi / 180
# Parameters (same as the Python version)

class StateVectorInterpolator:
    def __init__(self, teval, solution_x):
        self.interpolating_functions = [interp1d(teval, solution_x[i, :], kind='linear', fill_value="extrapolate") for i in range(solution_x.shape[0])]
    
    def __call__(self, t):
        return np.array([f(t) for f in self.interpolating_functions])




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
    "T_MAX": [-23e-3, 23e-3],  # Maximum torque (Nm)
    "PHI_DOT": [-0.1, 0.1],  # Limits for yaw rate (rad/s)
    "PHI": [-4*np.pi / 2, 4*np.pi / 2]  # Limits for yaw angle (rad)
}

print("Parameters initialized.")

deg2rad = numpy.pi / 180

# Deputy spacecraft relative orbital elements/ LVLH initial conditions
NOE_chief = numpy.array([6500,0.5,40*deg2rad,0.5,0.2,45*deg2rad])
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

print("Load numpy array for initial guass")
# yy_ref = np.load("solution_x_200_100.npy")



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

# Test for Gauss equation
mu = param["Primary"][0]
N_points = 100
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.05 * 365 * 24 * 60 * 60 / Torb
n_revolution =100  # n_revol_T
T_total = n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total, N_points)

# Load the function
with open('state_vector_function.pkl', 'rb') as f:
    state_vector_function = pickle.load(f)

yy_ref = numpy.zeros((14, len(teval)))
for i in range(len(teval)):
    yy_ref[:, i] = state_vector_function(teval[i])

print("yy_ref>size",yy_ref.shape, yy_ref[:,-1].shape)
l_colm = yy_ref[:,-1].reshape(-1,1)
a= np.concatenate((yy_ref,l_colm),axis =1)
print("a>size",a.shape)


print("Formation parameters set:", parameters)


# Simulate
uu_o = np.zeros((2, 1))  # Control inputs

param["Init"] = [NOE_chief[4], NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0

print("Simulation parameters set.")

# Setup tolerances and other required variables
tol1 = 0.2  # Define appropriate values
tol2 = 0.2
d_koz = 0.3  # Define collision avoidance distance


# Problem parameters
N = N_points  # Number of control intervals
T_total = 2 * np.pi * np.sqrt(6500**3 / 3.98600433e5)  # Orbital period
T = T_total
dt = T / N

# Define tolerance values
tol1 = 0.2
tol2 = 0.2
d_koz = 0.3  # Collision avoidance distance
r0 = np.linalg.norm(NSROE2LVLH(yy_o[0:6], yy_o[6:12], param))
start_time = time.time()
############################################
# # # Define your Opti problem
opti = ca.Opti()

# Define state variables (14 states: 6 NSROE deputy, 6 NSROE chief, 2 yaw angles)
X = opti.variable(14, N + 1)

# Define control variables (2 control variables for yaw dynamics)
U = opti.variable(2, N)

# Define the dynamics function
t = ca.MX.sym('t')
dynamics_casadi_sym = Dynamics_casadi(t, X[:, 0], param, U[:, 0])

# Objective: Minimize the semi-major axis of the chief
epsilon = 1e-8
control_cost = epsilon * ca.sumsqr(U)

opti.minimize(0.25*-X[6, -1]/1e3 + 0.8*(con_chief_deputy_vec(X, param)**2-X[13,-1]**2))

# Define RK5 integration method
def rk5_step(f, t, x, u, dt, param):
    k1 = f(t, x, param, u)
    k2 = f(t + 0.25 * dt, x + 0.25 * dt * k1, param, u)
    k3 = f(t + 0.375 * dt, x + 0.375 * dt * k2, param, u)
    k4 = f(t + 0.923076923 * dt, x + 0.923076923 * dt * k3, param, u)
    k5 = f(t + dt, x + dt * k4, param, u)
    x_next = x + dt * (0.1185185185 * k1 + 0.5189863548 * k2 + 0.50613149 * k3 + 0.018963184 * k4 + 0.2374078411 * k5)
    return x_next


# Collocation constraints using RK5 for the state and dynamics
dt = T_total / N
for k in range(N):
    xk = X[:, k]       # State at step k
    uk = U[:, k]       # Control at step k
    x_next = X[:, k+1]  # State at step k+1

    # Compute next state using RK5
    x_next_pred = rk5_step(Dynamics_casadi, t, xk, uk, dt, param)

    # Add the collocation constraint (RK5-based prediction should match the next state)
    opti.subject_to(x_next == x_next_pred)

    # # Calculate the relative position `r` between chief and deputy in LVLH frame
    # # r = ca.norm_2(NSROE2LVLH_casadi(xk[0:6], xk[6:12], param))

    # # Inequality Constraints: Position and Collision Avoidance

    # # opti.subject_to(r >= r0-0.5)  # Lower bound constraint
    # # opti.subject_to(r <= r0+3)  # Upper bound
    # # opti.subject_to(d_koz**2 < r**2)   # Collision avoidance

    # Dynamics derivative (dx/dt)
    dx_dt = Dynamics_casadi(t, xk, param, uk)

    # # Constraint: Yaw rate limits for chief and deputy (dx/dt[12] and dx/dt[13])
    # opti.subject_to(dx_dt[12] >= param["PHI_DOT"][0])
    # opti.subject_to(dx_dt[12] <= param["PHI_DOT"][1])
    # opti.subject_to(dx_dt[13] >= param["PHI_DOT"][0])
    # opti.subject_to(dx_dt[13] <= param["PHI_DOT"][1])

    # # # # Constraint: Yaw angle limits for chief and deputy (x[12] and x[13])
    # opti.subject_to(xk[12] >= param["PHI"][0])
    # opti.subject_to(xk[12] <= param["PHI"][1])
    # opti.subject_to(xk[13] >= param["PHI"][0])
    # opti.subject_to(xk[13] <= param["PHI"][1])

    # # Control input constraints (torque limits)
    opti.subject_to(uk[0] >= param["T_MAX"][0])  # Chief torque lower bound
    opti.subject_to(uk[0] <= param["T_MAX"][1])  # Chief torque upper bound
    opti.subject_to(uk[1] >= param["T_MAX"][0])  # Deputy torque lower bound
    opti.subject_to(uk[1] <= param["T_MAX"][1])  # Deputy torque upper bound

    # # phi_deputy = con_chief_deputy_vec(xk, param)
    # # opti.subject_to(phi_deputy - xk[13] <= 0.1)

# Initial condition constraints
opti.subject_to(X[:, 0] == yy_o)


# Set initial guesses for the optimization problem
opti.set_initial(X, np.concatenate((yy_ref,l_colm),axis =1))
opti.set_initial(U, np.zeros((2, N)))

# Solver options



print("Optimization problem set up...")
opti.solver('ipopt', {
    'expand': True,
    'ipopt.print_level': 4,  # Detailed print level
    # 'ipopt.max_iter': N_points,  # Max iterations
    'ipopt.tol': 1e-10,  # Convergence tolerance
    'print_time': True,  # Print computation time
    'ipopt.sb': 'yes',  # Show solver's internal message
    
})
print("Solver options set.")
print("Solving the problem...")

# Solve the problem
try:
    sol = opti.solve()

    # Check solver status
    if opti.stats()['success']:

        # Record the end time
        end_time = time.time()

        # Calculate the duration
        duration = end_time - start_time
        print(f"Time taken to solve the problem: {duration:.2f} seconds")

        print("Solver was successful.")
        # # Extract solution
        solution_x = sol.value(X)
        solution_u = sol.value(U)
        np.save("solution_x_casadi_opt_200_uub.npy",solution_x)
        np.save("solution_u_casadi_opt_200_uub.npy",solution_u )
    else:
        # Record the end time
        end_time = time.time()

        # Calculate the duration
        duration = end_time - start_time
        print(f"Time taken to solve the problem: {duration:.2f} seconds")

        print("Solver failed to find a solution.")
        exit()


    # print("Solution extracted.")

except RuntimeError as e:

    # Record the end time
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time
    print(f"Time taken to solve the problem: {duration:.2f} seconds")
    print("Solver encountered an error:", str(e))
    print("Solver encountered an error:", str(e))
    # Debugging - retrieve the latest values of the variables at the point of failure
    X_value = opti.debug.value(X)
    U_value = opti.debug.value(U)
    print("State variables at failure:", X_value)
    print("Control variables at failure:", U_value)


##################################

# # solution_x = np.load("solution_x_casadi_opt_200_uub.npy")
# # solution_u = np.load("solution_u_casadi_opt_200_uub.npy")

# # # # pplot control inputs
# # # print(max(solution_u[1, :]))
# # # print(min(solution_u[1, :]))

# # # Ensure solution_x and solution_u are defined
# # if solution_x is not None and solution_u is not None:
# #     # Plotting the results
# #     time_grid = np.linspace(0, T, N + 1)


# #     # Plot semi-major axis of the chief
# #     plt.figure()
# #     plt.plot(time_grid, solution_x[6, :], label='Chief semi-major axis')
# #     plt.xlabel('Time (s)')
# #     plt.ylabel('Semi-major axis')
# #     plt.title('Semi-major axis over time')
# #     plt.legend()

# #     # Plot yaw angles
# #     plt.figure()
# #     plt.plot(time_grid, solution_x[12, :], label='Chief yaw angle')
# #     plt.plot(time_grid, solution_x[13, :], label='Deputy yaw angle')
# #     plt.xlabel('Time (s)')
# #     plt.ylabel('Yaw angle')
# #     plt.title('Yaw angles over time')
# #     plt.legend()

# #     rr_s = np.zeros((3, len(solution_x[12, :])))
# #     angle_con_array = numpy.zeros(len(solution_x[12,:]))
# #     # For each time step, compute the position
# #     for i in range(len(solution_x[12, :])):
# #         yy1 = solution_x[0:6, i]  # Deputy NSROE
# #         yy2 = solution_x[6:12, i]  # Chief NSROE
# #         rr_s[:, i] = NSROE2LVLH(yy1, yy2, param)
# #         angle_con_array[i] = con_chief_deputy_angle(solution_x[:,i], param)

# #     # Plot the LVLH frame trajectory in 3D
# #     fig = plt.figure(figsize=(12, 6))
# #     ax1 = fig.add_subplot(121, projection='3d')
# #     ax1.plot3D(rr_s[0, :], rr_s[1, :], rr_s[2, :], 'black', linewidth=2, alpha=1, label='Deputy 1')
# #     ax1.set_xlabel('x (km)')
# #     ax1.set_ylabel('y (km)')
# #     ax1.set_zlabel('z (km)')
# #     ax1.set_title('LVLH frame - Deputy Spacecraft (Interactive)')
# #     ax1.legend(loc='best')

# #     # Plot the constraints angle over time
# #     fig, axs = plt.subplots(3, 1)
# #     axs[0].plot(time_grid, solution_x[0, :])
# #     axs[0].set_title('x')

# #     axs[1].plot(time_grid, solution_x[1, :])
# #     axs[1].set_title('y')

# #     axs[2].plot(time_grid, solution_x[2, :])
# #     axs[2].set_title('z')

# #     # Plot semi-major axis, mean true latitude, inclination
# #     fig, axs = plt.subplots(3, 1)
# #     axs[0].plot(time_grid, solution_x[0, :], label='semi-major axis')
# #     axs[1].plot(time_grid, solution_x[1, :], label='mean true latitude')
# #     axs[2].plot(time_grid, solution_x[2, :], label='inclination')

# #     axs[0].set_title('Semi-major axis')
# #     axs[1].set_title('Mean true latitude')
# #     axs[2].set_title('Inclination')

# #     plt.tight_layout()
# #     plt.show()

# #     # Plot q1, q2, right ascension of ascending node over time
# #     fig, axs = plt.subplots(3, 1)
# #     axs[0].plot(time_grid, solution_x[3, :], label='q1')
# #     axs[1].plot(time_grid, solution_x[4, :], label='q2')
# #     axs[2].plot(time_grid, solution_x[5, :], label='RAAN')

# #     axs[0].set_title('q1')
# #     axs[1].set_title('q2')
# #     axs[2].set_title('Right Ascension of Ascending Node')

# #     plt.tight_layout()
# #     plt.show()

# #     # Plot yaw angles
# #     fig, axs = plt.subplots(3, 1)
# #     axs[0].plot(time_grid, solution_x[12, :], label='Chief yaw angle')
# #     axs[0].set_title('Chief yaw angle')

# #     axs[1].plot(time_grid, solution_x[13, :], label='Deputy 1 yaw angle')
# #     axs[1].set_title('Deputy 1 yaw angle')

# #     axs[2].plot(time_grid, angle_con_array, label='Constraints angle')
# #     axs[2].set_title('Constraints angle')

# #     plt.tight_layout()





# #     plt.figure()
# #     plt.plot(time_grid[:-1], solution_u[0, :], label='Chief torque')
# #     plt.plot(time_grid[:-1], solution_u[1, :], label='Deputy 1 torque')
# #     plt.xlabel('Time (s)')
# #     plt.ylabel('Torque Nm')
# #     plt.legend()
# #     plt.title('Control inputs over time')
# #     plt.tight_layout()

# #     plt.show()

# #     print("Plots generated.")

# # else:
# #     print("No solution to plot.")