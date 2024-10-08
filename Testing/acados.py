import numpy as np
import numpy
import casadi as ca
import os
import sys
import matplotlib.pyplot as plt

from acados_template import AcadosOcp, AcadosOcpSolver


# Add the folder containing your modules to the Python path
path_core = "../core"
path_casadi_converter = "../Casadi_modules"

# Get absolute paths
module_path_core = os.path.abspath(path_core)
module_path_casadi_converter = os.path.abspath(path_casadi_converter)

# Check if the paths are not already in sys.path and add them
if module_path_core not in sys.path:
    sys.path.append(module_path_core)

if module_path_casadi_converter not in sys.path:
    sys.path.append(module_path_casadi_converter)

# Load the CasADi versions of the functions
from converted_functions import Dynamics_casadi, NSROE2LVLH_casadi, con_chief_deputy_angle_casadi
from TwoBP import Param2NROE, M2theta



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
    "T_MAX": [0, 23e-6],  # Maximum torque (Nm)
    "PHI_DOT": [0.0, 0.1],  # Limits for yaw rate (rad/s)
    "PHI": [-np.pi / 2, np.pi / 2]  # Limits for yaw angle (rad)
}




print("Parameters initialized.")

deg2rad = numpy.pi / 180

# CHECK Formation Establishment and Reconfiguration Using
# Differential Elements in J2-Perturbed Orbits and SENGUPTA
# Chaser spacecraft initial conditions
# orbital elements - non singular



# Deputy spacecraft relative orbital  elements/ LVLH initial conditions
# NOE_chief = numpy.array([a,lambda_0,i,q1,q2,omega])
NOE_chief = numpy.array([6500,0.1,63.45*deg2rad,0.5,0.2,270.828*deg2rad]) # numpy.array([6803.1366,0,97.04,0.005,0,270.828])
## MAKE SURE TO FOLLOW RIGHT orbital elements order
 

    # assigning the state variables
a =NOE_chief[0]
l =NOE_chief[1]
i =NOE_chief[2]
q1 =NOE_chief[3]
q2 =NOE_chief[4]
OM =NOE_chief[5]
mu = param["Primary"][0]


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

# Design parameters for the formation - Sengupta and Vadali 2007 Relative Motion and the Geometry of Formations in Keplerian Elliptic Orbits

rho_1 = 0 # [m]  - radial separation 
rho_3 =0 # [m]  - cross-track separation
alpha = 0#180 * deg2rad  # [rad] - angle between the radial and along-track separation
beta = 0#alpha + 90 * deg2rad # [rad] - angle between the radial and cross-track separation
vd = 0 #-10 # Drift per revolutions m/resolution
d= 1# [m] - along track separation
rho_2 = (2*(eta**2) * d) /(3-eta**2) # [m]  - along-track separation
print("RHO_2",rho_2)
print(d/1+e, d/1-e,  d*(1/(2*(eta**2)) /(3-eta**2)))
parameters=numpy.array([rho_1,rho_2,rho_3,alpha,beta,vd])

print("Formation parameters",parameters)
# Initial relative orbital elements
RNOE_0=Param2NROE(NOE_chief, parameters,param)
# RNOE_0[0]=0
# RNOE_0[2]=-RNOE_0[5]*numpy.cos(NOE_chief[2]) 

# angle of attack for the deputy spacecraft
yaw_1 = 10*deg2rad  # [rad] - angle of attack = 0 assumption that V_sat = V_rel
yaw_2 = 10*deg2rad  # [rad] - angle of attack = 0
yaw_c_d=numpy.array([yaw_1,yaw_2])

print("RELATIVE ORBITAL ELEMTNS INITIAL", RNOE_0)
print("CHIEF INTIIAL ORBITAL ELEMENTS", NOE_chief)

 
# statement matrix [RNOE_0,NOE_chief,yaw_c_d]
# [6x1,6x1,2x1]
yy_o=numpy.concatenate((RNOE_0,NOE_chief,yaw_c_d))


# test for gauess equation
mu=param["Primary"][0]
Torb = 2*numpy.pi*numpy.sqrt(NOE_chief[0]**3/mu)    # [s]    Orbital period
n_revol_T = 0.0005*365*24*60*60/Torb
n_revolution=  0.01 #n_revol_T
T_total=n_revolution*Torb

t_span=[0,T_total]
teval=numpy.linspace(0, T_total, 1000)
# K=numpy.array([k1,k2])

# Simulate
uu_o = np.zeros((2, 1))  # Control inputs

param["Init"] = [NOE_chief[4], NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0


# Define the symbolic variables for CasADi integration
t = ca.MX.sym('t')  # Time
yy = ca.MX.sym('yy', 14)  # 14 state variables (6 NSROE for deputy, 6 NSROE for chief, 2 for yaw)
uu = ca.MX.sym('uu', 2)   # Control inputs for yaw dynamics




# Setup tolerances and other required variables
tol1 = 0.2  # Define appropriate values
tol2 = 0.2
d_koz = 0.3  # Define collision avoidance distance

# Setup the optimization problem with Acados
ocp = AcadosOcp()

ocp.model.name = "Nano_sat_dynamics"

# Dynamics function in CasADi
dynamics_casadi_sym = Dynamics_casadi(t, yy, param, uu)
dynamics_function = ca.Function('dynamics', [t, yy, uu], [dynamics_casadi_sym])

# Constraint 1: tol2 - r < r < r + tol1
r = NSROE2LVLH_casadi(yy[0:6], yy[6:12], param)
constraint1_lower = tol2 - r
constraint1_upper = r - tol1

# Constraint 2: d_koz < r
constraint2 = d_koz - r

# Constraint 3 & 4: phi_dot range constraints for chief and deputy
constraint3 = ca.vertcat(param["PHI_DOT"][0] - yy[12], yy[12] - param["PHI_DOT"][1])
constraint4 = ca.vertcat(param["PHI_DOT"][0] - yy[13], yy[13] - param["PHI_DOT"][1])

# Constraint 5 & 6: yaw angle range constraints for chief and deputy
constraint5 = ca.vertcat(param["PHI"][0] - yy[12], yy[12] - param["PHI"][1])
constraint6 = ca.vertcat(param["PHI"][0] - yy[13], yy[13] - param["PHI"][1])

# Constraint 7 & 8: control input torque constraints
constraint7 = ca.vertcat(param["T_MAX"][0] - uu[0], uu[0] - param["T_MAX"][1])
constraint8 = ca.vertcat(param["T_MAX"][0] - uu[1], uu[1] - param["T_MAX"][1])

# Constraint 9: x_dot = f(x,u,t) - Dynamics
constraint9 = ca.vertcat(dynamics_casadi_sym - dynamics_function(t, yy, uu))

# Constraint 10: phi_deputy = con_deputy_angle_casadi(state_vector, param)
phi_deputy = con_chief_deputy_angle_casadi(yy, param)
constraint10 = phi_deputy

# Add all constraints to the problem
constraints = [
    constraint1_lower, constraint1_upper,
    constraint2,
    constraint3, constraint4,
    constraint5, constraint6,
    constraint7, constraint8,
    constraint9,
    constraint10
]
# for con in constraints:
#     ocp.constraints.constr_expr_h(con)

ocp.model.constr_expr_h = ca.vertcat(*constraints)

# Objective function: minimize -yy[6] (minimizing the semi-major axis of chief)
objective = -yy[6]
ocp.cost.expr_ext_cost_e = objective

# Define states and controls in the model
ocp.model.x = yy  # State vector
ocp.model.u = uu  # Control inputs

# Define the dynamics in the model
ocp.model.f_expl_expr = dynamics_casadi_sym


# Time discretization settings for multiple shooting
N_shooting_intervals = 40  # Number of intervals
time_horizon = T_total     # Total time horizon
t_grid = np.linspace(0, time_horizon, N_shooting_intervals + 1)  # Discretization grid
ocp.dims.N = N_shooting_intervals  # Set the number of shooting nodes

# Set up the ocp_solver options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # Set the QP ocp_solver type
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # Hessian approximation
ocp.solver_options.integrator_type = 'ERK'  # Implicit Runge-Kutta
ocp.solver_options.nlp_solver_type = 'SQP'  # Sequential Quadratic Programming
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'  # Globalization method




# Set up multiple shooting
ocp.solver_options.nlp_solver_steps =  N_shooting_intervals   # Number of multiple shooting nodes

# Time horizon for solving
ocp.solver_options.tf = time_horizon

# # Export ocp_solver and set options
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")


# Initialize state and control vectors at each node (multiple shooting nodes)
initial_state = yy_o  # Initial state vector
initial_control = uu_o  # Initial control vector (set this according to the problem)

for i in range(N_shooting_intervals + 1):
    ocp_solver.set(i, "x", initial_state)
    if i < N_shooting_intervals:
        ocp_solver.set(i, "u", initial_control)

ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
# Enable verbose output
ocp_solver.options_set('print_level', 2)  # Adjust the print level for more details

# Solve the problem
status = ocp_solver.solve()

# Check if the solution is successful
if status != 0:
    raise RuntimeError(f"Acados failed to solve the problem with status {status}")
else:
    print("SUCESSSSSSSSSSSSS")



# Retrieve the solution
solution_states = np.array([ocp_solver.get(i, "x") for i in range(N_shooting_intervals + 1)])
solution_controls = np.array([ocp_solver.get(i, "u") for i in range(N_shooting_intervals)])

# Check the number of iterations the ocp_solver performed
num_iterations = ocp_solver.get_stats("qp_iter")
print(f"ocp_solver completed in {num_iterations} iterations")

# Check the residuals
residuals = ocp_solver.get_stats("residuals")
print(f"Residuals: {residuals}")


print("Solution states:", solution_states)
print("Solution controls:", solution_controls)




# # set ocp_solver options
# ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # ocp_solver options for quadratic programming
# # Alternative choices include:
# # 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 
# # 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP'

# ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # Hessian approximation
# # Can also use 'EXACT' for more accuracy, but 'GAUSS_NEWTON' is typically faster.

# ocp.solver_options.integrator_type = 'IRK'  # Implicit Runge-Kutta (IRK) for better accuracy on nonlinear systems.
# # Other options include 'ERK' (Explicit Runge-Kutta), but 'IRK' is usually more stable for stiff systems.

# ocp.solver_options.nlp_solver_type = 'SQP'  # Sequential Quadratic Programming (SQP) for accurate solutions.
# # Alternatively, 'SQP_RTI' (Real-Time Iteration) for faster but less accurate solutions.

# ocp.solver_options.globalization = 'MERIT_BACKTRACKING'  # Globalization strategy to ensure convergence.
# # MERIT_BACKTRACKING is used to improve convergence, especially with non-smooth problems.