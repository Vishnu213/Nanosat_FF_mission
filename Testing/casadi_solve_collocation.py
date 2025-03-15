import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
from scipy.special import legendre, roots_legendre
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
    "T_MAX": [-23e-6, 23e-6],  # Maximum torque (Nm)
    "PHI_DOT": [-0.1, 0.1],  # Limits for yaw rate (rad/s)
    # "PHI": [-ca.pi / 2, ca.pi / 2],  # Limits for yaw angle (rad)
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

rho_1 = 1 # [m]  - radial separation
rho_3 =1 # [m]  - cross-track separation
alpha = 0*45* deg2rad#180 * deg2rad  # [rad] - angle between the radial and along-track separation
beta = alpha + 90 * deg2rad # [rad] - angle between the radial and cross-track separation
vd = 0.000 #-10 # Drift per revolutions m/resolution
d= -1# [m] - along track separation
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




# draw the unit vectors of the ECI frame on the 3d plot of earth
scaling_state = [6375]*16
scaling_state_inv = [1/6375]*16

for i in range(len(scaling_state)):
    if i == 0:
        scaling_state[i] = 1
        scaling_state_inv[i] = 1
    elif i == 6:
        scaling_state[i] = 1
        scaling_state_inv[i] = 1
    else:
        scaling_state[i] = 6375.0
        scaling_state_inv[i] = 1/6375.0

    

T = ca.diag(scaling_state)  # Transformation matrix
S = ca.diag([23e5, 235])  # control scaling
T_inv = ca.diag(scaling_state_inv)  # Inverse transformation matrix
S_inv = ca.diag([1/23e5, 1/23e5])  # Inverse control scaling
param['T']  = T
param['S']  = S
param['T_inv']  = T_inv
param['S_inv']  = S_inv


# statement matrix [RNOE_0,NOE_chief,yaw_c_d]
# [6x1,6x1,4x1]
yy_o=numpy.concatenate((RNOE_0,NOE_chief,yaw_c_d))

# Test for Gauss equation
mu = param["Primary"][0]
N_points = 100
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.05 * 365 * 24 * 60 * 60 / Torb
n_revolution =1 # n_revol_T
T_total = n_revolution * Torb

t_span = [0, T_total]
teval = numpy.linspace(0, T_total, N_points)

# Load the function
with open('state_vector_function.pkl', 'rb') as f:
    state_vector_function = pickle.load(f)


# Interpolate the control inputs
# Load the function
with open('control_vector_function.pkl', 'rb') as f:
    control_vector_function = pickle.load(f)

yy_ref = numpy.zeros((16, len(teval)))
uu_ref = numpy.zeros((2, len(teval)))
for i in range(len(teval)):
    yy_ref[:, i] = state_vector_function(teval[i])
    uu_ref[:, i] = control_vector_function(teval[i])

print("yy_ref>size",yy_ref.shape, yy_ref[:,-1].shape)
l_colm = yy_ref[:,-1].reshape(-1,1)
a= np.concatenate((yy_ref,l_colm),axis =1)
print("a>size",a.shape)
print("State vector interpolator loaded.")
print("size of yy_ref",yy_ref.shape)
print("size of l_colm",l_colm.shape)
print("control vector reference shapre.",uu_ref.shape)



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
T_times = T_total
dt = T_times / N

# Define tolerance values
tol1 = 0.2
tol2 = 0.2
d_koz = 0.3  # Collision avoidance distance
r0 = np.linalg.norm(NSROE2LVLH(yy_o[0:6], yy_o[6:12], param))
start_time = time.time()


# Define the number of collocation points
N_collocation = 5  # Number of collocation points per interval
N_intervals = N_points  # Number of intervals (same as before)

# Get Gauss-Lobatto collocation points and weights
tau_root = np.polynomial.legendre.leggauss(N_collocation - 1)[0]  # Roots of Legendre polynomial
tau_root = np.append(-1, tau_root)  # Include the left endpoint for Gauss-Lobatto
tau_root = (tau_root + 1) / 2  # Shift to [0, 1] for Gauss-Lobatto


def compute_quadrature_weights(tau_root):
    """
    Compute the quadrature weights for Gauss-Lobatto collocation points.
    """
    N = len(tau_root)
    B = np.zeros(N)

    # Symbolic variable for evaluation
    tau = ca.MX.sym('tau')

    # Lagrange basis functions
    L = lagrange_basis(tau, tau_root)

    # Compute quadrature weights
    for j in range(N):
        # Integrate the Lagrange polynomial
        pint = ca.integrator('pint', 'cvodes', {'x': tau, 'ode': L[j]}, {'t0': 0, 'tf': 1})
        result = pint(x0=0)
        B[j] = result['xf']

    return B


# Lagrange polynomial basis functions using CasADi
def lagrange_basis(tau, tau_root):
    N = len(tau_root)
    L = []
    for i in range(N):
        l = 1.0
        for j in range(N):
            if j != i:
                l *= (tau - tau_root[j]) / (tau_root[i] - tau_root[j])
        L.append(l)
    return L

def collocation_matrix(tau_root):
    N = len(tau_root)
    C = ca.MX.zeros(N, N)  # Collocation matrix
    D = ca.MX.zeros(N, N)  # Differentiation matrix
    B = np.zeros(N)        # Quadrature weights

    # Symbolic variable for evaluation
    tau = ca.MX.sym('tau')

    # Lagrange basis functions
    L = lagrange_basis(tau, tau_root)

    # Compute C, D, and B matrices
    for j in range(N):
        for k in range(N):
            # Evaluate Lagrange polynomial at collocation points
            C[j, k] = ca.substitute(L[k], tau, tau_root[j])

            # Compute derivative of Lagrange polynomial
            dL = ca.jacobian(L[k], tau)
            D[j, k] = ca.substitute(dL, tau, tau_root[j])

        # Compute quadrature weights
        pint = ca.integrator('pint', 'cvodes', {'x': tau, 'ode': L[j]}, {'t0': 0, 'tf': 1})
        result = pint(x0=0)
        B[j] = result['xf']

    return C, D, B

# Extend the differentiation matrix to handle the state vector
def extend_differentiation_matrix_C_D(C,D, state_size):
    """
    Extend the differentiation matrix D to handle a state vector of size `state_size`.
    """
    N = D.size1()  # Number of collocation points
    D_ext = ca.kron(D, ca.MX.eye(state_size))  # Kronecker product with identity matrix
    M = C.size1()  # Number of collocation points
    C_ext = ca.kron(C, ca.MX.eye(state_size))  # Kronecker product with identity matrix
    return C_ext, D_ext


N_collocation = 5  # Number of collocation points
state_size = 16  # Size of the state vector

# Get Gauss-Lobatto collocation points
tau_root = np.polynomial.legendre.leggauss(N_collocation - 1)[0]  # Roots of Legendre polynomial
tau_root = np.append(-1, tau_root)  # Include the left endpoint for Gauss-Lobatto
tau_root = (tau_root + 1) / 2  # Shift to [0, 1] for Gauss-Lobatto

# Compute collocation and differentiation matrices
C, D, B = collocation_matrix(tau_root)

# Extend the differentiation matrix to handle the state vector
C_ext,D_ext = extend_differentiation_matrix_C_D(C,D, state_size)

# # Print the extended differentiation matrix
# print("Extended differentiation matrix (D_ext):")
# print(D_ext)


print("Collocation matrices computed.")


# # Define the Opti problem
# opti = ca.Opti()

# # Compute collocation matrices
# C, D = collocation_matrix(tau_root)

# Define the Opti problem
opti = ca.Opti()

# State variables (16 states: 6 NSROE deputy, 6 NSROE chief, 2 yaw angles)
X = opti.variable(16, N_intervals + 1)

# Control variables (2 control variables for yaw dynamics)
U = opti.variable(2, N_intervals)

# Collocation states (16 states at each collocation point)
X_coll = opti.variable(16 * N_collocation, N_intervals)  # 2D variable

# Time grid
t = opti.parameter(N_intervals + 1)
t_value = np.linspace(0, T_total, N_intervals + 1)
opti.set_value(t, t_value)

# Initial condition constraints
opti.subject_to(X[:, 0] == yy_o)

print("Initial condition constraints set.")

# Collocation constraints
for k in range(N_intervals):
    # Control at the collocation points
    uc = U[:, k]

    # Reshape the state vector into a column vector of size (N_collocation * 16 x 1)
    xc_k = ca.reshape(X_coll[:, k], (N_collocation * state_size, 1))

    # Compute the derivative of the state at the collocation points
    dx_dt_k = ca.mtimes(D_ext, xc_k)  # D_ext is (N_collocation * 16 x N_collocation * 16), xc_k is (N_collocation * 16 x 1)

    # Reshape the derivative into a matrix of size (N_collocation x 16)
    dx_dt_k = ca.reshape(dx_dt_k, (N_collocation, state_size))

    # Enforce the dynamics at each collocation point
    for j in range(N_collocation):
        # State at the j-th collocation point
        xc_jk = xc_k[j * state_size : (j + 1) * state_size]  # Extract the j-th block (size 16 x 1)

        # State derivative at the j-th collocation point
        dx_dt_jk = dx_dt_k[j, :]  # Extract the j-th row (size 1 x 16)

        # Dynamics constraint
        opti.subject_to(dx_dt_jk == Dynamics_casadi(t[k] + tau_root[j] * dt, xc_jk, param, uc).T)

    # Continuity constraint between intervals
    opti.subject_to(X[:, k + 1] == ca.mtimes(C_ext[-state_size:,:], X_coll[:, k]))

# # Objective function
# control_cost = 0
# objective2 = 0
epsilon = 1e-7
# Define the weights
weight1 = 0.4  # Weight for the first objective
weight2 = 0.6 # Weight for the second objective


# print("Objective function set.")

# for k in range(N_intervals):
#     control_cost += epsilon * ca.sumsqr(U[:, k])
#     objective2 += con_chief_deputy_angle_casadi(ca.mtimes(T_inv, X[:, k]), param)**2

# # Combined objective
# objective1 = -X[0, -1]
# combined_objective = weight1 * objective1 + weight2 * objective2 + control_cost


# Initialize the objective function
control_cost = 0
objective2 = 0

# Loop over intervals
for k in range(N_intervals):
    # Control cost
    #control_cost += epsilon * ca.sumsqr(U[:, k])

    # State-dependent cost at collocation points
    for j in range(N_collocation):
        # State at the j-th collocation point in the k-th interval
        xc_jk = X_coll[j * state_size : (j + 1) * state_size, k]

        # Evaluate the state-dependent cost
        cost_jk = con_chief_deputy_angle_casadi(ca.mtimes(T_inv, xc_jk), param)**2

        # Add contribution to the objective function using quadrature weights
        objective2 += B[j] * cost_jk * dt

# Combined objective
objective1 = -X[0, -1]
combined_objective = weight1 * objective1 # + weight2 * objective2 + control_cost

print("size of l_colm",l_colm.shape)
print("concatenated",np.concatenate((yy_ref,l_colm),axis =1).shape)
print("multiple",np.dot(T,np.concatenate((yy_ref,l_colm),axis =1)).shape)

# yy_init = np.dot(T,np.concatenate((yy_ref,l_colm)))
# Set initial guesses for the optimization problem
opti.set_initial(X,np.concatenate((yy_ref,l_colm),axis =1))
opti.set_initial(U, uu_ref)

opti.minimize(combined_objective)


print("Objective function set.")

# Solver options
opti.solver('ipopt', {
    'ipopt.print_level': 5,
    'ipopt.sb': 'yes',
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-5,
    'expand': True,
})

# Solve the problem
sol = opti.solve()

print("Problem solved.")

if sol.stats()['success']:
    print("Optimization successful.")
else:
    print("Optimization failed.")
# Extract solution
solution_x = sol.value(X)
solution_u = sol.value(U)

# Save results
np.save("solution_x_casadi_opt_100_gauss_lobatto.npy", solution_x)
np.save("solution_u_casadi_opt_100_gauss_lobatto.npy", solution_u)