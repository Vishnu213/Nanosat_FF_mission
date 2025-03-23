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

T_value = np.diag(scaling_state)
T_inv_value = np.diag(scaling_state_inv)
S_value = np.diag([23e5, 235])
S_inv_value = np.diag([1/23e5, 1/23e5])


# statement matrix [RNOE_0,NOE_chief,yaw_c_d]
# [6x1,6x1,4x1]
yy_o=numpy.concatenate((RNOE_0,NOE_chief,yaw_c_d))

# Test for Gauss equation
mu = param["Primary"][0]
N_points = 10
Torb = 2 * numpy.pi * numpy.sqrt(NOE_chief[0]**3 / mu)  # [s] Orbital period
n_revol_T = 0.05 * 365 * 24 * 60 * 60 / Torb
n_revolution =10 # n_revol_T
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

##################################

if __name__ == "__main__":

    x_state = np.load("./../helper_files/solution_x_casadi_opt_100_noconstrains_12_10_2024.npy")
    print("x_state",x_state.shape)

    x = np.zeros((16, x_state.shape[1]))

    # for i in range(x_state.shape[1]):
    #     x[:, i] = np.matmul(T_inv_value, x_state[:, i])

    x = T_inv_value @ x_state

    print("x",x[:,-1])
    


    #x = np.matmul(T_inv_value, np.load("solution_x_casadi_opt_100_noconstrains_12_10_2024.npy"))
    u =np.load("./../helper_files/solution_u_casadi_opt_100_noconstrains_12_10_2024.npy") #  np.matmul(S_inv_value, 
    time_points = np.load("./../helper_files/solution_t_casadi_opt_100_noconstrains_12_10_2024.npy")

    ## shape of the x and u

    print("shape of x",x.shape)
    print("shape of u",u.T.shape)
    print("shape of time_points",time_points.shape)

    # Create interpolation functions
    state_interp = interp1d(time_points, x.T, kind='cubic', axis=0)  # Cubic interpolation
    print(u.T, time_points)
    print("first element ", u[:,0], numpy.array([0,0]).shape)
    print("Shape", u.shape,)
    u = np.concatenate((u, np.zeros((2, 1))), axis=1)  # Add an extra control input
    print(u.shape, time_points.shape)
    control_interp = interp1d(time_points, u.T, kind='cubic', axis=0)

    time_grid = np.linspace(0, time_points[-1], num=100)  # 1000 points between t0 and tN
    solution_x =  T_inv_value @ state_interp(time_grid).T
    print("solution_x",solution_x.shape)
    print("solution_x",solution_x)
    solution_u = control_interp(time_grid)

    # # pplot control inputs
    # print(max(solution_u[1, :]))
    # print(min(solution_u[1, :]))

    # Ensure solution_x and solution_u are defined
    if solution_x is not None and solution_u is not None:
        # Plotting the results
        # time_grid = np.linspace(0, T, N + 1)

        print("Shapes", solution_x.shape, solution_u.shape)
        # Plot semi-major axis of the chief
        plt.figure()
        plt.plot(time_grid, solution_x[6,:], label='Chief semi-major axis')
        plt.xlabel('Time (s)')
        plt.ylabel('Semi-major axis')
        plt.title('Semi-major axis over time')
        plt.legend()



        rr_s = np.zeros((3, len(solution_x[12,:])))
        angle_con_array = numpy.zeros(len(solution_x[12,:]))
        # For each time step, compute the position
        for i in range(len(solution_x[12,:])):
            yy1 = solution_x[0:6,i]  # Deputy NSROE
            yy2 = solution_x[6:12,i]  # Chief NSROE
            print("yy1",yy1,range(len(solution_x[12,:])) )
            rr_s[:, i] = NSROE2LVLH(yy1, yy2, param)
            print("yy1",yy2,range(len(solution_x[:, 12])) )
            rr_s[:, i] = NSROE2LVLH(yy1, yy2, param)
            angle_con_array[i] = con_chief_deputy_angle(solution_x[:,i], param)

        # Plot the LVLH frame trajectory in 3D
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot3D(rr_s[0, 0], rr_s[1, 0], rr_s[2, 0], 'green', marker='o', markersize=5, label='Initial')
        ax1.plot3D(rr_s[0, :], rr_s[1, :], rr_s[2, :], 'black', linewidth=2, alpha=1, label='Deputy 1')
        ax1.plot3D(rr_s[0, -1], rr_s[1, -1], rr_s[2, -1], 'red', marker='o', markersize=5, label='Final')
        ax1.plot3D(0, 0, 0, 'blue', marker='o', markersize=5, label='chief')
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
        axs[0].plot(time_grid, solution_x[0,:], label='semi-major axis')
        axs[1].plot(time_grid, solution_x[1,:], label='mean true latitude')
        axs[2].plot(time_grid, solution_x[2,:], label='inclination')

        axs[0].set_title('Semi-major axis')
        axs[1].set_title('Mean true latitude')
        axs[2].set_title('Inclination')

        plt.tight_layout()


        # Plot q1, q2, right ascension of ascending node over time
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(time_grid, solution_x[3,:], label='q1')
        axs[1].plot(time_grid, solution_x[4,:], label='q2')
        axs[2].plot(time_grid, solution_x[5,:], label='RAAN')

        axs[0].set_title('q1')
        axs[1].set_title('q2')
        axs[2].set_title('Right Ascension of Ascending Node')

        plt.tight_layout()


        # Plot yaw angles
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(time_grid, solution_x[12,:], label='Chief yaw angle')
        axs[0].set_title('Chief yaw angle')

        axs[1].plot(time_grid, solution_x[13,:], label='Deputy 1 yaw angle')
        axs[1].set_title('Deputy 1 yaw angle')

        # axs[2].plot(time_grid, angle_con_array, label='Constraints angle')
        # axs[2].set_title('Constraints angle')

        plt.tight_layout()





        print("Shape controls", solution_u.shape)
        print("time grid", time_grid.shape)
        plt.figure()
        plt.plot(time_grid, solution_u[:,0], label='Chief torque')
        plt.plot(time_grid, solution_u[:,1], label='Deputy 1 torque')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque Nm')
        plt.legend()
        plt.title('Control inputs over time')
        plt.tight_layout()

        plt.show()

        print("Plots generated.")

    else:
        print("No solution to plot.")