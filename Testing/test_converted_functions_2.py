import casadi as ca
import numpy as np



# Define the symbolic variables for time t, state yy, and control input uu
t = ca.SX.sym('t')
yy = ca.SX.sym('yy', 14)  # 14 state variables (6 NSROE for deputy, 6 NSROE for chief, 2 for yaw)
uu = ca.SX.sym('uu', 2)   # 2 control inputs for yaw dynamics

# Load polynomials and scalers from the provided files
loaded_polynomials = load_polynomials('path_to_polynomials.pkl')
scaler_min = np.array([value1, value2, value3])  # Replace with actual values from your scaler
scaler_max = np.array([value4, value5, value6])  # Replace with actual values from your scaler
target_scaler_min = np.array([target_value1, target_value2, target_value3])
target_scaler_max = np.array([target_value4, target_value5, target_value6])

# Define your parameters (same as you did in the Python version)
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
}

# Define the CasADi function for dynamics using your converted `Dynamics_casadi`
dynamics_function = ca.Function('dynamics', [t, yy, uu], [Dynamics_casadi(t, yy, param, uu, loaded_polynomials)])

# Set up the integrator in CasADi
integrator = ca.integrator('integrator', 'cvodes', {'x': yy, 't': t, 'ode': dynamics_function(yy)}, {'t0': 0, 'tf': T_total})

# Initialize the state `yy_o` based on your setup
yy_o = np.concatenate((RNOE_0, NOE_chief, yaw_c_d))

# Run the CasADi integration over the time span
result = integrator(x0=yy_o, p=uu)

# Extract the integrated states from the result
integrated_states = result['xf']



