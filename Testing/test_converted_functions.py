# # test_converted_functions.py

# import numpy as np
# import casadi as ca

# # Import the original function
# from TwoBP import M2theta as M2theta_original

# # Import the CasADi-converted function
# # Assuming it's saved in a file named 'converted_functions.py'
# from converted_functions import M2theta_casadi

# def test_M2theta():
#     print("\nTesting M2theta_casadi() function...")
    
#     # Test inputs
#     M_test = 0.75  # Mean anomaly in radians
#     e_test = 0.1   # Eccentricity
    
#     # Print inputs
#     print(f"Input Mean Anomaly (M): {M_test}")
#     print(f"Input Eccentricity (e): {e_test}")
    
#     # Original function result
#     theta_original, _ = M2theta_original(M_test, e_test, 1e-10)
    
#     # CasADi function result
#     theta_symbolic = M2theta_casadi(M_test, e_test)
#     theta_casadi_func = ca.Function('theta_casadi', [], [theta_symbolic])
#     theta_casadi_value = theta_casadi_func().full().item()
    
#     # Print outputs
#     print(f"Original theta: {theta_original}")
#     print(f"CasADi theta: {theta_casadi_value}")
    
#     # Check if the results are close within a specified tolerance
#     tolerance = 1e-6
#     if np.isclose(theta_original, theta_casadi_value, atol=tolerance):
#         print("Test Passed: The results match within the acceptable tolerance.")
#     else:
#         print("Test Failed: The results do not match.")

# # Run the test
# if __name__ == "__main__":
#     test_M2theta()


import numpy as np
import casadi as ca
import os
import sys
import pickle
import pytest



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


from TwoBP import M2theta as M2theta_original, NSROE2car as NSROE2car_original, PQW2ECI as PQW2ECI_original

from converted_functions import M2theta_casadi, NSROE2car_casadi, PQW2ECI_casadi

from CL_CD_modified_sentman import calculate_cd_cl 
from converted_functions import calculate_cd_cl_casadi, calculate_rms_speed_casadi


# Import the original and CasADi-converted functions from the relevant files
from TwoBP import car2kep  # Original function from TwoBP.py
from converted_functions import car2kep_casadi  # CasADi version from converted functions

from TwoBP import Param2NROE
from converted_functions import Param2NROE_casadi

from TwoBP import lagrage_J2_diff
from converted_functions import lagrange_J2_diff_casadi


from Transformations import (C1, C2, C3,
    PQW2ECI, RSW2ECI,
    LVLHframe, Frenet2LVLH)

from converted_functions import (
    C1_casadi, C2_casadi, C3_casadi,
    PQW2ECI_casadi, RSW2ECI_casadi,
    LVLHframe_casadi, Frenet2LVLH_casadi
)

from dynamics import yaw_dynamics
from converted_functions import yaw_dynamics_casadi


from TwoBP import car2NNSOE_density  # Assuming this is the original NumPy version
from converted_functions import car2NNSOE_density_casadi  # CasADi version


from lift_drag import lookup_surface_properties  # Original function from TwoBP.py
from converted_functions import lookup_surface_properties_casadi


from converted_functions import calculate_aerodynamic_forces_casadi  # CasADi version
from lift_drag import calculate_aerodynamic_forces  # Python version


from converted_functions import compute_forces_for_entities_casadi  # CasADi-converted function
from lift_drag import compute_forces_for_entities  # Python version


from dynamics import absolute_NSROE_dynamics
from converted_functions import absolute_NSROE_dynamics_casadi


def test_M2theta():
    M_test = 0.75
    e_test = 0.1

    # Original function (numerical)
    theta_original, _ = M2theta_original(M_test, e_test, 1e-10)

    # Create CasADi symbolic variables for M and e
    M_sym = ca.SX.sym('M')
    e_sym = ca.SX.sym('e')

    # CasADi symbolic function
    theta_symbolic = M2theta_casadi(M_sym, e_sym)

    # Create the CasADi function that takes M and e as inputs
    theta_casadi_func = ca.Function('theta_casadi', [M_sym, e_sym], [theta_symbolic])

    # Evaluate the CasADi function with numerical inputs
    theta_casadi_value = theta_casadi_func(M_test, e_test).full().item()

    # Assert that the values are close within tolerance
    assert np.isclose(theta_original, theta_casadi_value, atol=1e-6)


def test_PQW2ECI_casadi():
    # Test inputs (angles in radians)
    OM_test = np.deg2rad(30)  # Right Ascension of the Ascending Node
    om_test = np.deg2rad(45)  # Argument of Periapsis
    i_test = np.deg2rad(60)   # Inclination

    # Original function result (NumPy)
    PQW2ECI_matrix_original = PQW2ECI_original(OM_test, om_test, i_test)

    print(f"Original PQW to ECI Matrix: \n{PQW2ECI_matrix_original}")

    # CasADi function result
    # Convert inputs to CasADi symbols
    OM_sym = ca.SX.sym('OM')
    om_sym = ca.SX.sym('om')
    i_sym = ca.SX.sym('i')

    # Create CasADi function
    PQW2ECI_matrix_sym = PQW2ECI_casadi(OM_sym, om_sym, i_sym)
    PQW2ECI_casadi_func = ca.Function('PQW2ECI_casadi_func', [OM_sym, om_sym, i_sym], [PQW2ECI_matrix_sym])

    # Evaluate the CasADi function with test inputs
    PQW2ECI_matrix_casadi = PQW2ECI_casadi_func(OM_test, om_test, i_test)
    PQW2ECI_matrix_casadi_value = PQW2ECI_matrix_casadi.full()

    print(f"CasADi PQW to ECI Matrix: \n{PQW2ECI_matrix_casadi_value}")

    # Check if the results are close within a specified tolerance
    tolerance = 1e-6
    assert np.allclose(PQW2ECI_matrix_original, PQW2ECI_matrix_casadi_value, atol=tolerance), \
        f"Original={PQW2ECI_matrix_original}, CasADi={PQW2ECI_matrix_casadi_value}"


#
def test_NSROE2car_casadi():
    # Test inputs
    ROE_test = np.array([7000, 0.1, np.deg2rad(45), 0.01, 0.02, np.deg2rad(30)])
    param_test = {
        "Primary": [398600.433, 6378.16, 7.2921150e-5],  # mu, radius, rotation rate
    }

    # Original function result
    RR_original, VV_original = NSROE2car_original(ROE_test, param_test)

    print(f"Original RR: {RR_original}")
    print(f"Original VV: {VV_original}")

    # CasADi function result
    # Convert inputs to CasADi symbols
    ROE_sym = ca.SX.sym('ROE', 6)
    param_sym = param_test  # Parameters can be used directly

    # Create CasADi function
    RR_sym, VV_sym= NSROE2car_casadi(ROE_sym, param_sym)
    NSROE2car_casadi_func = ca.Function('NSROE2car_casadi_func', [ROE_sym], [RR_sym, VV_sym])

    RR_casadi, VV_casadi = NSROE2car_casadi_func(ROE_test)

    RR_casadi_value = RR_casadi.full().flatten()
    VV_casadi_value = VV_casadi.full().flatten()

    print(f"CasADi RR: {RR_casadi_value}")
    print(f"CasADi VV: {VV_casadi_value}")

    # Check if the results are close within a specified tolerance
    tolerance = 1e-6
    assert np.allclose(RR_original, RR_casadi_value, atol=tolerance), \
        f"RR_original={RR_original}, RR_casadi={RR_casadi_value}"
    assert np.allclose(VV_original, VV_casadi_value, atol=tolerance), \
        f"VV_original={VV_original}, VV_casadi={VV_casadi_value}"



def test_calculate_cd_cl_casadi():
    # Test inputs
    S_i = 1.0         # Area of the plate (m^2)
    S_ref_i = 1000.0  # Reference area (m^2)
    gamma_i = 0.8     # Cosine of angle between velocity and normal vector
    l_i = 0.6         # Sine of angle between velocity and normal vector
    v_inc = 7500.0    # Incoming velocity (m/s)
    Ma = 28.9647      # Mean molar mass of air (g/mol)
    Ta = 1000.0       # Ambient temperature (K)
    alpha_acc = 1.0   # Accommodation coefficient

    # Original function result
    Cd_original, Cl_original = calculate_cd_cl(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc)

    # CasADi function result
    Cd_casadi_sym, Cl_casadi_sym = calculate_cd_cl_casadi(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc)

    
    # Create the CasADi function, but use actual inputs instead of an empty list `[]`
    calculate_cd_cl_casadi_func = ca.Function('calculate_cd_cl_casadi_func', 
                                              [ca.SX.sym('S_i'), ca.SX.sym('S_ref_i'), ca.SX.sym('gamma_i'), 
                                               ca.SX.sym('l_i'), ca.SX.sym('v_inc'), ca.SX.sym('Ma'), 
                                               ca.SX.sym('Ta'), ca.SX.sym('alpha_acc')], 
                                              [Cd_casadi_sym, Cl_casadi_sym])
    
    # Evaluate the function with the test inputs
    Cd_casadi_value, Cl_casadi_value = calculate_cd_cl_casadi_func(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc)
    
    # Convert CasADi output to numerical values
    Cd_casadi_value = Cd_casadi_value.full().item()
    Cl_casadi_value = Cl_casadi_value.full().item()

    # Check if the results are close within a specified tolerance
    tolerance = 1e-6
    assert np.isclose(Cd_original, Cd_casadi_value, atol=tolerance), \
        f"Cd_original={Cd_original}, Cd_casadi={Cd_casadi_value}"
    assert np.isclose(Cl_original, Cl_casadi_value, atol=tolerance), \
        f"Cl_original={Cl_original}, Cl_casadi={Cl_casadi_value}"

# Test function for car2kep_casadi
def test_car2kep_casadi():
    # Test inputs
    r_vec = np.array([7000, 0.25, 0])  # km
    v_vec = np.array([-1.2, 7, 0])   # km/s
    mu = 398600.433  # Standard gravitational parameter for Earth (km^3/s^2)

    # Original function result
    COE_original_vec, COE_original_mag = car2kep(r_vec, v_vec, mu)
    
    # CasADi function result
    r_vec_sym = ca.SX.sym('r_vec', 3)
    v_vec_sym = ca.SX.sym('v_vec', 3)
    mu_sym = ca.SX.sym('mu')
    
    COE_casadi_sym_vec, COE_casadi_sym_mag = car2kep_casadi(r_vec_sym, v_vec_sym, mu_sym)
    car2kep_casadi_func = ca.Function('car2kep_casadi_func', [r_vec_sym, v_vec_sym, mu_sym], [COE_casadi_sym_vec, COE_casadi_sym_mag])
    
    COE_casadi_value_vec, COE_casadi_value_mag = car2kep_casadi_func(r_vec, v_vec, mu)
    
    COE_casadi_value_vec = COE_casadi_value_vec.full().flatten()
    COE_casadi_value_mag = COE_casadi_value_mag.full().flatten()

    # Print the results
    print(f"Original COE_vec: {COE_original_vec}")
    print(f"CasADi COE_vec: {COE_casadi_value_vec}")
    print(f"Original COE_mag: {COE_original_mag}")
    print(f"CasADi COE_mag: {COE_casadi_value_mag}")
    
    # Assert they are close within tolerance
    tolerance = 1e-6
    assert np.allclose(COE_original_vec, COE_casadi_value_vec, atol=tolerance), \
        f"COE_vec mismatch: {COE_original_vec} vs {COE_casadi_value_vec}"
    assert np.allclose(COE_original_mag, COE_casadi_value_mag, atol=tolerance), \
        f"COE_mag mismatch: {COE_original_mag} vs {COE_casadi_value_mag}"

import numpy as np
import casadi as ca
from TwoBP import kep2car  # Original function
from converted_functions import kep2car_casadi  # CasADi-converted function
from converted_functions import PQW2ECI_casadi  # Ensure PQW2ECI_casadi is imported

def test_kep2car_casadi():
    # Test inputs
    COE = np.array([8000, 0.01, np.deg2rad(30), np.deg2rad(40), np.deg2rad(50), np.deg2rad(60)])
    mu = 398600.433  # Gravitational parameter of Earth (km^3/s^2)

    # Original function result
    RR_original, VV_original = kep2car(COE, mu)
    
    # CasADi function result
    COE_sym = ca.SX.sym('COE', 6)
    mu_sym = ca.SX.sym('mu')

    RR_casadi_sym, VV_casadi_sym = kep2car_casadi(COE_sym, mu_sym)
    kep2car_casadi_func = ca.Function('kep2car_casadi_func', [COE_sym, mu_sym], [RR_casadi_sym, VV_casadi_sym])
    
    RR_casadi_value, VV_casadi_value = kep2car_casadi_func(COE, mu)
    
    RR_casadi_value = RR_casadi_value.full().flatten()
    VV_casadi_value = VV_casadi_value.full().flatten()

    # Print the results
    print(f"Original RR: {RR_original}")
    print(f"CasADi RR: {RR_casadi_value}")
    print(f"Original VV: {VV_original}")
    print(f"CasADi VV: {VV_casadi_value}")
    
    # Assert they are close within tolerance
    tolerance = 1e-6
    assert np.allclose(RR_original, RR_casadi_value, atol=tolerance), \
        f"RR mismatch: {RR_original} vs {RR_casadi_value}"
    assert np.allclose(VV_original, VV_casadi_value, atol=tolerance), \
        f"VV mismatch: {VV_original} vs {VV_casadi_value}"


def test_Param2NROE_casadi():
    # Test data
    NOE_test = np.array([6500, 0.1, np.deg2rad(63.45), 0.5, 0.2, np.deg2rad(270.828)])
    parameters_test = np.array([0, -0.40394247, 0, 0, 0, 0])
    
    data_test = {
        "Primary": [3.98600433e5, 6378.16, 7.2921150e-5]
    }
    
    # Original function result
    original_result = Param2NROE(NOE_test, parameters_test, data_test)

    # Create symbolic inputs
    NOE_sym = ca.SX.sym('NOE', 6)  # 6 elements for NOE
    params_sym = ca.SX.sym('params', 6)  # 6 elements for parameters
    data_sym = ca.SX(data_test["Primary"])  # The data is constant, so it can be treated as a DM

    # Create CasADi function
    param2nroe_casadi_func = ca.Function('param2nroe_casadi_func', 
                                         [NOE_sym, params_sym], 
                                         [Param2NROE_casadi(NOE_sym, params_sym, data_sym)])

    # Call the CasADi function with numeric values
    casadi_result = param2nroe_casadi_func(NOE_test, parameters_test)

    # Convert CasADi result to numpy
    casadi_result_numpy = casadi_result.full().flatten()

    # Check if the CasADi result matches the original result
    tolerance = 1e-6
    assert np.allclose(original_result, casadi_result_numpy, atol=tolerance), \
        f"Original result: {original_result}, CasADi result: {casadi_result_numpy}"

    # Check if the CasADi result matches the original result
    tolerance = 1e-6
    assert np.allclose(original_result, casadi_result_numpy, atol=tolerance), \
        f"Original result: {original_result}, CasADi result: {casadi_result_numpy}"

    print("Test passed!")


def test_lagrange_J2_diff_casadi():
    # Test input values
    yy_test = np.array([6.49992613e+03, 5.59773493e-01, 1.10741825e+00, 5.00055243e-01, 1.99846501e-01, 4.72620248e+00])
    t_test = 381.7395  # Time value
    data_test = {
        "Primary": [3.98600433e5, 6378.16, 7.2921150e-5],  # mu, radius, rotation rate
        "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients
    }
    
    # Original function result (using NumPy for reference)
    f_dot_original = lagrage_J2_diff(t_test, yy_test, data_test)
    
    # CasADi symbolic representation for yy
    yy_sym = ca.SX.sym('yy', 6)
    
    # CasADi function result
    f_dot_casadi_sym = lagrange_J2_diff_casadi(t_test, yy_sym, data_test)
    lagrange_J2_diff_casadi_func = ca.Function('lagrange_J2_diff_casadi_func', [yy_sym], [f_dot_casadi_sym])
    f_dot_casadi = lagrange_J2_diff_casadi_func(yy_test)

    # Flatten and compare results
    f_dot_casadi_value = f_dot_casadi.full().flatten()

    # Check if the results are close within a specified tolerance
    tolerance = 1e-6
    assert np.allclose(f_dot_original, f_dot_casadi_value, atol=tolerance), \
        f"f_dot_original={f_dot_original}, f_dot_casadi={f_dot_casadi_value}"

    print("Test passed!")


def test_rotation_matrices_casadi():
    angle_test = np.deg2rad(30)  # Test angle: 30 degrees

    # C1 matrix test
    C1_result = C1_casadi(angle_test)
    expected_C1 = np.array([[1, 0, 0], [0, np.cos(angle_test), np.sin(angle_test)], [0, -np.sin(angle_test), np.cos(angle_test)]])
    assert np.allclose(np.array(C1_result), expected_C1), f"C1_result: {C1_result}"

    # C2 matrix test (Corrected the sign to match CasADi's output)
    C2_result = C2_casadi(angle_test)
    expected_C2 = np.array([[np.cos(angle_test), 0, -np.sin(angle_test)], [0, 1, 0], [np.sin(angle_test), 0, np.cos(angle_test)]])
    assert np.allclose(np.array(C2_result), expected_C2), f"C2_result: {C2_result}"

    # C3 matrix test
    C3_result = C3_casadi(angle_test)
    expected_C3 = np.array([[np.cos(angle_test), np.sin(angle_test), 0], [-np.sin(angle_test), np.cos(angle_test), 0], [0, 0, 1]])
    assert np.allclose(np.array(C3_result), expected_C3), f"C3_result: {C3_result}"

    
# Test PQW2ECI transformation
def test_PQW2ECI_casadi():
    omega = np.deg2rad(45)
    Omega = np.deg2rad(30)
    i = np.deg2rad(60)

    PQW2ECI_result = PQW2ECI_casadi(Omega, omega, i)
    assert PQW2ECI_result.shape == (3, 3), "PQW2ECI matrix shape is incorrect"

# Test RSW2ECI transformation
def test_RSW2ECI_casadi():
    omega = np.deg2rad(45)
    Omega = np.deg2rad(30)
    i = np.deg2rad(60)
    theta = np.deg2rad(90)

    RSW2ECI_result = RSW2ECI_casadi(Omega, omega, i, theta)
    assert RSW2ECI_result.shape == (3, 3), "RSW2ECI matrix shape is incorrect"

# Test LVLHframe transformation with rr and vv
def test_LVLHframe_casadi():
    rr = np.array([1627.79788358, 5283.50771074, 3396.3559798])  # Position vector
    vv = np.array([-1.38668496, 7.27603177, -2.58379587])        # Velocity vector

    # Convert to CasADi SX symbols
    r_casadi = ca.SX(rr)
    v_casadi = ca.SX(vv)

    LVLHframe_result = LVLHframe_casadi(r_casadi, v_casadi)
    assert LVLHframe_result.shape == (3, 3), "LVLHframe matrix shape is incorrect"
    print("LVLH frame result:", LVLHframe_result)

# Test Frenet2LVLH transformation with rr and vv
def test_Frenet2LVLH_casadi():
    rr = np.array([1627.79788358, 5283.50771074, 3396.3559798])  # Position vector
    vv = np.array([-1.38668496, 7.27603177, -2.58379587])        # Velocity vector

    # Convert to CasADi SX symbols
    r_casadi = ca.SX(rr)
    v_casadi = ca.SX(vv)

    Frenet2LVLH_result = Frenet2LVLH_casadi(r_casadi, v_casadi)
    assert Frenet2LVLH_result.shape == (3, 3), "Frenet2LVLH output shape is incorrect"
    print("Frenet2LVLH result:", Frenet2LVLH_result)




def test_yaw_dynamics():
    # Test parameters
    t = 0  # time
    yy = np.array([0.1, 0.2])  # yaw angles for chief and deputy
    uu = np.array([0.05, 0.03])  # control inputs for chief and deputy

    param = {
        "sat": [1.5, 1.2],  # Inertia for chief and deputy
    }

    # Test the original yaw_dynamics function
    y_dot_original = yaw_dynamics(t, yy, param, uu)

    # Create CasADi variables for the CasADi function
    t_casadi = ca.MX.sym('t')
    yy_casadi = ca.MX.sym('yy', 2)
    uu_casadi = ca.MX.sym('uu', 2)

    # Call the CasADi function
    y_dot_casadi_func = yaw_dynamics_casadi(t_casadi, yy_casadi, param, uu_casadi)
    y_dot_casadi = ca.Function('y_dot', [yy_casadi, uu_casadi], [y_dot_casadi_func])

    # Test the CasADi version
    y_dot_casadi_result = y_dot_casadi(yy, uu).full().flatten()

    # Assert that the results are close enough
    assert np.allclose(y_dot_original, y_dot_casadi_result), \
        f"Mismatch: {y_dot_original} vs {y_dot_casadi_result}"
    


def test_car2NNSOE_density():
    # Example input values
    r = np.array([7000, 0, 0])  # Example position vector in km
    v = np.array([0, 7.5, 0])  # Example velocity vector in km/s
    mu = 3.986004418e5  # Gravitational parameter in km^3/s^2

    # Call the NumPy-based function
    result_numpy = car2NNSOE_density(r, v, mu)

    # Convert to CasADi input symbols
    r_casadi = ca.MX.sym('r', 3)
    v_casadi = ca.MX.sym('v', 3)
    mu_casadi = ca.MX.sym('mu')

    # Create CasADi function for car2NNSOE_density_casadi
    car2NNSOE_density_func = ca.Function('car2NNSOE_density_func', [r_casadi, v_casadi, mu_casadi], [car2NNSOE_density_casadi(r_casadi, v_casadi, mu_casadi)])

    # Evaluate the CasADi function with numerical inputs
    result_casadi = car2NNSOE_density_func(r, v, mu)

    # Convert the CasADi result to NumPy for comparison
    result_casadi_np = np.array(result_casadi).flatten()

    # Assert that the results are close enough
    assert np.allclose(result_numpy, result_casadi_np, rtol=1e-5), \
        f"Results differ! NumPy result: {result_numpy}, CasADi result: {result_casadi_np}"

    print("Test passed: NumPy and CasADi outputs are close.")


# Function to load polynomial coefficients (already saved as a .pkl file)
def load_polynomials(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
# Test function
def test_lookup_surface_properties_casadi():
    # Load the polynomial coefficients
    poly_coeffs = load_polynomials('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl')

    angle = 45  # Example angle in degrees

    # NumPy-based results using your saved function
    numpy_results = lookup_surface_properties(np.deg2rad(angle), poly_coeffs)

    # CasADi-based results using the CasADi version
    angle_casadi = ca.MX.sym('angle_casadi')  # Symbolic angle variable
    angle_num = np.deg2rad(angle)  # Convert the numeric angle to radians
    casadi_results = lookup_surface_properties_casadi(angle_casadi, poly_coeffs, angle_num)

    # Convert the numeric angle to radians and evaluate using the function
    casadi_results_evaluated = casadi_results.full().T

    # Compare the results from both versions
    assert np.allclose(numpy_results, casadi_results_evaluated, atol=1e-6), \
        f"Results differ between Python and CasADi. Python: {numpy_results}, CasADi: {casadi_results_evaluated}"
    
# Helper function to set up test data for both Python and CasADi
def get_test_data():
    # Example test data for spacecraft and atmosphere
    v_rel = np.array([7.5, 0.5, 0.3])  # Relative velocity (km/s)
    rho = 0.0001  # Density (kg/m^3)
    M = 28.9647  # Molar mass (g/mol)
    T = 250  # Temperature (K)
    AOA = np.radians(5)  # Angle of attack in radians
    
    data = {
        "S/C": [300, 2],  # Mass (kg) and cross-sectional area (m^2)
        "Primary": [398600, 6378, 7.2921150e-5]  # Example primary body data (Earth)
    }
    
    # Surface properties example (replace with actual surface properties)
    surface_properties = np.array([
        [1.17939671, -0.32889057, 0.0, 1.17939671],
        [0.2342222, -0.35585337, 0.0, 0.2342222]
    ]).T  

    return v_rel, rho, surface_properties, M, T, data, AOA

@pytest.mark.parametrize("v_rel, rho, surface_properties, M, T, data, AOA", [get_test_data()])
def test_calculate_aerodynamic_forces(v_rel, rho, surface_properties, M, T, data, AOA):
    # Call the Python version of the function
    python_drag, python_lift = calculate_aerodynamic_forces(v_rel, rho, surface_properties.T, M, T, data, AOA)

    # Convert input arrays to CasADi DM objects
    v_rel_casadi = ca.DM(v_rel)
    surface_properties_casadi = ca.DM(surface_properties)
    AOA_casadi = ca.DM(AOA)

    # Call the CasADi version of the function
    casadi_drag, casadi_lift = calculate_aerodynamic_forces_casadi(v_rel_casadi, rho, surface_properties_casadi, M, T, data, AOA_casadi)

    # Convert CasADi outputs to NumPy arrays for comparison
    casadi_drag_np = np.array(casadi_drag.full()).flatten()
    casadi_lift_np = np.array(casadi_lift.full()).flatten()

    # Compare Python and CasADi results
    assert np.allclose(python_drag, casadi_drag_np, atol=1e-6), f"Drag results differ! Python: {python_drag}, CasADi: {casadi_drag_np}"
    assert np.allclose(python_lift, casadi_lift_np, atol=1e-6), f"Lift results differ! Python: {python_lift}, CasADi: {casadi_lift_np}"


# Test function for calculating aerodynamic forces using Python and CasADi
@pytest.mark.parametrize("v_rel, rho, surface_properties, M, T, data, AOA", [get_test_data()])
def test_compute_aerodynamic_forces(v_rel, rho, surface_properties, M, T, data, AOA):
    """
    Test function to compare the Python and CasADi versions of aerodynamic force calculations.
    """

    # Call the Python version
    python_drag, python_lift = calculate_aerodynamic_forces(v_rel, rho, surface_properties.T, M, T, data, AOA)

    # Convert input arrays to CasADi DM objects
    v_rel_casadi = ca.DM(v_rel)
    surface_properties_casadi = ca.DM(surface_properties)
    AOA_casadi = ca.DM(AOA)

    # Call the CasADi version of the function
    casadi_drag, casadi_lift = calculate_aerodynamic_forces_casadi(v_rel_casadi, rho, surface_properties_casadi, M, T, data, AOA_casadi)

    # Convert CasADi outputs to NumPy arrays for comparison
    casadi_drag_np = np.array(casadi_drag.full()).flatten()
    casadi_lift_np = np.array(casadi_lift.full()).flatten()

    # Compare Python and CasADi results
    assert np.allclose(python_drag, casadi_drag_np, atol=1e-6), f"Drag results differ! Python: {python_drag}, CasADi: {casadi_drag_np}"
    assert np.allclose(python_lift, casadi_lift_np, atol=1e-6), f"Lift results differ! Python: {python_lift}, CasADi: {casadi_lift_np}"

    print("Test passed: Python and CasADi outputs match for aerodynamic forces.")


def get_test_data_multiple_entities():
    # Load the polynomial coefficients from a saved file
    loaded_polynomials = load_polynomials("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl")

    # Define data for the primary body (Earth) and spacecraft (S/C)
    data = {
        "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients
        "S/C": [300, 2, 0.9, 300],           # Mass (kg), cross-sectional area (m^2), Cd, ballistic coefficient
        "Primary": [3.98600433e5, 6378.16, 7.2921150e-5]  # Gravitational parameter (mu), Earth's radius (RE), Earth's rotation rate (w)
    }

    # Example positions (r) and velocities (v) for two spacecraft (chief and deputy)
    rr = np.array([
        [100.7097218, -6000.5465031, -3291.97461733],  # Chief position vector (km)
        [82.15330852, -5684.43257548, -3114.50144513]  # Deputy position vector (km)
    ])

    vv = np.array([
        [0.05719481, -3.95747941, 6.78077862],  # Chief velocity vector (km/s)
        [0.05410418, -3.74362997, 6.90235197]   # Deputy velocity vector (km/s)
    ])

    # Example angles of attack (AOA) for each spacecraft in radians
    alpha_list = [np.radians(5), np.radians(3)]  # Chief and deputy angles of attack in radians

    # Return the test data
    return data, loaded_polynomials, alpha_list, vv, rr


@pytest.mark.parametrize("data, loaded_polynomials, alpha_list, vv, rr", [get_test_data_multiple_entities()])
def test_compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr):
    # Python version
    forces_python = compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr)

    # CasADi version: convert input arrays to CasADi DM objects
    vv_casadi = ca.DM(vv)
    rr_casadi = ca.DM(rr)
    alpha_list_casadi = ca.DM(alpha_list)

    # Call CasADi function
    forces_casadi = compute_forces_for_entities_casadi(data, loaded_polynomials, alpha_list_casadi, vv_casadi, rr_casadi)

    # Convert CasADi outputs to NumPy arrays for comparison
    forces_casadi_np = np.array(forces_casadi.full()).reshape(3, 1)  # Convert and reshape to 3x1

    # Compare Python and CasADi results
    for force_python, force_casadi in zip(forces_python, forces_casadi_np):
        assert np.allclose(force_python, force_casadi, atol=1e-6), \
            f"Force mismatch! Python: {force_python}, CasADi: {force_casadi}"

    print("Test passed: Python and CasADi outputs are close.")


from dynamics import Lagrange_deri, guess_nonsingular_Bmat  # Make sure the import path is correct
from converted_functions import Lagrange_deri_casadi, guess_nonsingular_Bmat_casadi  # Make sure the import path is correct


def create_sample_param():
    """
    Function to provide sample parameter data used in the tests.
    """
    return {
        "Primary": [3.98600433e5, 6378.16, 7.2921150e-5],  # mu, Re, w (rotation rate)
        "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients
        "satellites": {
            "chief": {"mass": 300, "area": 2, "C_D": 0.9},
            "deputy_1": {"mass": 250, "area": 1.8, "C_D": 0.85},
        },
        "N_deputies": 2,
        "Init": [0.5, 0.3, 0],  # Initial parameters for q1, q2, and t0
        "sat": [1.2, 1.2],  # Moment of inertia for the satellites
    }


def create_sample_yy():
    """
    Function to provide a sample state vector (non-singular orbital elements).
    """
    return np.array([7000, 0.1, np.pi / 4, 0.1, 0.05, 0.2])


def create_sample_yaw():
    """
    Function to provide sample yaw angle values.
    """
    return np.array([0.1, 0.05])



# Sample data creation function for tests
def create_sample_param():
    return {
        "Primary": [3.98600433e5, 6378.16, 7.2921150e-5],
        "J": [0.1082626925638815e-2, 0, 0],
        "Init": [0.5, 0.3, 0],
    }


def create_sample_yy():
    return np.array([7000, 0.1, np.pi / 4, 0.1, 0.05, 0.2])

def test_lagrange_deri_casadi():
    # Define test parameters
    t = 0.0  # Example test time value
    yy = [7000, 0.1, 0.785398, 0.1, 0.05, 0.2]  # Example test state vector
    param = {
        "Primary": [398600.4418, 6378.137, 7.2921150e-5],
        "J": [1.08262668e-3, 0, 0],
        "Init": [0.1, 0.05, 0]
    }

    # Call the original Python function
    A_mat_original = Lagrange_deri(t, yy, param)

    # Define CasADi symbolic variables
    t_sym = ca.MX.sym('t')  # Symbolic time variable
    yy_sym = ca.MX.sym('yy', 6)  # A 6-element symbolic vector for yy

    # Call the CasADi function with symbolic input
    A_mat_casadi = Lagrange_deri_casadi(t_sym, yy_sym, param)

    # Create a CasADi function for evaluation with the constant parameters
    # Here, param is passed as constants (non-symbolic) inside the CasADi function
    A_func = ca.Function('A_func', [t_sym, yy_sym], [A_mat_casadi])

    # Evaluate the CasADi function at the test values of t and yy
    A_mat_casadi_eval = np.array(A_func(t, yy))

    # Shape test
    assert A_mat_casadi_eval.shape == A_mat_original.shape, (
        f"Expected shape {A_mat_original.shape}, "
        f"but got {A_mat_casadi_eval.shape}"
    )

    # Value test: Check that both functions return similar results
    np.testing.assert_allclose(A_mat_casadi_eval, A_mat_original, rtol=1e-5, atol=1e-8)


def test_guess_nonsingular_Bmat_casadi():
    # Define test parameters
    t = 0.0  # Example test time value
    yy = [7000, 0.1, 0.785398, 0.1, 0.05, 0.2]  # Example test state vector
    param = {
        "Primary": [398600.4418, 6378.137, 7.2921150e-5],
        "J": [1.08262668e-3, 0, 0]
    }
    
    # Call the original Python function
    B_mat_original = guess_nonsingular_Bmat(t, yy, param)

    # Define CasADi symbolic variables
    t_sym = ca.MX.sym('t')  # Symbolic time variable
    yy_sym = ca.MX.sym('yy', 6)  # A 6-element symbolic vector for yy

    # Call the CasADi function with symbolic input
    B_mat_casadi = guess_nonsingular_Bmat_casadi(t_sym, yy_sym, param)

    # Create a CasADi function for evaluation
    B_func = ca.Function('B_func', [t_sym, yy_sym], [B_mat_casadi])

    # Evaluate the CasADi function at the test values of t and yy
    B_mat_casadi_eval = np.array(B_func(t, yy))

    # Shape test
    assert B_mat_casadi_eval.shape == B_mat_original.shape, (
        f"Expected shape {B_mat_original.shape}, "
        f"but got {B_mat_casadi_eval.shape}"
    )

    # Value test: Check that both functions return similar results
    np.testing.assert_allclose(B_mat_casadi_eval, B_mat_original, rtol=1e-5, atol=1e-8)


############# Absolute NSROE dynamics test #############

# Test parameters from the reference script
data = {
    "Primary": [3.98600433e5, 6378.16, 7.2921150e-5],
    "J": [0.1082626925638815e-2, 0, 0],
    "satellites": {
        "chief": {"mass": 300, "area": 2, "C_D": 0.9},
        "deputy_1": {"mass": 250, "area": 1.8, "C_D": 0.85}
    },
    "N_deputies": 2,
    "sat": [1.2, 1.2, 1.2],
    "Init": [0.05, 0.2, 0]
}

# Initial conditions for NOE_chief and yy_o from the reference script
deg2rad = np.pi / 180
NOE_chief = np.array([6500, 0.1, 63.45 * deg2rad, 0.5, 0.2, 270.828 * deg2rad])
yaw_c_d = np.array([0.12, 0.08])
RNOE_0 = Param2NROE(NOE_chief, np.array([0, 0, 0, 0, 0, 0]), data)
yy_o = np.concatenate((RNOE_0, NOE_chief, yaw_c_d))

# Load the polynomial data for aerodynamic calculations
loaded_polynomials = load_polynomials('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl')

# Placeholder for control inputs
uu = np.zeros((2, 1))

# Test time
t = 0.0

# # Test function for absolute_NSROE_dynamics
def test_absolute_NSROE_dynamics():
    # Evaluate the original function
    y_dot_original, u_chief_original = absolute_NSROE_dynamics(t, NOE_chief, data, yy_o)

    # Convert the state and parameters into CasADi symbolic variables
    t_sym = ca.MX.sym('t')
    yy_sym = ca.MX.sym('yy', 6)
    yy_o_sym = ca.MX.sym('yy_o', 14)

    # Call the CasADi version of the function
    y_dot_casadi, u_chief_casadi = absolute_NSROE_dynamics_casadi(t_sym, yy_sym, data, yy_o_sym)

    # Create CasADi functions for evaluation
    abs_NSROE_func = ca.Function('abs_NSROE_func', [t_sym, yy_sym, yy_o_sym], [y_dot_casadi, u_chief_casadi])

    # Evaluate CasADi function at the test values
    y_dot_casadi_eval, u_chief_casadi_eval = abs_NSROE_func(t, NOE_chief, yy_o)

    # Shape test
    assert y_dot_casadi_eval.shape == y_dot_original.shape, (
        f"Expected shape {y_dot_original.shape}, but got {y_dot_casadi_eval.shape}"
    )
    assert u_chief_casadi_eval.shape == u_chief_original.shape, (
        f"Expected shape {u_chief_original.shape}, but got {u_chief_casadi_eval.shape}"
    )

    # Value test: Check that both functions return similar results
    np.testing.assert_allclose(y_dot_casadi_eval, y_dot_original, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(u_chief_casadi_eval, u_chief_original, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":

    test_lookup_surface_properties_casadi()

