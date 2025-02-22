# # test_converted_functions.py

# import numpy as np
# import casadi as ca

# # Import the original function
# from TwoBP import M2theta as M2theta_original

# # Import the CasADi-converted function
# # Assuming it's saved in a file named 'converted_functions_original.py'
# from converted_functions_original import M2theta_casadi

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
#     #  tolerance = 1e-6
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
from pytest import approx



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

from converted_functions_original import M2theta_casadi, NSROE2car_casadi, PQW2ECI_casadi

from CL_CD_modified_sentman import calculate_cd_cl 
from converted_functions_original import calculate_cd_cl_casadi, calculate_rms_speed_casadi


# Import the original and CasADi-converted functions from the relevant files
from TwoBP import car2kep  # Original function from TwoBP.py
from converted_functions_original import car2kep_casadi  # CasADi version from converted functions

from TwoBP import Param2NROE
from converted_functions_original import Param2NROE_casadi

from TwoBP import lagrage_J2_diff
from converted_functions_original import lagrange_J2_diff_casadi


from Transformations import (C1, C2, C3,
    PQW2ECI, RSW2ECI,
    LVLHframe, Frenet2LVLH)

from converted_functions_original import (
    C1_casadi, C2_casadi, C3_casadi,
    PQW2ECI_casadi, RSW2ECI_casadi,
    LVLHframe_casadi, Frenet2LVLH_casadi
)

from dynamics import yaw_dynamics
from converted_functions_original import yaw_dynamics_casadi


from TwoBP import car2NNSOE_density  # Assuming this is the original NumPy version
from converted_functions_original import car2NNSOE_density_casadi  # CasADi version


from lift_drag import lookup_surface_properties  # Original function from TwoBP.py
from converted_functions_original import lookup_surface_properties_casadi


from converted_functions_original import calculate_aerodynamic_forces_casadi  # CasADi version
from lift_drag import calculate_aerodynamic_forces  # Python version

from converted_functions_original import compute_aerodynamic_forces_casadi  # CasADi-converted function
from lift_drag import compute_aerodynamic_forces  # Python version


from converted_functions_original import compute_forces_for_entities_casadi  # CasADi-converted function
from lift_drag import compute_forces_for_entities  # Python version


from dynamics import absolute_NSROE_dynamics
from converted_functions_original import absolute_NSROE_dynamics_casadi

from constrains import con_chief_deputy_angle  
from converted_functions_original import con_chief_deputy_angle_casadi

tolerance = 1e-11

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
    #  tolerance = 1e-6
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
    #  tolerance = 1e-6
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
    #  tolerance = 1e-6
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
    #  tolerance = 1e-6
    assert np.allclose(COE_original_vec, COE_casadi_value_vec, atol=tolerance), \
        f"COE_vec mismatch: {COE_original_vec} vs {COE_casadi_value_vec}"
    assert np.allclose(COE_original_mag, COE_casadi_value_mag, atol=tolerance), \
        f"COE_mag mismatch: {COE_original_mag} vs {COE_casadi_value_mag}"

import numpy as np
import casadi as ca
from TwoBP import kep2car  # Original function
from converted_functions_original import kep2car_casadi  # CasADi-converted function
from converted_functions_original import PQW2ECI_casadi  # Ensure PQW2ECI_casadi is imported

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
    #  tolerance = 1e-6
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
    #  tolerance = 1e-6
    assert np.allclose(original_result, casadi_result_numpy, atol=tolerance), \
        f"Original result: {original_result}, CasADi result: {casadi_result_numpy}"

    # Check if the CasADi result matches the original result
    #  tolerance = 1e-6
    assert np.allclose(original_result, casadi_result_numpy, atol=tolerance), \
        f"Original result: {original_result}, CasADi result: {casadi_result_numpy}"

    print("Test passed!")


# Test function for lagrange_J2_diff_casadi
def test_lagrange_J2_diff_casadi():
    # Define test parameters
    t = 0.0  # Example test time value
    yy = [7000, 0.1, 0.785398, 0.1, 0.05, 0.2]  # Example test state vector
    data = {
        "Primary": [398600.4418, 6378.137, 7.2921150e-5],
        "J": [1.08262668e-3, 0, 0],
        "S/C": [300, 2, 0.9, 300]  # Additional data (optional, not used)
    }

    # Call the original Python function
    f_dot_original = lagrage_J2_diff(t, yy, data)

    # Define CasADi symbolic variables
    t_sym = ca.MX.sym('t')  # Symbolic time variable
    yy_sym = ca.MX.sym('yy', 6)  # A 6-element symbolic vector for yy

    # Call the CasADi function with symbolic input
    f_dot_casadi = lagrange_J2_diff_casadi(t_sym, yy_sym, data)

    # Create a CasADi function for evaluation
    f_dot_func = ca.Function('f_dot_func', [t_sym, yy_sym], [f_dot_casadi])

    # Evaluate the CasADi function at the test values of t and yy
    f_dot_casadi_eval = f_dot_func(t, yy).full().flatten()
    print(f"Original f_dot: {f_dot_original}")
    print(f"CasADi f_dot: {f_dot_casadi_eval}")
    # Shape test
    assert f_dot_casadi_eval.shape[0] == f_dot_original.shape[0], (
        f"Expected shape {f_dot_original.shape}, "
        f"but got {f_dot_casadi_eval.shape}"
    )

    # Value test: Check that both functions return similar results
    np.testing.assert_allclose(f_dot_casadi_eval, f_dot_original, rtol=1e-5, atol=tolerance)


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

    angle_num = 45  # Example angle in degrees

    # NumPy-based results using your saved function
    numpy_results = lookup_surface_properties(np.deg2rad(angle_num), poly_coeffs)

    # Test input for angle (symbolic)
    angle = ca.MX.sym('angle')

    # Call the CasADi function to evaluate the surfaces symbolically
    surfaces_sym = lookup_surface_properties_casadi(angle, poly_coeffs)

    # Create a CasADi function for the symbolic evaluation
    surface_eval_func = ca.Function('surface_eval', [angle], [surfaces_sym])

    # Evaluate the function numerically at an angle of 1.5
    casadi_results = surface_eval_func(np.deg2rad(angle_num))

    # Now, convert the results to a NumPy array
    # casadi_results_evaluated = ca.Function('f', [], [casadi_results])()  # Evaluate MX
    casadi_results_evaluated = casadi_results.full()  # Convert to NumPy
    print("sdadsa",casadi_results.full().shape)
    non_zero_columns_mask = np.any(casadi_results_evaluated != 0, axis=1)
    # Extract columns that are not all zeros
    filtered_array = casadi_results_evaluated[non_zero_columns_mask,:]
    print("sdadsa",filtered_array)
    print("sdadsa 1111",numpy_results)
    # Check if shapes are the same
    assert numpy_results.shape == filtered_array.shape, \
        f"Shapes differ between Python and CasADi. Python shape: {numpy_results.shape}, CasADi shape: {filtered_array.shape}"

    # Compare the values using np.allclose with a tolerance
    assert np.allclose(numpy_results, filtered_array, atol=tolerance), \
        f"Results differ between Python and CasADi. Python: {numpy_results}, CasADi: {filtered_array}"

    # If the code reaches here, both shape and values are equal
    print("The shapes and values match between Python and CasADi results.")


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
    ])

    return v_rel, rho, surface_properties, M, T, data, AOA

@pytest.mark.parametrize("v_rel, rho, surface_properties, M, T, data, AOA", [get_test_data()])

def test_calculate_aerodynamic_forces(v_rel, rho, surface_properties, M, T, data, AOA):
    
    v_rel, rho, surface_properties, M, T, data, AOA = get_test_data()
    # Call the Python version of the function
    python_drag, python_lift = calculate_aerodynamic_forces(v_rel, rho, surface_properties, M, T, data, AOA)

    # Create symbolic variables for the inputs
    v_rel_sym = ca.MX.sym('v_rel', v_rel.shape[0])  # v_rel is a 1D array
    surface_properties_sym = ca.MX.sym('surface_properties', surface_properties.shape[0], surface_properties.shape[1])  # surface_properties is 2D
    AOA_sym = ca.MX.sym('AOA')  # AOA is a scalar or 1D array

    rho_sym = ca.MX.sym('rho')  # Scalar symbolic for rho
    M_sym = ca.MX.sym('M')  # Scalar symbolic for molar mass
    T_sym = ca.MX.sym('T')  # Scalar symbolic for temperature
    data_sym = ca.MX.sym('data')  # Spacecraft mass (assumed scalar)

    # Create a CasADi function from calculate_aerodynamic_forces_casadi
    casadi_func = ca.Function(
        'calc_aero_forces',
        [v_rel_sym, rho_sym, surface_properties_sym, M_sym, T_sym, data_sym, AOA_sym],
        calculate_aerodynamic_forces_casadi(v_rel_sym, rho_sym, surface_properties_sym, M_sym, T_sym, data, AOA_sym))
    

    # Convert input arrays to CasADi DM objects (numeric)
    v_rel_casadi = ca.DM(v_rel)
    surface_properties_casadi = ca.DM(surface_properties)
    AOA_casadi = ca.DM(AOA)

    # Call the CasADi function and get numeric results
    casadi_drag, casadi_lift = casadi_func(v_rel_casadi, ca.DM(rho), surface_properties_casadi, ca.DM(M), ca.DM(T), ca.DM(data['S/C'][0]), AOA_casadi)

    # Convert CasADi results to NumPy for comparison
    casadi_drag_np = np.array(casadi_drag.full()).flatten()  # Drag output as NumPy array
    casadi_lift_np = np.array(casadi_lift.full()).flatten()  # Lift output as NumPy array


    print("Python drag:", python_drag)
    print("CasADi drag:", casadi_drag_np)
    print("Python lift:", python_lift)
    print("CasADi lift:", casadi_lift_np)
 # Shape checks
    assert python_drag.shape == casadi_drag_np.shape, f"Drag shape mismatch! Python: {python_drag.shape}, CasADi: {casadi_drag_np.shape}"
    assert python_lift.shape == casadi_lift_np.shape, f"Lift shape mismatch! Python: {python_lift.shape}, CasADi: {casadi_lift_np.shape}"

    # Compare Python and CasADi results
    assert np.allclose(python_drag, casadi_drag_np, atol=tolerance), f"Drag results differ! Python: {python_drag}, CasADi: {casadi_drag_np}"
    assert np.allclose(python_lift, casadi_lift_np, atol=tolerance), f"Lift results differ! Python: {python_lift}, CasADi: {casadi_lift_np}"

def get_test_data_compute():
    # Define the input data for the test
    entity_data = {
        "S/C": [300, 2],  # Spacecraft mass (kg) and cross-sectional area (m^2)
        "Primary": [398600, 6378, 7.2921150e-5]  # Primary body (Earth) with gravitational constant, radius, and rotation rate
    }

    # Load the polynomial coefficients from a saved file
    loaded_polynomials = load_polynomials("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl")

    # Angle of attack in radians
    AOA = 0.025  # Equivalent to 5 degrees in radians

    # Relative position and velocity vectors for the spacecraft
    rr = np.array([82.15330852, -5684.43257548, -3114.50144513])  # Example r2 vector
    vv = np.array([0.05410418, -3.74362997, 6.90235197])  # Example v2 vector

    # Atmospheric density (example value), molar mass, and temperature
    rho = 2e-12  # Example density in kg/m^3

    return entity_data, loaded_polynomials, AOA, vv, rr

@pytest.mark.parametrize("entity_data, loaded_polynomials, AOA, vv, rr", [get_test_data_compute()])
def test_compute_aerodynamic_forces(entity_data, loaded_polynomials, AOA, vv, rr):
    # Call the Python version of the function
    python_drag, python_lift = compute_aerodynamic_forces(entity_data, loaded_polynomials, AOA, vv, rr)

    # Create symbolic variables for the inputs
    v_rel_sym = ca.MX.sym('v_rel', 3)
    rr_sym = ca.MX.sym('rr', 3)
    AOA_sym = ca.MX.sym('AOA')


    # Call the CasADi version of the function (symbolic)
    casadi_drag_sym, casadi_lift_sym = compute_aerodynamic_forces_casadi(entity_data, loaded_polynomials, AOA_sym, v_rel_sym, rr_sym)

    # Define the CasADi function for symbolic inputs
    casadi_func = ca.Function('aero_forces', [v_rel_sym, rr_sym, AOA_sym], [casadi_drag_sym, casadi_lift_sym])

    # Convert input arrays to CasADi DM objects (numeric inputs)
    vv_casadi = ca.DM(vv)
    rr_casadi = ca.DM(rr)
    AOA_casadi = ca.DM(AOA)

    # Call the CasADi function with numeric inputs
    casadi_results = casadi_func(vv_casadi, rr_casadi, AOA_casadi)

    # Convert CasADi results to NumPy arrays for comparison
    casadi_drag_np = np.array(casadi_results[0].full()).flatten()
    casadi_lift_np = np.array(casadi_results[1].full()).flatten()

    # Shape checks
    assert python_drag.shape == casadi_drag_np.shape, f"Drag shape mismatch! Python: {python_drag.shape}, CasADi: {casadi_drag_np.shape}"
    assert python_lift.shape == casadi_lift_np.shape, f"Lift shape mismatch! Python: {python_lift.shape}, CasADi: {casadi_lift_np.shape}"

    # Value checks (tolerance of 1e-6 for small differences)
    assert np.allclose(python_drag, casadi_drag_np, atol=tolerance), f"Drag values mismatch! Python: {python_drag}, CasADi: {casadi_drag_np}"
    assert np.allclose(python_lift, casadi_lift_np, atol=tolerance), f"Lift values mismatch! Python: {python_lift}, CasADi: {casadi_lift_np}"

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
    alpha_list = np.array([np.radians(5), np.radians(3)])  # Chief and deputy angles of attack in radians

    # Return the test data
    return data, loaded_polynomials, alpha_list, vv, rr


@pytest.mark.parametrize("data, loaded_polynomials, alpha_list, vv, rr", [get_test_data_multiple_entities()])
def test_compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr):
    # Python version
    forces_python = compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr)

    # Create symbolic variables for the inputs
    alpha_sym = ca.MX.sym('alpha',alpha_list.shape[0])
    vv_sym = ca.MX.sym('vv',vv.shape[0],vv.shape[1])
    rr_sym = ca.MX.sym('rr',rr.shape[0],rr.shape[1])

    # Call CasADi version of the function (symbolic)
    print("data",data)
    forces_ca_sym = compute_forces_for_entities_casadi(data, loaded_polynomials, alpha_sym, vv_sym, rr_sym)

    # Define the CasADi function
    forces_casadi_func = ca.Function('forces_casadi', [alpha_sym, vv_sym, rr_sym], [forces_ca_sym])

    # Convert input arrays to CasADi DM objects (numeric)
    vv_casadi = ca.DM(vv)
    rr_casadi = ca.DM(rr)
    alpha_list_casadi = ca.DM(alpha_list)

    # Evaluate the CasADi function with numeric inputs
    forces_casadi_result = forces_casadi_func(alpha_list_casadi, vv_casadi, rr_casadi)

    # Convert CasADi outputs to NumPy arrays for comparison
    forces_casadi_np = np.array(forces_casadi_result.full()).reshape(3, 1)  # Convert and reshape to 3x1

    print("Python forces:", forces_python)
    print("CasADi forces:", forces_casadi_np)
    print("Python forces shape:", forces_python.shape)
    print("CasADi forces shape:", forces_casadi_np.shape)

# Compare Python and CasADi results

    assert forces_python.shape[0] == forces_casadi_np.shape[0], f"Drag shape mismatch! Python: {forces_python.shape}, CasADi: {forces_casadi_np.shape}"

    # Check the values using np.allclose with a tolerance
    assert np.allclose(forces_python, forces_casadi_np, atol=tolerance), \
        f"Force mismatch! Python: {forces_python}, CasADi: {forces_casadi_np}"

    print("Test passed: Python and CasADi outputs match in both shape and values.")



from dynamics import Lagrange_deri, guess_nonsingular_Bmat  # Make sure the import path is correct
from converted_functions_original import Lagrange_deri_casadi, guess_nonsingular_Bmat_casadi  # Make sure the import path is correct


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
    A_func = ca.Function('A_func', [t_sym, yy_sym], [A_mat_casadi])

    # Evaluate the CasADi function at the test values of t and yy
    A_mat_casadi_eval = np.array(A_func(t, yy))

    # Shape test
    assert A_mat_casadi_eval.shape == A_mat_original.shape, (
        f"Expected shape {A_mat_original.shape}, "
        f"but got {A_mat_casadi_eval.shape}"
    )

    # Value test: Check that both functions return similar results
    np.testing.assert_allclose(A_mat_casadi_eval, A_mat_original, rtol=1e-5, atol=tolerance)

    print(f"Lagrange derivative CasADi and Python output shapes: {A_mat_casadi_eval.shape}")


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
    np.testing.assert_allclose(B_mat_casadi_eval, B_mat_original, rtol=1e-5, atol=tolerance)

    print(f"Nonsingular B-matrix CasADi and Python output shapes: {B_mat_casadi_eval.shape}")

############# Absolute NSROE dynamics test #############



# # Test function for absolute_NSROE_dynamics
def test_absolute_NSROE_dynamics():
    # Evaluate the original function
# Parameters that is of interest to the problem

    data = {
        "Primary": [3.98600433e5,6378.16,7.2921150e-5],
        "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients

        # Satellites data including chief and deputies
        "satellites": {
            "chief": {
                "mass": 300,         # Mass in kg
                "area": 2,           # Cross-sectional area in m^2
                "C_D": 0.9,          # Drag coefficient
            },
            "deputy_1": {
                "mass": 250,
                "area": 1.8,
                "C_D": 0.85,
            }
        },
        "N_deputies": 2,  # Number of deputies
        "sat": [1.2, 1.2,1.2],  # Moment of inertia for each satellite

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


    y_dot_original, u_chief_original = absolute_NSROE_dynamics(t, NOE_chief, data, yy_o)

    # Convert the state and parameters into CasADi symbolic variables
    t_sym = ca.MX.sym('t')
    yy_sym = ca.MX.sym('yy', NOE_chief.shape[0])
    yy_o_sym = ca.MX.sym('yy_o', yy_o.shape[0])

    # Call the CasADi version of the function
    y_dot_casadi, u_chief_casadi = absolute_NSROE_dynamics_casadi(t_sym, yy_sym, data, yy_o_sym)

    # Create CasADi functions for evaluation
    abs_NSROE_func = ca.Function('abs_NSROE_func', [t_sym, yy_sym, yy_o_sym], [y_dot_casadi, u_chief_casadi])

    # Evaluate CasADi function at the test values
    y_dot_casadi_eval, u_chief_casadi_eval = abs_NSROE_func(t, NOE_chief, yy_o)

    # Shape test
    assert y_dot_casadi_eval.shape[0] == y_dot_original.shape[0], (
        f"Expected shape {y_dot_original.shape}, but got {y_dot_casadi_eval.shape}"
    )
    assert u_chief_casadi_eval.shape[0] == u_chief_original.shape[0], (
        f"Expected shape {u_chief_original.shape}, but got {u_chief_casadi_eval.shape}"
    )

    print(f"Absolute NSROE dynamics CasADi and Python output shapes: {y_dot_casadi_eval.shape} and {u_chief_casadi_eval.shape}")
    print("Python y_dot:", y_dot_original)
    print("CasADi y_dot:", y_dot_casadi_eval)
    print("Python u_chief:", u_chief_original)
    print("CasADi u_chief:", u_chief_casadi_eval)
    # Value test: Check that both functions return similar results
    np.testing.assert_allclose(y_dot_casadi_eval, y_dot_original, rtol=1e-5, atol=tolerance)
    np.testing.assert_allclose(u_chief_casadi_eval, u_chief_original, rtol=1e-5, atol=tolerance)


from dynamics import Dynamics  # Make sure the import path is correct
from converted_functions_original import Dynamics_casadi  # Make sure the import path is correct

# Test function for Dynamics_casadi
def test_Dynamics_casadi():

# Parameters that is of interest to the problem

    param = {
        "Primary": [3.98600433e5,6378.16,7.2921150e-5],
        "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients

        # Satellites data including chief and deputies
        "satellites": {
            "chief": {
                "mass": 300,         # Mass in kg
                "area": 2,           # Cross-sectional area in m^2
                "C_D": 0.9,          # Drag coefficient
            },
            "deputy_1": {
                "mass": 250,
                "area": 1.8,
                "C_D": 0.85,
            }
        },
        "N_deputies": 2,  # Number of deputies
        "sat": [1.2, 1.2,1.2],  # Moment of inertia for each satellite
    "T_MAX": 23e-6,  # Maximum torque (Nm)
    "PHI_DOT": [0.1, 0.1],  # Limits for yaw rate (rad/s)
    "PHI": [-ca.pi / 2, ca.pi / 2],  # Limits for yaw angle (rad)
    "T_period": 2000.0, # Period of the sine wave
    }

    # Initial conditions for NOE_chief and yy_o from the reference script
    deg2rad = np.pi / 180
    NOE_chief = np.array([6500, 0.1, 63.45 * deg2rad, 0.5, 0.2, 270.828 * deg2rad])
    yaw_c_d = np.array([0.12, 0.08,0,0])
    RNOE_0 = Param2NROE(NOE_chief, np.array([0, 0, 0, 0, 0, 0]), param)
    yy = np.concatenate((RNOE_0, NOE_chief, yaw_c_d))

    # Load the polynomial data for aerodynamic calculations
    loaded_polynomials = load_polynomials('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl')

    # Placeholder for control inputs
    uu = np.zeros((2, 1))

    # Test time
    t = 0.0

    param["Init"] = [NOE_chief[4],NOE_chief[3], 0]  # Initial parameters for q1, q2, and t0


    # Call the original Python Dynamics function
    y_python = Dynamics(t, yy, param, uu)

    # Convert the inputs to CasADi symbolic variables
    t_sym = ca.MX.sym('t')
    yy_sym = ca.MX.sym('yy', len(yy))
    uu_sym = ca.MX.sym('uu', len(uu))


    # Call the CasADi function for Dynamics
    dynamics_casadi_sym = Dynamics_casadi(t_sym, yy_sym, uu_sym)
    dynamics_func = ca.Function('dynamics_func', [t_sym, yy_sym, uu_sym], [dynamics_casadi_sym])

    # Evaluate the CasADi function with numeric inputs
    dynamics_casadi_result = dynamics_func(t, yy, uu)

    # Convert CasADi result to NumPy array for comparison
    y_casadi = np.array(dynamics_casadi_result.full()).flatten()

    # Now test the result for expected behavior (compare with original Python version or numerical expectations)
    # Since this is a placeholder, you can assert for shape, or specific known expected values.
    
    # Compare the Python and CasADi results
    assert y_python.shape == y_casadi.shape, f"Shape mismatch: Python {y_python.shape}, CasADi {y_casadi.shape}"
    assert np.allclose(y_python, y_casadi, atol=tolerance), f"Values mismatch: Python {y_python}, CasADi {y_casadi}"

from TwoBP import NSROE2LVLH  # Make sure the import path is correct
from converted_functions_original import NSROE2LVLH_casadi  # Make sure the import path is correct

def test_NSROE2LVLH_casadi():
    data = {
        "Primary": [3.98600433e5,6378.16,7.2921150e-5],
        "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients

        # Satellites data including chief and deputies
        "satellites": {
            "chief": {
                "mass": 300,         # Mass in kg
                "area": 2,           # Cross-sectional area in m^2
                "C_D": 0.9,          # Drag coefficient
            },
            "deputy_1": {
                "mass": 250,
                "area": 1.8,
                "C_D": 0.85,
            }
        },
        "N_deputies": 2,  # Number of deputies
        "sat": [1.2, 1.2,1.2],  # Moment of inertia for each satellite
    "T_MAX": 23e-6,  # Maximum torque (Nm)
    "PHI_DOT": [0.1, 0.1],  # Limits for yaw rate (rad/s)
    "PHI": [-ca.pi / 2, ca.pi / 2],  # Limits for yaw angle (rad)
    "T_period": 2000.0  # Period of the sine wave
    }

    # Initial conditions for NOE_chief and yy_o from the reference script
    deg2rad = np.pi / 180
    NSOE0_test = np.array([6500, 0.1, 63.45 * deg2rad, 0.5, 0.2, 270.828 * deg2rad])
    yaw_c_d = np.array([0.12, 0.08])

    NSROE_test = Param2NROE(NSOE0_test, np.array([ 0,-0.40394247,0,0,np.pi/2,0]), data)
    print("RNOE_0",NSROE_test)
    print("NOE_chief",NSOE0_test)
    yy = np.concatenate((NSROE_test, NSOE0_test, yaw_c_d))
    # Original function result
    rr_np = NSROE2LVLH(NSROE_test, NSOE0_test, data)

    # CasADi function result
    NSROE_sym = ca.SX.sym('NSROE', 6)
    NSOE0_sym = ca.SX.sym('NSOE0', 6)
    rr_casadi_sym = NSROE2LVLH_casadi(NSROE_sym, NSOE0_sym, data)

    # Create a CasADi function
    rr_casadi_func = ca.Function('rr_casadi_func', [NSROE_sym, NSOE0_sym], [rr_casadi_sym])

    # Evaluate the CasADi function with numeric values
    rr_casadi = rr_casadi_func(ca.DM(NSROE_test), ca.DM(NSOE0_test)).full().flatten()
    print("Python result:", rr_np)
    print("CasADi result:", rr_casadi)
    # Compare the results
    assert rr_casadi == approx(rr_np, abs=1e-11), f"NumPy: {rr_np}, CasADi: {rr_casadi}"


def test_con_chief_deputy_angle():
# Parameters that is of interest to the problem

    data = {
        "Primary": [3.98600433e5,6378.16,7.2921150e-5],
        "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients

        # Satellites data including chief and deputies
        "satellites": {
            "chief": {
                "mass": 300,         # Mass in kg
                "area": 2,           # Cross-sectional area in m^2
                "C_D": 0.9,          # Drag coefficient
            },
            "deputy_1": {
                "mass": 250,
                "area": 1.8,
                "C_D": 0.85,
            }
        },
        "N_deputies": 2,  # Number of deputies
        "sat": [1.2, 1.2,1.2],  # Moment of inertia for each satellite

    }

    # Initial conditions for NOE_chief and yy_o from the reference script
    deg2rad = np.pi / 180
    NOE_chief = np.array([6500, 0.1, 63.45 * deg2rad, 0.5, 0.2, 270.828 * deg2rad])
    yaw_c_d = np.array([0.12, 0.08])

    RNOE_0 = Param2NROE(NOE_chief, np.array([ 0,-0.40394247,0,0,np.pi/2,0]), data)
    print("RNOE_0",RNOE_0)
    yy = np.concatenate((RNOE_0, NOE_chief, yaw_c_d))


    # Original NumPy function output
    original_angle = con_chief_deputy_angle(yy, data)
    print("yy",yy.shape[0])
    # CasADi function input (convert to MX)
    yy_casadi = ca.MX.sym('yy', yy.shape[0],1)
    
    # CasADi function output
    casadi_ang = con_chief_deputy_angle_casadi(yy_casadi, data)
    
    ca_func = ca.Function('con_chief_deputy_angle', [yy_casadi], [casadi_ang])
    casadi_angle_val = ca_func(ca.DM(yy)).full().flatten()
    print("Python result:", original_angle)
    print("CasADi result:", casadi_angle_val)
    print("Python shape:", original_angle.shape)
    print("CasADi shape:", casadi_angle_val.shape)
    # Test the shape
    assert casadi_angle_val.shape[0] == 1, "Shape mismatch between original and CasADi outputs"

    # Test the value within a tolerance
    original_angle_val = original_angle
    casadi_angle_val = casadi_angle_val  # Convert CasADi output to NumPy array
    np.testing.assert_allclose(original_angle_val, casadi_angle_val, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    # data, loaded_polynomials, alpha_list, vv, rr=get_test_data_multiple_entities()
    # v_rel, rho, surface_properties, M, T, data, AOA = get_test_data()
    # test_compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr)

    test_Dynamics_casadi()
    # # # test_con_chief_deputy_angle()
    #test_NSROE2LVLH_casadi()

    # test_con_chief_deputy_angle()