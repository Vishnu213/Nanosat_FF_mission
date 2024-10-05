import casadi as ca
import numpy as np
import pickle

from density_model_ca import density_get_casadi

def M2theta_casadi(M, e, tol=1e-10, max_iter=20):
    # Ensure e is within valid range
    e = ca.fmax(ca.fmin(e, 0.9999), 0.0)  # Limit eccentricity to less than 1

    # Initial guess for Eccentric Anomaly E
    E = ca.if_else(M < ca.pi, M + (e / 2), M - (e / 2))

    # Newton-Raphson method with fixed iterations
    for _ in range(max_iter):
        f = E - e * ca.sin(E) - M
        f_prime = 1 - e * ca.cos(E)
        delta_E = -f / f_prime
        E = E + delta_E
        # Note: In CasADi, loops are unrolled, so max_iter should be kept small

    # Calculate theta (True Anomaly)
    sqrt_expr = ca.sqrt((1 + e) / (1 - e))
    theta = 2 * ca.atan(sqrt_expr * ca.tan(E / 2))

    return theta  # Return in radians




def NSROE2car_casadi(ROE, param):
    mu = param["Primary"][0]

    # Assigning the state variables
    a = ROE[0]
    l = ROE[1]
    i = ROE[2]
    q1 = ROE[3]
    q2 = ROE[4]
    OM = ROE[5]

    # Calculations
    e = ca.sqrt(q1**2 + q2**2)
    h = ca.sqrt(mu * a * (1 - e**2))
    eta = 1 - q1**2 - q2**2
    n = ca.sqrt(mu / (a**3))

    # # Handle the condition e == 0 using CasADi's conditional functions
    # e_zero = ca.fabs(e) < 1e-8  # Define a small threshold

    # Check if e is approximately zero
    e_zero = ca.fabs(ca.sqrt(q1**2 + q2**2)) < 1e-8

    # Precompute the expressions for both branches
    omega_peri = ca.if_else(e_zero, 0, ca.acos(q1 / e))
    mean_anomaly = l - omega_peri
    theta = M2theta_casadi(mean_anomaly, e)
    u_false = theta + omega_peri  # What to do if e != 0

    # Now pass these precomputed expressions to if_else
    u = ca.if_else(e_zero, l, u_false)



    # Compute r
    r = (a * eta**2) / (1 + q1 * ca.cos(u) + q2 * ca.sin(u))

    # Position and velocity in perifocal frame
    factor = (h**2 / mu) * (1 / (1 + e * ca.cos(u)))
    rp = factor * (ca.cos(u) * ca.vertcat(1, 0, 0) + ca.sin(u) * ca.vertcat(0, 1, 0))
    vp = (mu / h) * (-ca.sin(u) * ca.vertcat(1, 0, 0) + (e + ca.cos(u)) * ca.vertcat(0, 1, 0))



    # Transformation to ECI frame
    PQW_to_ECI = PQW2ECI_casadi(OM, omega_peri, i)
    RR = ca.mtimes(PQW_to_ECI, rp)
    VV = ca.mtimes(PQW_to_ECI, vp)

    return RR, VV

def calculate_rms_speed_casadi(T, M_mean):
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    N_A = 6.022e23      # Avogadro's number (mol^-1)

    # Convert molar mass to kg per molecule
    M_mean_kg_per_mol = M_mean * 1e-3  # Convert g/mol to kg/mol
    m_mean = M_mean_kg_per_mol / N_A   # Calculate mass per molecule (kg)

    # RMS speed (m/s)
    v_rms = ca.sqrt(3 * k_B * T / m_mean)
    return v_rms

# Conversion of compute_coefficients() to CasADi
def compute_coefficients_casadi(v_inc, Ma, Ta, gamma_i, l_i, alpha_acc):
    R = 8.314462618  # Universal gas constant
    T_w = 300  # Wall temperature in Kelvin

    # Molecular speed ratio s
    s = calculate_rms_speed_casadi(Ta, Ma) * ca.sqrt(Ma) / ca.sqrt(2 * R * Ta)

    # Pi
    P_i = ca.exp(-gamma_i**2 / s**2) * s**2

    # Q and G
    Q = 1 + 1 / (2 * s**2)
    G = 1 / (2 * s**2)

    # Zi
    Z_i = 1 + ca.erf(gamma_i * s)

    # T_inc
    T_inc = 300  # Assuming a constant value

    # v_rem
    v_rem = v_inc * ca.sqrt(2/3 * (1 + alpha_acc * (T_w / T_inc - 1)))

    return P_i, Q, G, Z_i, v_rem

# Conversion of calculate_cd_cl() to CasADi
def calculate_cd_cl_casadi(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc):
    # Get the coefficients Pi, Q, G, Zi, and v_rem
    P_i, Q, G, Z_i, v_rem = compute_coefficients_casadi(v_inc, Ma, Ta, gamma_i, l_i, alpha_acc)

    sqrt_pi = ca.sqrt(ca.pi)

    # Calculate Cd
    Cd = (S_i / S_ref_i) * (P_i / sqrt_pi + gamma_i * Q * Z_i + 
         (gamma_i * v_rem) / (2 * v_inc) * (gamma_i * sqrt_pi * Z_i + P_i))

    # Calculate Cl
    Cl = (S_i / S_ref_i) * (l_i * G * Z_i + 
         (l_i * v_rem) / (2 * v_inc) * (gamma_i * sqrt_pi * Z_i + P_i))

    return Cd, Cl

def car2kep_casadi(r, v, mu):
    """
    Convert from Cartesian coordinates (r, v) to classical orbital elements (COEs) in CasADi.
    
    Parameters:
    r (casadi.SX or MX): Position vector [x, y, z] (km)
    v (casadi.SX or MX): Velocity vector [vx, vy, vz] (km/s)
    mu (float or SX/MX): Standard gravitational parameter (km^3/s^2)
    
    Returns:
    COE: Tuple containing two CasADi objects:
         1. COE_vec: [h_x, h_y, h_z, e_x, e_y, e_z, i, RAAN, omega, theta] in radians
         2. COE_mag: [h_m, e_m, i (deg), RAAN (deg), omega (deg), theta (deg)]
    """
    
    # Magnitude of position and velocity
    r_mag = ca.norm_2(r)
    v_mag = ca.norm_2(v)
    
    # Specific angular momentum
    h = ca.cross(r, v)
    h_mag = ca.norm_2(h)
    
    # Inclination
    i = ca.acos(h[2] / h_mag)
    
    # Node vector (used for RAAN)
    K = ca.vertcat(0, 0, 1)
    N = ca.cross(K, h)
    N_mag = ca.norm_2(N)
    
    # Right Ascension of Ascending Node (RAAN)
    Omega = ca.if_else(N_mag != 0, ca.acos(N[0] / N_mag), 0)
    Omega = ca.if_else(N[1] < 0, 2 * ca.pi - Omega, Omega)
    
    # Eccentricity vector
    e_vec = (1 / mu) * (ca.cross(v, h) - mu * (r / r_mag))
    e_mag = ca.norm_2(e_vec)

    # Argument of periapsis (omega)
    omega = ca.if_else(ca.logic_and(N_mag != 0, e_mag != 0),
                       ca.acos(ca.fmax(ca.fmin(ca.dot(N / N_mag, e_vec / e_mag), 1), -1)), 0)
    omega = ca.if_else(e_vec[2] < 0, 2 * ca.pi - omega, omega)

    # True anomaly (theta)
    theta = ca.if_else(e_mag != 0,
                       ca.acos(ca.dot(e_vec / e_mag, r / r_mag)), 0)
    theta = ca.if_else(ca.dot(r, v) < 0, 2 * ca.pi - theta, theta)
    
    # Semi-major axis (a)
    a = 1 / ((2 / r_mag) - (v_mag ** 2 / mu))
    
    # COE_vec contains the components of h, e, i, Omega, omega, and theta (in radians)
    COE_vec = ca.vertcat(h[0], h[1], h[2], e_vec[0], e_vec[1], e_vec[2], i, Omega, omega, theta)
    
    # COE_mag contains the magnitudes of h, e, and the angular elements converted to degrees
    COE_mag = ca.vertcat(h_mag, e_mag, i, Omega, omega, theta)
    
    # Return tuple
    return (COE_vec, COE_mag)


def kep2car_casadi(COE, mu):
    """
    Convert Keplerian elements (COE) to Cartesian coordinates (r, v) in CasADi.
    
    Parameters:
    COE (casadi.SX or MX): Keplerian elements [h, e, i, OM, om, TA]
    mu (float or casadi.SX/MX): Standard gravitational parameter (km^3/s^2)
    
    Returns:
    r (casadi.SX): Position vector [x, y, z] (km)
    v (casadi.SX): Velocity vector [vx, vy, vz] (km/s)
    """
    # Extract Keplerian elements
    h = COE[0]
    e = COE[1]
    i = COE[2]
    OM = COE[3]
    om = COE[4]
    TA = COE[5]

    # Position and velocity in perifocal reference frame
    rp = (h ** 2 / mu) * (1 / (1 + e * ca.cos(TA))) * (ca.cos(TA) * ca.vertcat(1, 0, 0) + ca.sin(TA) * ca.vertcat(0, 1, 0))
    vp = (mu / h) * (-ca.sin(TA) * ca.vertcat(1, 0, 0) + (e + ca.cos(TA)) * ca.vertcat(0, 1, 0))

    # Transformation from perifocal to ECI frame using PQW2ECI
    RR = ca.mtimes(PQW2ECI_casadi(OM, om, i), rp)
    VV = ca.mtimes(PQW2ECI_casadi(OM, om, i), vp)

    return (RR, VV)


def theta2M_casadi(theta, e, tol=1e-8):
    # Eccentric anomaly
    E = 2 * ca.atan(ca.sqrt((1 - e) / (1 + e)) * ca.tan(theta / 2))
    
    # Mean anomaly
    M = E - e * ca.sin(E)
    
    return M


def Param2NROE_casadi(NOE, parameters, data):
    """
    Convert from design parameters to relative orbital elements using CasADi.
    """

    # Unpack NOE using indexing
    a = NOE[0]
    lambda_ = NOE[1]
    i = NOE[2]
    q1 = NOE[3]
    q2 = NOE[4]
    omega = NOE[5]
    
    # Unpack parameters
    rho_1 = parameters[0]
    rho_2 = parameters[1]
    rho_3 = parameters[2]
    alpha_0 = parameters[3]
    beta_0 = parameters[4]
    v_d = parameters[5]
    
    # Primary parameter
    mu = data[0]  # Access the first element of data["Primary"]

    # CasADi symbolic operations
    eta = ca.sqrt(1 - q1**2 - q2**2)
    n = ca.sqrt(mu / a**3)
    e = ca.sqrt(q1**2 + q2**2)
    h = ca.sqrt(mu * a * (1 - e**2))
    p = (h**2) / mu

    # Delta values (equations from the paper)
    delta_a = (-2 * eta * v_d) / (3 * n)
    
    # Equation 16: delta_Omega
    delta_Omega = (-rho_3 * ca.sin(beta_0)) / (p * ca.sin(i))
    
    # Equation 12: delta_lambda
    delta_lambda = (rho_2 / p) - delta_Omega * ca.cos(i) - (((1 + eta + eta**2) / (1 + eta)) * (rho_1 / p)) * (q1 * ca.cos(alpha_0) - q2 * ca.sin(alpha_0))
    
    # Equation 13: delta_i
    delta_i = (rho_3 / p) * ca.cos(beta_0)
    
    # Equation 14: delta_q1
    delta_q1 = -(1 - q1**2) * (rho_1 / p) * ca.sin(alpha_0) + (q1 * q2 * (rho_1 / p) * ca.cos(alpha_0)) - q2 * (rho_2 / p - delta_Omega * ca.cos(i))
    
    # Equation 15: delta_q2
    delta_q2 = -(1 - q2**2) * (rho_1 / p) * ca.cos(alpha_0) + (q1 * q2 * (rho_1 / p) * ca.sin(alpha_0)) + q1 * (rho_2 / p - delta_Omega * ca.cos(i))
    
    # Return as vector
    return ca.vertcat(delta_a, delta_lambda, delta_i, delta_q1, delta_q2, delta_Omega)


import casadi as ca

def lagrange_J2_diff_casadi(t, yy, data):
    """
    Convert the lagrange_J2_diff() to CasADi for symbolic computation.
    
    Inputs:
        t: Time variable (CasADi SX)
        yy: State vector (CasADi SX)
        data: Dictionary containing constants (mu, Re, J2, etc.)
    
    Returns:
        f_dot: CasADi SX vector of computed dynamics
    """
    mu = data["Primary"][0]
    Re = data["Primary"][1]
    J2 = data["J"][0]

    # Assign the state variables
    a = yy[0]
    l = yy[1]
    i = yy[2]
    q1 = yy[3]
    q2 = yy[4]
    OM = yy[5]

    e = ca.sqrt(q1**2 + q2**2)
    h = ca.sqrt(mu * a * (1 - e**2))
    p = (h**2) / mu
    eta = ca.sqrt(1 - q1**2 - q2**2)

    # Components computation
    component_1 = 0
    n = ca.sqrt(mu / a**3)  # Added this line to compute n
    component_2 = n + ((3/4) * J2 * (Re / p)**2 * n) * (eta * (3 * ca.cos(i)**2 - 1) + (5 * ca.cos(i)**2 - 1))
    component_3 = 0
    component_4 = -(3/4) * J2 * (Re / p)**2 * n * (3 * ca.cos(i)**2 - 1) * q2
    component_5 = (3/4) * J2 * (Re / p)**2 * n * (3 * ca.cos(i)**2 - 1) * q1
    component_6 = -(3/2) * J2 * (Re / p)**2 * n * ca.cos(i)

    # Combine into CasADi vector
    f_dot = ca.vertcat(component_1, component_2, component_3, component_4, component_5, component_6)

    return f_dot

# Rotation Matrices
def C1_casadi(theta):
    C = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(theta), ca.sin(theta)),
        ca.horzcat(0, -ca.sin(theta), ca.cos(theta))
    )
    return C

def C2_casadi(theta):
    C = ca.vertcat(
        ca.horzcat(ca.cos(theta), 0, -ca.sin(theta)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(ca.sin(theta), 0, ca.cos(theta))
    )
    return C

def C3_casadi(theta):
    C = ca.vertcat(
        ca.horzcat(ca.cos(theta), ca.sin(theta), 0),
        ca.horzcat(-ca.sin(theta), ca.cos(theta), 0),
        ca.horzcat(0, 0, 1)
    )
    return C

# PQW to ECI Frame
def PQW2ECI_casadi(OM, om, i):
    C = ca.mtimes(C3_casadi(-OM), ca.mtimes(C1_casadi(-i), C3_casadi(-om)))
    return C

# RSW to ECI Frame
def RSW2ECI_casadi(OM, om, i, theta):
    C = ca.mtimes(C3_casadi(-OM), ca.mtimes(C1_casadi(-i), C3_casadi(-(om + theta))))
    return C

# LVLH Frame
def LVLHframe_casadi(rr, vv):
    # rr and vv are in ECI frame
    r_norm = ca.sqrt(rr[0]**2 + rr[1]**2 + rr[2]**2)
    x_unit = rr / r_norm

    z_unit = ca.cross(rr, vv) / ca.norm_2(ca.cross(rr, vv))
    y_unit = ca.cross(z_unit, x_unit)

    # Matrix to convert from ECI to LVLH frame
    Rot_LVLH = ca.horzcat(x_unit, y_unit, z_unit)
    return Rot_LVLH

# Frenet Frame
def Frenetframe_casadi(rr, vv):
    # rr and vv are in ECI frame
    v_norm = ca.sqrt(vv[0]**2 + vv[1]**2 + vv[2]**2)
    T_unit = vv / v_norm

    W_unit = ca.cross(rr, vv) / ca.norm_2(ca.cross(rr, vv))
    N_unit = ca.cross(T_unit, W_unit)

    # Matrix to convert from ECI to Frenet frame
    Rot_FrenetFrame = ca.horzcat(T_unit, N_unit, W_unit)
    return Rot_FrenetFrame

# Frenet to LVLH Frame Transformation
def Frenet2LVLH_casadi(rr, vv):

    # Assert that rr and vv are 3x1 vectors
    assert rr.shape == (3, 1), f"rr shape mismatch: expected (3, 1), got {rr.shape}"
    assert vv.shape == (3, 1), f"vv shape mismatch: expected (3, 1), got {vv.shape}"


    # Compute Frenet frame and LVLH frame
    RR_Frenet = Frenetframe_casadi(rr, vv)
    RR_LVLH = LVLHframe_casadi(rr, vv)
    
    # Assert that both RR_Frenet and RR_LVLH are 3x3 matrices
    assert RR_Frenet.shape == (3, 3), f"RR_Frenet shape mismatch: expected (3, 3), got {RR_Frenet.shape}"
    assert RR_LVLH.shape == (3, 3), f"RR_LVLH shape mismatch: expected (3, 3), got {RR_LVLH.shape}"
    
    # Compute the transformation from Frenet to LVLH frame
    Rot_F2LVLH = ca.mtimes(ca.transpose(RR_LVLH), RR_Frenet)
    
    # Assert that the result is also a 3x3 matrix
    assert Rot_F2LVLH.shape == (3, 3), f"Rot_F2LVLH shape mismatch: expected (3, 3), got {Rot_F2LVLH.shape}"
    
    return Rot_F2LVLH


def yaw_dynamics_casadi(t, yy, param, uu):
    # Extracting parameters for inertia
    Izc = param["sat"][0]  # Chief satellite's moment of inertia
    Izd = param["sat"][1]  # Deputy satellite's moment of inertia

    # Initialize symbolic state derivatives and control input
    y_dot = ca.MX.zeros(2, 1)
    u = ca.MX.zeros(2, 1)

    # Calculate yaw dynamics for both chief and deputy
    y_dot[0] = -Izc * uu[0]  # Chief's yaw dynamics
    y_dot[1] = -Izd * uu[1]  # Deputy's yaw dynamics

    return y_dot

def car2NNSOE_density_casadi(r, v, mu):
    """
    Convert from Cartesian coordinates (r, v) directly to non-singular elements using CasADi.
    
    Parameters:
    r (casadi.MX or numpy.ndarray): Position vector [x, y, z] (km)
    v (casadi.MX or numpy.ndarray): Velocity vector [vx, vy, vz] (km/s)
    mu (casadi.MX or float): Standard gravitational parameter (km^3/s^2)
    
    Returns:
    Non-singular elements [a, l, i, u, q1, q2, Omega]
    """
    
    # Step 1: Get classical orbital elements (COEs) from position and velocity
    _, COE_temp = car2kep_casadi(r, v, mu)

    # Semi-major axis calculation using CasADi expressions
    a = (COE_temp[0] ** 2) / (mu * (1 - COE_temp[1] ** 2))

    COE = ca.vertcat(a, COE_temp[1], COE_temp[2], COE_temp[3], COE_temp[4], COE_temp[5])

    # Calculate non-singular elements
    M = theta2M_casadi(COE[5], COE[1])
    l = M + COE[4]
    u = COE[5] + COE[4]
    i = COE[2]
    q1 = COE[1] * ca.sin(COE[4])
    q2 = COE[1] * ca.cos(COE[4])
    OM = COE[3]
    
    # Return non-singular elements
    return ca.vertcat(a, l, i, u)


def normalize_casadi(v):
    """Normalize a vector using CasADi."""
    return v / ca.norm_2(v)


km2m = 1e3  # Conversion factor from kilometers to meters


def calculate_aerodynamic_forces_casadi(v_rel, rho, surface_properties, M, T, data, AOA):
    a_drag_total = ca.MX.zeros(3)  # Initialize drag acceleration vector
    a_lift_total = ca.MX.zeros(3)  # Initialize lift acceleration vector
    
    spacecraft_mass = data["S/C"][0]  # Spacecraft mass (kg)
    Area = data["S/C"][1]  # Cross-sectional area (m^2)
    
    for surface in surface_properties:
        normal_vector = ca.MX(surface[:3])  # Extract normal vector
        projected_area = surface[3]  # Extract projected area

        S_i = projected_area  # Area of the plate (m^2)
        S_ref_i = 1000  # Reference area for CL and CD

        v_inc_normalized = normalize_casadi(v_rel)
        n_i_normalized = normalize_casadi(normal_vector)
        theta = ca.acos(ca.dot(v_inc_normalized, n_i_normalized))
        gamma_i = ca.cos(theta)

        lift_direction = ca.cross(normal_vector, v_rel)
        lift_direction_normalized = normalize_casadi(lift_direction)
        l_i = ca.sin(theta)

        v_inc = ca.norm_2(v_rel)  # Incoming velocity (m/s)
        alpha_acc = 1.0  # Accommodation coefficient

        # Calculate Cd and Cl using CasADi versions of calculate_cd_cl (you may need to convert it too)
        C_D, C_L = calculate_cd_cl_casadi(S_i, S_ref_i, gamma_i, l_i, v_inc, M, T, alpha_acc)

        B_D = spacecraft_mass / (Area * projected_area * C_D)
        B_L = spacecraft_mass / (Area * projected_area * C_L)

        drag_direction = v_rel / ca.norm_2(v_rel)
        lift_direction = ca.cross(lift_direction_normalized, v_inc_normalized)

        a_drag = 0.5 * rho * (v_inc * km2m) ** 2 * (1 / B_D) * drag_direction
        a_lift = 0.5 * rho * (v_inc * km2m) ** 2 * (1 / B_L) * lift_direction

        a_drag_total += a_drag / spacecraft_mass
        a_lift_total += a_lift / spacecraft_mass

    return a_drag_total, a_lift_total




# Function to load the precomputed polynomial coefficients from a file
def load_polynomials(filename='polynomials.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)


loaded_polynomials = load_polynomials("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl")



# Custom CasADi polynomial evaluation (Horner's method) with index-based access
def casadi_polyval(coeffs, x):
    """
    Evaluate a polynomial at a given value using Horner's method.
    
    coeffs: CasADi MX vector of polynomial coefficients (highest degree first).
    x: CasADi symbolic variable or numeric value at which to evaluate the polynomial.
    """
    p = coeffs[0]
    for i in range(1, coeffs.numel()):
        p = p * x + coeffs[i]
    return p


# def lookup_surface_properties_casadi(angle, poly_coeffs):
#     surfaces_data = []

#     for surface, coeffs in poly_coeffs.items():
#         # Symbolic polynomial evaluation using custom Horner's method
#         normal_x = casadi_polyval(ca.MX(coeffs['normal_x']), angle)
#         normal_y = casadi_polyval(ca.MX(coeffs['normal_y']), angle)
#         normal_z = casadi_polyval(ca.MX(coeffs['normal_z']), angle)
#         projected_area = casadi_polyval(ca.MX(coeffs['area']), angle)

#         # Use CasADi if_else instead of a Python if statement for symbolic comparison
#         condition = projected_area > 0

#         # Create a zero vector with the same shape as the output
#         zero_vector = ca.DM.zeros(4)

#         # Append the result, ensuring both true and false branches have the same dimensions
#         surfaces_data.append(ca.if_else(condition, ca.vertcat(normal_x, normal_y, normal_z, projected_area), zero_vector))

#     # Concatenate all surfaces symbolically
#     surfaces_concat = ca.horzcat(*surfaces_data) if surfaces_data else ca.DM([])

#     # Now evaluate the result numerically using CasADi's function evaluation
#     eval_function = ca.Function('eval_surfaces', [angle], [surfaces_concat])

#     return eval_function  # Return the CasADi function

# def lookup_surface_properties_casadi(angle, poly_coeffs, angle_numeric):
#     surfaces_data = []

#     for surface, coeffs in poly_coeffs.items():
#         # Symbolic polynomial evaluation using custom Horner's method
#         normal_x = casadi_polyval(ca.MX(coeffs['normal_x']), angle)
#         normal_y = casadi_polyval(ca.MX(coeffs['normal_y']), angle)
#         normal_z = casadi_polyval(ca.MX(coeffs['normal_z']), angle)
#         projected_area = casadi_polyval(ca.MX(coeffs['area']), angle)

#         # Use CasADi if_else instead of a Python if statement for symbolic comparison
#         condition = projected_area > 0

#         # Create a zero vector with the same shape as the output (4x1)
#         zero_vector = ca.DM.zeros(4)

#         # Append the result, ensuring both true and false branches have the same dimensions (4x1)
#         surfaces_data.append(ca.if_else(condition, ca.vertcat(normal_x, normal_y, normal_z, projected_area), zero_vector))

#     # Concatenate all surfaces symbolically (4xN)
#     surfaces_concat = ca.horzcat(*surfaces_data) if surfaces_data else ca.DM([])

#     # Now evaluate the result numerically using CasADi's function evaluation
#     eval_function = ca.Function('eval_surfaces', [angle], [surfaces_concat])

#     # Evaluate the result for a specific numeric angle
#     surfaces_concat_evaluated = eval_function(ca.DM(angle_numeric))

#     # Convert to full matrix for comparison
#     surfaces_concat_evaluated = surfaces_concat_evaluated.full()

#     # Remove any zero rows (all-zero vectors) after evaluation (Ensure at least 4xN output)
#     non_zero_surfaces = surfaces_concat_evaluated[~np.all(surfaces_concat_evaluated == 0, axis=1)]
    
#     print(non_zero_surfaces)

#     return non_zero_surfaces



def lookup_surface_properties_casadi(angle, poly_coeffs, angle_numeric):
    surfaces_data = []

    for surface, coeffs in poly_coeffs.items():
        # Symbolic polynomial evaluation using custom Horner's method
        normal_x = casadi_polyval(ca.MX(coeffs['normal_x']), angle)
        normal_y = casadi_polyval(ca.MX(coeffs['normal_y']), angle)
        normal_z = casadi_polyval(ca.MX(coeffs['normal_z']), angle)
        projected_area = casadi_polyval(ca.MX(coeffs['area']), angle)

        # Use CasADi if_else instead of a Python if statement for symbolic comparison
        condition = projected_area > 0

        # Create a zero vector for the case when the projected area is zero or negative
        zero_vector = ca.DM.zeros(4)

        # Use `ca.if_else` to either use the valid surface or a zero vector
        valid_surface = ca.if_else(condition, ca.vertcat(normal_x, normal_y, normal_z, projected_area), zero_vector)
        
        surfaces_data.append(valid_surface)


    # Concatenate all surfaces symbolically (should be 4xN)
    surfaces_concat = ca.horzcat(*surfaces_data) if surfaces_data else ca.DM([])

    # Now evaluate the result numerically using CasADi's function evaluation
    eval_function = ca.Function('eval_surfaces', [angle], [surfaces_concat])

    # Evaluate the result for a specific numeric angle
    surfaces_concat_evaluated = eval_function(ca.DM(angle_numeric)).full()
    print(surfaces_concat_evaluated)
    # Remove columns with all zero values
    non_zero_columns = surfaces_concat_evaluated[:, ~np.all(surfaces_concat_evaluated == 0, axis=0)]
    print(non_zero_columns)
    # Convert back to CasADi DM format
    non_zero_surfaces_dm = ca.DM(non_zero_columns.transpose())
    print(non_zero_surfaces_dm)
    return non_zero_surfaces_dm.T


def calculate_aerodynamic_forces_casadi(v_rel, rho, surface_properties, M, T, data, AOA):
    a_drag_total = ca.DM.zeros(3)  # Initialize drag acceleration vector
    a_lift_total = ca.DM.zeros(3)  # Initialize lift acceleration vector
    
    spacecraft_mass = data["S/C"][0]  # Spacecraft mass (kg)
    Area = data["S/C"][1]  # Cross-sectional area (m^2)  
    print("surface_properties:", surface_properties)
    # Ensure that surface_properties is in the correct shape (4xN)
    assert surface_properties.shape[0] == 4, "Each surface should have 4 components (normal_x, normal_y, normal_z, projected_area)"
    
    # Iterate through surface properties using horzsplit to split into individual surfaces
    surfaces_split = ca.horzsplit(surface_properties, 1)  # Split into individual surfaces

    for surface in surfaces_split:
        normal_vector = surface[:3]  # Extract normal vector
        projected_area = surface[3]  # Extract projected area

        # CL and CD calculation
        S_i = projected_area  # Area of the plate (m^2)
        S_ref_i = 1000  # large common denominator for CL and CD

        # gamma for this surface
        v_inc_normalized = v_rel / ca.norm_2(v_rel)
        n_i_normalized = normal_vector / ca.norm_2(normal_vector)
        theta = ca.acos(ca.dot(v_inc_normalized, n_i_normalized))
        gamma_i = ca.cos(theta)

        # direction cosine for lift direction for this surface
        lift_direction = ca.cross(normal_vector, v_rel)
        lift_direction_normalized = lift_direction / ca.norm_2(lift_direction)
        
        l_i = ca.sin(theta)
        v_inc = ca.norm_2(v_rel)  # Incoming velocity (m/s)

        Ma = M  # Mean molar mass of air (g/mol)
        Ta = T  # Ambient temperature (K)
        alpha_acc = 1.0  # Accommodation coefficient

        # Dummy Cd and Cl calculation functions (replace with actual implementation)
        C_D, C_L = calculate_cd_cl_casadi(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc)

        # Calculate drag coefficient for this surface
        B_D = spacecraft_mass / (Area * projected_area * C_D)
        
        # Calculate lift coefficient for this surface
        B_L = spacecraft_mass / (Area * projected_area * C_L)
        
        # Drag acts opposite to velocity
        drag_direction = v_rel / ca.norm_2(v_rel)
        
        lift_direction = ca.cross(lift_direction_normalized, v_inc_normalized)
        
        # Calculate the contribution to drag acceleration from this surface
        a_drag = 0.5 * rho * (v_inc * 1e3)**2 * (1 / B_D) * drag_direction
        a_drag_total += a_drag / spacecraft_mass
        
        # Calculate the contribution to lift acceleration from this surface
        a_lift = 0.5 * rho * (v_inc * 1e3)**2 * (1 / B_L) * lift_direction
        a_lift_total += a_lift / spacecraft_mass
    
    return a_drag_total, a_lift_total

# # Generic CasADi-based function to compute aerodynamic forces for a spacecraft entity
# def compute_aerodynamic_forces_casadi(entity_data, loaded_polynomials, AOA, vv, rr):
#     # Relative velocity of the spacecraft
#     v_rel = vv - ca.cross(ca.MX([0, 0, entity_data["Primary"][2]]), rr)

#     # Magnitude of position vector
#     rr_mag = ca.norm_2(rr)

#     # Convert the position vector and velocity to NNSOE
#     NNSOE_den = car2NNSOE_density_casadi(rr, vv, entity_data["Primary"][0])
#     i = NNSOE_den[2]
#     u = NNSOE_den[3]

#     # Density model: use `density_get_casadi` function (CasADi-based)
#     h = rr_mag - entity_data["Primary"][1]
#     rho, M, T = density_get_casadi(h, u, i)

#     # Lookup surface properties using the CasADi function
#     surface_properties = lookup_surface_properties_casadi(AOA * 180 / np.pi, loaded_polynomials, AOA)

#     # Calculate drag and lift for the spacecraft
#     a_drag, a_lift = calculate_aerodynamic_forces_casadi(v_rel, rho, surface_properties, M, T, entity_data, AOA)

#     return a_drag, a_lift


# CasADi version of the function to compute aerodynamic forces for a spacecraft entity
def compute_aerodynamic_forces_casadi(entity_data, loaded_polynomials, AOA, vv, rr):
    # Relative velocity of the spacecraft
    vv = ca.reshape(vv, 3, 1)
    rr = ca.reshape(rr, 3, 1)
    
    v_rel = vv - ca.cross(ca.DM([0, 0, entity_data["Primary"][2]]), rr)  # Absolute velocity - Earth rotation factor

    # Magnitude of the position vector
    rr_mag = ca.norm_2(rr)

    # Convert the position vector and velocity to NNSOE using CasADi version
    NNSOE_den = car2NNSOE_density_casadi(rr, vv, entity_data["Primary"][0])
    i = NNSOE_den[2]
    u = NNSOE_den[3]

    # Compute the altitude
    h = rr_mag - entity_data["Primary"][1]

    print("h:", h.shape)
    print("u:", u.shape)
    print("i:", i.shape)
    # Get density, molar mass, and temperature using CasADi version of density_get
    test_input_casadi = ca.vertcat(h, u, i) 
    density_output = density_get_casadi(test_input_casadi)

    # Unpack the output properly
    rho = density_output[0]  # Get the first element
    M = density_output[1]    # Get the second element
    T = density_output[2]    # Get the third element

    # Convert AOA to symbolic before passing it to CasADi functions
    AOA_sym = ca.MX.sym('AOA')

    # Lookup surface properties using the CasADi version (now using symbolic AOA)
    surface_properties = lookup_surface_properties_casadi(AOA_sym, loaded_polynomials, AOA)

    # Calculate drag and lift using the CasADi version of calculate_aerodynamic_forces
    a_drag, a_lift = calculate_aerodynamic_forces_casadi(v_rel, rho, surface_properties, M, T, entity_data, AOA)

    return a_drag, a_lift


def compute_forces_for_entities_casadi(entity_data, loaded_polynomials, alpha_list, vv, rr):
    num_entities = alpha_list.shape[0]
    forces_casadi = []
    
    for i in range(num_entities):
        # Extract the velocity, position, and angle of attack for each entity

        print("vvsdsdsd",vv.shape)
        print(rr.shape)

        # Use CasADi logic to reshape vectors to 3x1 if they are not in that shape
        v_rel = ca.if_else(vv.shape[0] == 3, vv, ca.reshape(vv[:,i], 3, 1))
        r = ca.if_else(rr.shape[0] == 3, rr, ca.reshape(rr[:,i], 3, 1))

        print(v_rel.shape)
        print(r.shape)
        
        #reshape the velocity and position vectors
        v_rel = ca.reshape(v_rel, 3, 1)
        r = ca.reshape(r, 3, 1)

        AOA = alpha_list[i]

        # Compute aerodynamic forces for each spacecraft using the CasADi version
        a_drag, a_lift = compute_aerodynamic_forces_casadi(entity_data, loaded_polynomials, AOA, v_rel, r)

        print(C1_casadi(AOA).shape)
        print(ca.vertcat(a_drag, a_lift).shape)
        # Ensure a_drag and a_lift are column vectors (3x1)
        a_drag = ca.reshape(a_drag, 3, 1)
        a_lift = ca.reshape(a_lift, 3, 1)

                # Check the shapes again after reshaping
        assert a_drag.shape == (3, 1), f"a_drag reshape failed: {a_drag.shape}"
        assert a_lift.shape == (3, 1), f"a_lift reshape failed: {a_lift.shape}"


        force_sum = a_drag + a_lift
        print(C1_casadi(AOA).shape)
        print(force_sum.shape)
        # Transform the forces into LVLH frame
        rel_f = ca.mtimes(C1_casadi(AOA), force_sum)  # Combine and rotate drag and lift


        F_LVLH_l = ca.mtimes(Frenet2LVLH_casadi(r, v_rel), rel_f)  # Rotate to LVLH frame

        forces_casadi=F_LVLH_l
    
    return forces_casadi  # Concatenate forces for all spacecraft entities


########### Dynamics modules ####################

import casadi as ca

def Lagrange_deri_casadi(t, yy, param):
    """
    CasADi version of Lagrange_deri function using ca.if_else for symbolic logic.
    """
    # Extracting necessary parameters
    mu = param["Primary"][0]
    J2 = param["J"][0]
    Re = param["Primary"][1]
    q1_0 = param["Init"][0]
    q2_0 = param["Init"][1]
    t0 = param["Init"][2]

    # Assigning the state variables from yy
    a = yy[0]
    l = yy[1]
    i = yy[2]
    q1 = yy[3]
    q2 = yy[4]
    OM = yy[5]

    e = ca.sqrt(q1**2 + q2**2)
    h = ca.sqrt(mu * a * (1 - e**2))
    p = h**2 / mu
    eta = ca.sqrt(1 - q1**2 - q2**2)
    n = ca.sqrt(mu / (a**3))

    # Using ca.if_else to handle conditional logic
    u = ca.if_else(e == 0, l, 0)  # If e == 0, use l, otherwise use a default value like 0
    r = ca.if_else(e == 0, (a * eta**2) / (1 + (q1 * ca.cos(u)) + (q2 * ca.sin(u))), 0)
    
    omega_peri = ca.arccos(q1 / e)
    mean_anamoly = l - omega_peri
    theta= M2theta_casadi(mean_anamoly, e, 1e-8)
    u_else = theta + omega_peri
    r_else = (a * eta**2) / (1 + (q1 * ca.cos(u_else)) + (q2 * ca.sin(u_else)))
    
    # Use conditional assignment to apply values when e != 0
    u = ca.if_else(e != 0, u_else, u)
    r = ca.if_else(e != 0, r_else, r)

    epsilon = J2 * (Re / p)**2 * n

    w_dot = (3 * epsilon / 4) * (5 * ca.cos(i)**2 - 1)
    q1_dot = q1_0 * ca.cos(w_dot * (t - t0)) - q2_0 * ca.sin(w_dot * (t - t0))
    q2_dot = q1_0 * ca.sin(w_dot * (t - t0)) + q2_0 * ca.cos(w_dot * (t - t0))

    term_l_a_1 = -(3 * n) / (2 * a)
    term_l_a_2 = (21 * epsilon) / (8 * a) * (eta * (3 * ca.cos(i)**2 - 1) + (5 * ca.cos(i)**2 - 1))
    term_l_a = term_l_a_1 - term_l_a_2

    term_l_i = (-3 * epsilon / 4) * (3 * eta + 5) * ca.sin(2 * i)

    term_l_q1 = (3 * epsilon / (4 * eta**2)) * (3 * eta * (3 * ca.cos(i)**2 - 1) + 4 * (5 * ca.cos(i)**2 - 1)) * q1
    term_l_q2 = (3 * epsilon / (4 * eta**2)) * (3 * eta * (3 * ca.cos(i)**2 - 1) + 4 * (5 * ca.cos(i)**2 - 1)) * q2

    term_q1_a = (21 * epsilon / (8 * a)) * (5 * ca.cos(i)**2 - 1) * q2
    term_q1_i = (15 * epsilon / 4) * q2 * ca.sin(2 * i)
    term_q1_q1 = (-3 * epsilon / eta**2) * (5 * ca.cos(i)**2 - 1) * q1 * q2
    term_q1_q2 = (-3 * epsilon / 4) * (1 + (4 * q2**2) / eta**2) * (5 * ca.cos(i)**2 - 1)

    term_q2_a = (-21 * epsilon / (8 * a)) * (5 * ca.cos(i)**2 - 1) * q1
    term_q2_i = (-15 * epsilon / 4) * q1 * ca.sin(2 * i)
    term_q2_q1 = (3 * epsilon / 4) * (1 + (4 * q1**2) / eta**2) * (5 * ca.cos(i)**2 - 1)
    term_q2_q2 = (3 * epsilon / eta**2) * (5 * ca.cos(i)**2 - 1) * q1 * q2

    term_OM_a = (21 * epsilon / (4 * a)) * ca.cos(i)
    term_OM_i = (3 * epsilon / 2) * ca.sin(i)
    term_OM_q1 = (-6 * epsilon / eta**2) * q1 * ca.cos(i)
    term_OM_q2 = (-6 * epsilon / eta**2) * q2 * ca.cos(i)

    A_mat = ca.vertcat(
        ca.horzcat(0, 0, 0, 0, 0, 0),
        ca.horzcat(term_l_a, 0, term_l_i, term_l_q1, term_l_q2, 0),
        ca.horzcat(0, 0, 0, 0, 0, 0),
        ca.horzcat(term_q1_a, 0, term_q1_i, term_q1_q1, term_q1_q2, 0),
        ca.horzcat(term_q2_a, 0, term_q2_i, term_q2_q1, term_q2_q2, 0),
        ca.horzcat(term_OM_a, 0, term_OM_i, term_OM_q1, term_OM_q2, 0)
    )

    return A_mat



def guess_nonsingular_Bmat_casadi(t, yy, param):
    """
    CasADi version of guess_nonsingular_Bmat for B-matrix calculation.

    Inputs:
        t: Time variable (CasADi SX or MX)
        yy: State vector (CasADi SX or MX)
        param: Dictionary containing parameters like 'Primary' and 'satellites'

    Returns:
        B_mat: CasADi SX or MX matrix (6x6)
    """
    
    mu = param["Primary"][0]
    a = yy[0]
    l = yy[1]
    i = yy[2]
    q1 = yy[3]
    q2 = yy[4]
    OM = yy[5]

    e = ca.sqrt(q1**2 + q2**2)
    h = ca.sqrt(mu * a * (1 - e**2))
    p = (h**2) / mu
    eta = ca.sqrt(1 - q1**2 - q2**2)
    rp = a * (1 - e)
    n = ca.sqrt(mu / (a**3))

    # Compute u and r based on the eccentricity value
    u_circular = l  # Circular orbits: u = l
    r_circular = (a * eta**2) / (1 + q1 * ca.cos(u_circular) + q2 * ca.sin(u_circular))  # r for circular orbit

    # For elliptical orbits, compute u and r
    omega_peri_elliptical = ca.acos(q1 / e)
    mean_anamoly_elliptical = l - omega_peri_elliptical
    theta_tuple = M2theta_casadi(mean_anamoly_elliptical, e, 1e-8)
    theta_elliptical = theta_tuple[0]
    u_elliptical = theta_elliptical + omega_peri_elliptical
    r_elliptical = (a * eta**2) / (1 + q1 * ca.cos(u_elliptical) + q2 * ca.sin(u_elliptical))

    # Use if_else to switch between circular and elliptical cases
    u = ca.if_else(e == 0, u_circular, u_elliptical)
    r = ca.if_else(e == 0, r_circular, r_elliptical)

    # Compute B matrix elements using CasADi symbolic expressions
    y_dot_0 = ((2 * a**2) / h) * ca.vertcat((q1 * ca.sin(u)) - q2 * ca.cos(u), (p / r), 0)
    
    t1 = (-p / (h * (1 + eta))) * (q1 * ca.cos(u) + q2 * ca.sin(u)) - ((2 * eta * r) / h)
    t2 = ((p + r) / (h * (1 + eta))) * (q1 * ca.sin(u) - q2 * ca.cos(u))
    t3 = (r * ca.sin(u) * ca.cos(i)) / (h * ca.sin(i))
    y_dot_1 = ca.vertcat(t1, t2, -t3)

    y_dot_2 = ca.vertcat(0, 0, (r * ca.cos(u)) / h)
    
    t1 = (p * ca.sin(u)) / h
    t2 = (1 / h) * ((p + r) * ca.cos(u) + r * q1)
    t3 = (r * q2 * ca.sin(u) * ca.cos(i)) / (h * ca.sin(i))
    y_dot_3 = ca.vertcat(t1, t2, t3)
    
    t1 = (p * ca.cos(u)) / h
    t2 = (1 / h) * ((p + r) * ca.cos(u) + r * q1)
    t3 = (r * q1 * ca.sin(u) * ca.cos(i)) / (h * ca.sin(i))
    y_dot_4 = ca.vertcat(-t1, t2, -t3)

    y_dot_5 = ca.vertcat(0, 0, (r * ca.sin(u)) / (h * ca.sin(i)))

    # Combine all the y_dot components to form the B matrix
    B_mat = ca.vertcat(y_dot_0.T, y_dot_1.T, y_dot_2.T, y_dot_3.T, y_dot_4.T, y_dot_5.T)
    
    return B_mat


import casadi as ca

def absolute_NSROE_dynamics_casadi(t, yy, param, yy_o):
    # CasADi version of lagrange_J2_diff and guess_nonsingular_Bmat
    A = lagrange_J2_diff_casadi(t, yy, param)
    B = guess_nonsingular_Bmat_casadi(t, yy, param)

    # Convert NSROE to ECI frame using NSROE2car CasADi version
    rr, vv = NSROE2car_casadi(yy, param)

    # Prepare the data for the chief satellite
    data = {}
    data['Primary'] = param['Primary']
    data['S/C'] = [param["satellites"]["chief"]["mass"], param["satellites"]["chief"]["area"]]

    # Use CasADi vectors for rr and vv
    rr_1 = ca.vertcat(rr)
    vv_1 = ca.vertcat(vv)

    print("rr_1.shape:", rr_1.shape)
    print("vv_1.shape:", vv_1.shape)
    # Compute forces for the chief using CasADi version
    u_chief = compute_forces_for_entities_casadi(data, loaded_polynomials, yy_o[12:13], vv_1, rr_1)

    # Scale the chief force output
    u_chief_scale = u_chief * 1e12

    # Compute the derivative y_dot
    y_dot = A + ca.mtimes(B, u_chief)

    return y_dot, u_chief

