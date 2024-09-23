import numpy as np
import pickle
from pyatmos import expo
from scipy.special import erf
import matplotlib.pyplot as plt

# Constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
N_A = 6.022e23  # Avogadro's number in mol^-1
R = 8.314462618  # J/mol·K

T_w = 300  # Wall temperature in Kelvin

# Function to calculate RMS speed given temperature and mean molar mass
def calculate_rms_speed(T, M_mean):
    """
    Calculate the root mean square (RMS) speed of gas molecules.
    
    Parameters:
    T (float): Temperature in Kelvin (K).
    M_mean (float): Mean molar mass in grams per mole (g/mol).
    
    Returns:
    float: RMS speed in meters per second (m/s).
    """
    # Convert molar mass to kg per molecule
    M_mean_kg_per_mol = M_mean * 1e-3  # Convert g/mol to kg/mol
    m_mean = M_mean_kg_per_mol / N_A  # Mean molecular mass in kg per molecule
    
    # Calculate RMS speed
    v_rms = np.sqrt(3 * k_B * T / m_mean)
    
    return v_rms
# Function for erf as described in Eq (8)


def erf_function(x):
    return erf(x)

# Function to calculate Cp (Pressure Coefficient)
def compute_cp(alpha_E, delta, s):


    term1 = (np.cos(delta)**2 + 1 / (2 * s**2)) * (1 + erf(s * np.cos(delta)))
    term2 = np.cos(delta) / (s * np.sqrt(np.pi)) * np.exp(-s**2 * np.cos(delta)**2)
    
    term3_part1 = np.sqrt(np.pi) * np.cos(delta) * (1 + erf(s * np.cos(delta)))
    term3_part2 = (1 / s) * (np.exp(-s**2 * np.cos(delta)**2))
    term3 = (1 / 2) * np.sqrt((2 / 3) * (1 + (alpha_E * T_w / (T_i - 1)))) * (term3_part1 + term3_part2)
    
    Cp = term1 + term2 + term3
    return Cp

# Function to calculate Ctau (Shear Coefficient)
def compute_ctau(delta, s):
    term1 = np.sin(delta) * np.cos(delta) * (1 + erf(s * np.cos(delta)))
    term2 = np.sin(delta) / (s * np.sqrt(np.pi)) * np.exp(-s**2 * np.cos(delta)**2)
    
    C_tau = term1 + term2
    return C_tau



# # Example usage
delta = np.radians(30)  # Example delta in radians
alpha_E = 0.9  # Example accommodation coefficient
## use th density model to get the temperature, mean molar mass
T_i = 900 # Incident temperature in Kelvin
M = 15.97 # Mean molar mass of air in g/mol

s=calculate_rms_speed(T_i,M)*np.sqrt(M)/np.sqrt(2*R*T_i)

transformation_matrix = lambda delta: np.array([
    [np.cos(delta), np.sin(delta)],
    [np.sin(delta), np.cos(delta)]
])

# Calculate the speed scaling factor
s = calculate_rms_speed(T_i, M) * np.sqrt(M) / np.sqrt(2 * R * T_i)

# Angles for delta in radians (0 to 360 degrees)
delta_values = np.radians(np.linspace(0, 360, 360))

# Initialize arrays for storing Cd and Cl
Cd_values = []
Cl_values = []

# Loop over delta values and compute Cd and Cl
for delta in delta_values:
    Cp = compute_cp(alpha_E, delta, s)
    C_tau = compute_ctau(delta, s)
    
    # Apply the transformation matrix
    Cd_Cl = np.dot(np.array([Cp, C_tau]), transformation_matrix(delta))
    
    # Store Cd and Cl values
    Cd_values.append(Cd_Cl[0])
    Cl_values.append(Cd_Cl[1])

# Convert lists to arrays
Cd_values = np.array(Cd_values)
Cl_values = np.array(Cl_values)

# Plotting Cd and Cl
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(delta_values), Cd_values, label='C_d (Drag Coefficient)', color='r')
plt.plot(np.degrees(delta_values), Cl_values, label='C_l (Lift Coefficient)', color='b')
plt.xlabel('Delta (Degrees)')
plt.ylabel('Coefficient Value')
plt.title('Drag (Cd) and Lift (Cl) Coefficients vs. Angle of Attack (Delta)')
plt.legend()
plt.grid(True)
plt.show()



# print(f"Pressure coefficient (Cp): {Cp}")
# print(f"Shear coefficient (C_tau): {C_tau}")
