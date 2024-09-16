import numpy as np
import pickle
from pyatmos import expo
from scipy.special import erf

# Function for erf as described in Eq (8)
def erf_function(x):
    return erf(x)

# Function to calculate Cp (Pressure Coefficient)
def compute_cp(alpha_E, delta, s, T_w, T_i):
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
# delta = np.radians(30)  # Example delta in radians
# s = 1.5  # Speed ratio 
# s=vv_inc*np.sqrt(M_a)/np.sqrt(2*R*T_a)
# T_w = 300  # Wall temperature in Kelvin
# T_i = 250  # Incident temperature in Kelvin
# alpha_E = 0.9  # Example accommodation coefficient

# Cp = compute_cp(alpha_E, delta, s, T_w, T_i)
# C_tau = compute_ctau(delta, s)

# print(f"Pressure coefficient (Cp): {Cp}")
# print(f"Shear coefficient (C_tau): {C_tau}")
