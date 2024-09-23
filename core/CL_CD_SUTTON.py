"""
Nanosat Formation Flying Project

Modified Sentman's equation for CL and CD 

Author:
    Vishnuvardhan Shakthibala
    
"""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
# Constants
R = 8.314462618  # Universal gas constant in J/mol·K
# Constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
N_A = 6.022e23  # Avogadro's number in mol^-1


T_w = 300  # Wall temperature in Kelvin

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

# Function to calculate Pi, Q, G, Zi, v_rem
def compute_coefficients(v_inc, Ma, Ta, gamma_i, l_i, alpha_acc):
    # Molecular speed ratio
    s = calculate_rms_speed(Ta,Ma)*np.sqrt(Ma)/np.sqrt(2*R*Ta)
    # print(s)
    # print(calculate_rms_speed(Ta,Ma))
    # Calculate Pi
    P_i = np.exp(-gamma_i**2 / s**2) * s**2
    
    # Calculate Q and G
    Q = 1 + 1 / (2 * s**2)
    G = 1 / (2 * s**2)
    
    # Calculate Zi
    Z_i = 1 + erf(gamma_i * s)
    
    # Calculate T_inc
    T_inc = 300

    
    # Calculate v_rem
    v_rem = v_inc * np.sqrt(2/3 * (1 + alpha_acc * (T_w / T_inc - 1)))
    #rint(v_rem)
    return P_i, Q, G, Z_i, v_rem

# Function to calculate Drag and Lift Coefficients (CDi and CLi)
def calculate_cd_cl(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc):
    # Get the coefficients Pi, Q, G, Zi, and v_rem
    P_i, Q, G, Z_i, v_rem = compute_coefficients(v_inc, Ma, Ta, gamma_i, l_i, alpha_acc)
    
    # Calculate Drag Coefficient (CDi)
    Cd = (S_i / S_ref_i) * (P_i / np.sqrt(np.pi) + gamma_i * Q * Z_i + 
                            (gamma_i * v_rem) / (2 * v_inc) * (gamma_i * np.sqrt(np.pi) * Z_i + P_i))
    #print((P_i / np.sqrt(np.pi) + gamma_i * Q * Z_i + 
                            #(gamma_i * v_rem) / (2 * v_inc) * (gamma_i * np.sqrt(np.pi) * Z_i + P_i)))
    
    # Calculate Lift Coefficient (CLi)
    Cl = (S_i / S_ref_i) * (l_i * G * Z_i + (l_i * v_rem) / (2 * v_inc) * (gamma_i * np.sqrt(np.pi) * Z_i + P_i))
    #print((l_i * G * Z_i + (l_i * v_rem) / (2 * v_inc) * (gamma_i * np.sqrt(np.pi) * Z_i + P_i)))
    
    return Cd, Cl

# Function to calculate Cd and Cl for different angles of attack (0 to 360 degrees)
def calculate_cd_cl_for_angles(S_i, S_ref_i, l_i, v_inc, Ma, Ta, alpha_acc):
    angles = np.linspace(0, 360, 360)  # Angles from 0 to 360 degrees
    Cd_values = []
    Cl_values = []
    
    for angle in angles:
        # Calculate gamma_i based on angle of attack (using cosine of the angle)
        gamma_i = np.cos(np.radians(angle))
        l_i = np.sin(np.radians(angle))
        
        # Calculate Cd and Cl for this angle
        Cd, Cl = calculate_cd_cl(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc)
        Cd_values.append(Cd)
        Cl_values.append(Cl)
    
    return angles, Cd_values, Cl_values


if __name__ == "__main__":
    # Example inputs (adjust these as needed)
    S_i = 1.0  # Area of the plate (m^2)
    S_ref_i = 1000  # Reference area (m^2)
    l_i = 0.3  # Example l value
    v_inc = 7500  # Incoming velocity (m/s) - Example for low Earth orbit speed
    Ma = 28.9647  # Mean molar mass of air (g/mol)
    Ta = 1000  # Ambient temperature (K)
    alpha_acc = 1.0  # Accommodation coefficient

    # Generate Cd and Cl values for angles of attack from 0 to 360 degrees
    angles, Cd_values, Cl_values = calculate_cd_cl_for_angles(S_i, S_ref_i, l_i, v_inc, Ma, Ta, alpha_acc)

    # Plot the Cd and Cl values with separate scales for drag and lift coefficients
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot lift coefficient (Cl) on the left y-axis
    ax1.set_xlabel('Angle of Attack (degrees)')
    ax1.set_ylabel('Lift Coefficient (Cl)', color='b')
    ax1.plot(angles, Cl_values, label='Lift Coefficient (Cl)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for drag coefficient (Cd)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Drag Coefficient (Cd)', color='r')
    ax2.plot(angles, Cd_values, label='Drag Coefficient (Cd)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add grid and title
    plt.title('Drag and Lift Coefficients vs Angle of Attack (0 to 360 degrees)')
    fig.tight_layout()
    plt.grid(True)

    plt.show()


