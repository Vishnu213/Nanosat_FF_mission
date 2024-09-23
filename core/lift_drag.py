import numpy as np
import pickle
from pyatmos import expo
from scipy.special import erf
from CL_CD_SUTTON import  calculate_cd_cl

## calculate lift and drag forces on a spacecraft

def lookup_surface_properties(angle, poly_coeffs):
    surfaces_data = []
    
    for surface, coeffs in poly_coeffs.items():
        normal_x = np.polyval(coeffs['normal_x'], angle)
        normal_y = np.polyval(coeffs['normal_y'], angle)
        normal_z = np.polyval(coeffs['normal_z'], angle)
        projected_area = np.polyval(coeffs['area'], angle)
        
        # If the projected area is positive, include the surface in the results
        if projected_area > 0:
            surfaces_data.append([normal_x, normal_y, normal_z, projected_area])
    
    return np.array(surfaces_data)

# Function to calculate drag and lift for a given spacecraft
def calculate_aerodynamic_forces(v_rel, rho, surface_properties, C_D, C_L, area_reference, spacecraft_mass):
    a_drag_total = np.zeros(3)  # Initialize drag acceleration vector
    a_lift_total = np.zeros(3)  # Initialize lift acceleration vector
    
    # Loop through the surfaces and calculate drag and lift contributions
    for surface in surface_properties:
        normal_vector = np.array(surface[:3])  # Extract normal vector
        projected_area = surface[3]  # Extract projected area
        
        # Calculate drag coefficient for this surface
        B_D = (projected_area * C_D) / spacecraft_mass
        
        # Calculate lift coefficient for this surface
        B_L = (projected_area * C_L) / spacecraft_mass
        
        # Drag acts opposite to velocity
        drag_direction = -v_rel / np.linalg.norm(v_rel)
        
        # Lift acts perpendicular to velocity and normal vector
        lift_direction_temp = np.cross(np.cross(v_rel,normal_vector),v_rel)
        lift_direction = lift_direction_temp / np.linalg.norm(lift_direction_temp)  # Normalize
        
        # Calculate the contribution to drag acceleration from this surface
        a_drag = 0.5 * rho * np.dot(normal_vector, v_rel) * np.linalg.norm(v_rel) * B_D * drag_direction
        a_drag_total += a_drag
        
        # Calculate the contribution to lift acceleration from this surface
        a_lift = 0.5 * rho * np.dot(normal_vector, v_rel) * np.linalg.norm(v_rel) * B_L * lift_direction
        a_lift_total += a_lift
    
    return a_drag_total, a_lift_total

# Generic function to compute aerodynamic forces for a spacecraft entity
def compute_aerodynamic_forces(entity_data, loaded_polynomials, alpha, vv, rr, h):
    # Relative velocity of the spacecraft
    v_rel = vv - np.cross([0, 0, entity_data["Primary"][2]], rr) # absoluate velocity - Earth rotation factor
    
    # Density value at the spacecraft's altitude
    rho_val =expo(h, 'geopotential')
    rho = rho_val.rho[0]
    
    # Lookup surface properties based on the angle of attack
    surface_properties = lookup_surface_properties(alpha, loaded_polynomials)
    
    # Reference area (this can be modified if AoA affects it)
    A_cross = 0.25  # Simplified constant cross-sectional area
    
    # Drag and lift coefficients (simplified models)
    C_D = 0.9
    C_L = 0.1
    
    # Calculate drag and lift for the spacecraft
    a_drag, a_lift = calculate_aerodynamic_forces(v_rel, rho, surface_properties, C_D, C_L, A_cross, entity_data["S/C"][0])
    
    return a_drag, a_lift

# Main function to compute forces for multiple spacecraft
def compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr, h_list):
    # vv and rr are matrix
    forces = []
    for i, alpha in enumerate(alpha_list):
        h = h_list[i]
        entity_data = data # we need to add two spacecraft details
        a_drag, a_lift = compute_aerodynamic_forces(entity_data, loaded_polynomials, alpha, vv[i], rr[i], h)
        forces.append((a_drag, a_lift))
    return forces

# Function to load the precomputed polynomial coefficients from a file
def load_polynomials(filename='polynomials.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)



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


# Example usage:
if __name__ == "__main__":
    # Load the polynomial coefficients from a saved file
    loaded_polynomials = load_polynomials('../helper_files/polynomials.pkl')
    
    # Assume `data`, `vv`, `rr`, `alpha_list`, and `h_list` are defined
    # `alpha_list` contains angles of attack for each entity (chief, deputy, or others)
    # `h_list` contains the altitudes for each entity

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}

    # Example data for one or more entities
    vv= np.array([[1, 2, 3], [4, 5, 6]])  # Relative velocity for each entity
    rr = np.array([[7, 8, 9], [10, 11, 12]])  # Position vector for each entity
    h_list = [100, 200]  # Altitude for each entity
    alpha_list = [10, 20]  # Angle of attack for each entity
    forces = compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr, h_list)
    
    for i, (a_drag, a_lift) in enumerate(forces):
        print(f"Entity {i+1} drag acceleration: {a_drag}")
        print(f"Entity {i+1} lift acceleration: {a_lift}")
