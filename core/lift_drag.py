import numpy as np
import pickle
# from pyatmos import expo
from scipy.special import erf
from CL_CD_modified_sentman import  calculate_cd_cl
from TwoBP import car2NNSOE, car2NNSOE_density
from Density_model import model_density, scaler, target_scaler, density_get
from Transformations import C1, Frenet2LVLH


km2m = 1e3  # Conversion factor from kilometers to meters

## calculate lift and drag forces on a spacecraft

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def lookup_surface_properties(angle, poly_coeffs):
    surfaces_data = []
    for surface, coeffs in poly_coeffs.items():
        normal_x = np.polyval(coeffs['normal_x'], angle)
        normal_y = np.polyval(coeffs['normal_y'], angle)
        normal_z = np.polyval(coeffs['normal_z'], angle)
        projected_area = np.polyval(coeffs['area'], angle)

        # print("##############")
        # print("normal_x",normal_x)
        # print("normal_y",normal_y)
        # print("normal_z",normal_z)
        # print("projecteted area",projected_area)
        # If the projected area is positive, include the surface in the results
        # print("##############")
        # print("projecteted area_python",projected_area)
        if projected_area > 0:
            surfaces_data.append([normal_x, normal_y, normal_z, projected_area])
    # print("##############")
    # print("surfaces_data",surfaces_data)
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
        a_drag = 0.5 * rho * np.dot(normal_vector, v_rel) * np.linalg.norm(v_rel) * B_D * drag_direction * 0
        a_drag_total += a_drag
        
        # Calculate the contribution to lift acceleration from this surface
        a_lift = 0.5 * rho * np.dot(normal_vector, v_rel) * np.linalg.norm(v_rel) * B_L * lift_direction * 1
        a_lift_total += a_lift
    
    return a_drag_total, a_lift_total

# Function to calculate drag and lift for a given spacecraft
def calculate_aerodynamic_forces(v_rel, rho, surface_properties, M, T, data, AOA):
    a_drag_total = np.zeros(3)  # Initialize drag acceleration vector
    a_lift_total = np.zeros(3)  # Initialize lift acceleration vector
    
    spacecraft_mass = data["S/C"][0]  # Spacecraft mass (kg)
    Area = data["S/C"][1]  # Cross-sectional area (m^2)  
    # Loop through the surfaces and calculate drag and lift contributions
    for surface in surface_properties:
        normal_vector = np.array(surface[:3])  # Extract normal vector
        projected_area = surface[3]  # Extract projected area

        # CL and CD calculation
        S_i = projected_area  # Area of the plate (m^2)
        S_ref_i = 1000  # large common demoninator for CL and CD


        # gamma for this surface
        v_inc_normalized = normalize(v_rel)
        n_i_normalized = normalize(normal_vector)
        theta = np.arccos(np.dot(v_inc_normalized, n_i_normalized))
        gamma_i = np.cos(theta)

        # direction cosine for llift direction for this surface
        lift_direction = np.cross( normal_vector,v_rel)
        lift_direction_normalized = normalize(lift_direction)
        

        l_i = np.sin(theta)

        v_inc = np.linalg.norm(v_rel)  # Incoming velocity (m/s) - Example for low Earth orbit speed
        Ma = M  # Mean molar mass of air (g/mol)
        Ta = T  # Ambient temperature (K)
        alpha_acc = 1.0  # Accommodation coefficient

        # Calculate Cd and Cl for this surface
        C_D, C_L = calculate_cd_cl(S_i, S_ref_i, gamma_i, l_i, v_inc, Ma, Ta, alpha_acc)

        # Calculate drag coefficient for this surface
        B_D =  spacecraft_mass / (Area*projected_area * C_D) 
        
        # Calculate lift coefficient for this surface
        B_L = spacecraft_mass / (Area*projected_area * C_L) 
        
        # Drag acts opposite to velocity
        drag_direction = v_rel / np.linalg.norm(v_rel)
        
        lift_direction = np.cross(lift_direction_normalized, v_inc_normalized)
        
        # Calculate the contribution to drag acceleration from this surface
        a_drag = 0.5 * rho * (v_inc*km2m)**2  * (1/B_D) * drag_direction * 0
        a_drag_total += a_drag/spacecraft_mass
        
        # Calculate the contribution to lift acceleration from this surface
        a_lift = 0.5 * rho * (v_inc*km2m)**2 * (1/B_L) * lift_direction * 0
        a_lift_total += a_lift/spacecraft_mass
    
    return a_drag_total, a_lift_total

# Generic function to compute aerodynamic forces for a spacecraft entity
def compute_aerodynamic_forces(entity_data, loaded_polynomials, AOA, vv, rr):
    # Relative velocity of the spacecraft
    v_rel = vv - np.cross([0, 0, entity_data["Primary"][2]], rr) # absoluate velocity - Earth rotation factor
    

    # Density value at the spacecraft's altitude
    # h = np.linalg.norm(rr) - entity_data["Primary"][1]
    # rho_val =expo(h, 'geopotential')
    # rho = rho_val.rho[0]

    rr_mag = np.linalg.norm(rr)

    # Convert the position vector and velocity to NNSOE
    NNSOE_den = car2NNSOE_density(rr, vv, entity_data["Primary"][0])
    i = NNSOE_den[2]
    u = NNSOE_den[3]
    

    # Query the KNN model to get density, molar mass, and temperature
    #rho, M, T = query_knn(rr_mag, u, i, kdtree, density_flat, M_flat, T_flat)
    # print("rr_mag---------------",rr_mag)
    h = rr_mag - entity_data["Primary"][1]
    rho, M , T = density_get(h,u,i,model_density, scaler, target_scaler)
    # print("altitude----------------",h)
    # print("rho----------------",rho)
    # Lookup surface properties based on the angle of attack
    surface_properties = lookup_surface_properties(AOA*180/np.pi, loaded_polynomials)
    
    
    # Calculate drag and lift for the spacecraft
    a_drag, a_lift = calculate_aerodynamic_forces(v_rel, rho, surface_properties, M, T , entity_data,AOA)
    # print("a_drag",a_drag)
    # print("a_lift",a_lift)
    return a_drag, a_lift

# Main function to compute forces for multiple spacecraft
def compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr):
    # vv and rr are matrix
    forces = []
    for i, alpha in enumerate(alpha_list):
        entity_data = data # we need to add two spacecraft details
        a_drag, a_lift = compute_aerodynamic_forces(entity_data, loaded_polynomials, alpha, vv[i], rr[i])
        # print(a_drag)
        # print(a_lift)
        rel_f = np.matmul(C1(alpha),np.array(a_drag + a_lift))
        # F_frenet_l = np.matmul(C1(alpha),np.array(a_lift))
        # F_LVLH_l = np.matmul(Frenet2LVLH(rr[i],vv[i]), F_frenet_l)
        # F_frenet_D = np.matmul(C1(alpha),np.array(a_drag))
        # F_LVLH_D = np.matmul(Frenet2LVLH(rr[i],vv[i]),F_frenet_D)
        F_LVLH_l = np.matmul(Frenet2LVLH(rr[i],vv[i]), np.array(rel_f))



    return F_LVLH_l

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



## Loaded polynomial coefficients
loaded_polynomials = load_polynomials("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl")






# # Example usage:
# if __name__ == "__main__":
#     # Load the polynomial coefficients from a saved file
#     loaded_polynomials = load_polynomials("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl")
    
#     # Assume `data`, `vv`, `rr`, `alpha_list`, and `h_list` are defined
#     # `alpha_list` contains angles of attack for each entity (chief, deputy, or others)
#     # `h_list` contains the altitudes for each entity

#     # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
#     data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}

#     # Example data for one or more entities
#     r1 = np.array([100.7097218, -6000.5465031, -3291.97461733])
#     print(np.linalg.norm(r1))
#     v1 = np.array([0.05719481, -3.95747941, 6.78077862])

#     r2 = np.array([82.15330852, -5684.43257548, -3114.50144513])
#     v2 = np.array([0.05410418, -3.74362997, 6.90235197])

#     print("Loading density model...")


#     print("Computing forces for entities...")
#     print(np.linalg.norm(r1))

#     # vv = np.vstack([v1, v2])
#     # rr = np.vstack([r1, r2])

#     vv = np.vstack([v1])
#     rr = np.vstack([r1])
#     h_list = [200]  # Altitude for each entity
#     alpha_list = [80*np.pi/180]  # Angle of attack for each entity
#     forces = compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr)
    
# # Assuming forces is a 2xN array where the first row is drag and the second row is lift
# for i in range(forces.shape[1]):
#     a_drag = forces[0, i]  # First row corresponds to drag
#     a_lift = forces[1, i]  # Second row corresponds to lift
#     print(f"Entity {i+1} drag acceleration: {a_drag}")
#     print(f"Entity {i+1} lift acceleration: {a_lift}")

