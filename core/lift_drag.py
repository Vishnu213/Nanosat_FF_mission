import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
# from pyatmos import expo
from scipy.special import erf
from CL_CD_modified_sentman import  calculate_cd_cl
from TwoBP import car2NNSOE, car2NNSOE_density
from Density_model import model_density, scaler, target_scaler, density_get
from Transformations import C1, Frenet2LVLH
from projected_area import lookup_table_surface_properties



# Load the JSON lookup table
def load_lookup_table(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Example usage after loading
lookup_table_path = 'C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nano_sat_casadi\\Nanosat_FF_mission\\helper_files\\lookup_table_projected_area.json'
loaded_lookup_table = load_lookup_table(lookup_table_path)


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
def calculate_aerodynamic_forces(v_rel, rho, surface_properties, M, T, data, AOA):
    a_drag_total = np.zeros(3)  # Initialize drag acceleration vector
    a_lift_total = np.zeros(3)  # Initialize lift acceleration vector
    
    spacecraft_mass = data["S/C"][0]  # Spacecraft mass (kg)
    Area = data["S/C"][1]  # Cross-sectional area (m^2)  
    combined_pro_area = 0  # Initialize combined projected area
    CL_sum = 0
    CD_sum = 0
    
    # Loop through the surfaces and calculate drag and lift contributions
    counter = 0
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
        #print("S_i ",S_i, "S_ref_i ",S_ref_i, "gamma_i ",gamma_i, "l_i ",l_i, "v_inc ",v_inc, "Ma ",Ma, "Ta ",Ta, "alpha_acc ",alpha_acc)
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
        a_drag = -0.5 * rho * (v_inc*km2m)**2  * (1/B_D) * drag_direction * 1
        a_drag_total += a_drag/spacecraft_mass
        
        # Calculate the contribution to lift acceleration from this surface
        a_lift = -0.5 * rho * (v_inc*km2m)**2 * (1/B_L) * lift_direction * 1
        a_lift_total += a_lift/spacecraft_mass

        # Add the projected area to the combined projected area
        combined_pro_area += projected_area
        CL_sum += C_L
        CD_sum += C_D
        counter += 1


    if counter == 1 and not (AOA == 0):
        # print("counter",counter)
        # print(AOA)
        dd=0

    return a_drag_total, a_lift_total,CD_sum, CL_sum, combined_pro_area


import numpy as np

def atmosphere(h):
    """
    Computes the atmospheric density using an exponential model.

    Parameters:
    h (float): Altitude from Earth's surface [km]

    Returns:
    float: Atmospheric density [kg/m^3]
    """
    # Reference altitude vector [km]
    h_0 = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                    180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
    
    # Reference density vector [kg/m^3]
    rho_0 = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 
                      8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 
                      8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10, 2.798e-10, 7.248e-11, 
                      2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13,
                      3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15])
    
    # Scale height vector [km]
    H = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 
                  5.877, 7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546,
                  53.628, 53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64,
                  181.05, 268.00])

    # Find the index of the reference parameters
    j = 27  # Default to the highest range if altitude is greater than the last value
    for k in range(27):
        if h_0[k] <= h <= h_0[k + 1]:
            j = k
            break

    # Compute density [kg/m^3] using exponential model
    rho = rho_0[j] * np.exp(-(h - h_0[j]) / H[j])

    return rho


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
    #surface_properties = lookup_surface_properties(AOA*180/np.pi, loaded_polynomials)

    surface_properties  = lookup_table_surface_properties(AOA,  loaded_lookup_table)  
    # print("AOA",AOA)
    # print("surface_properties",surface_properties)
    
    
    rho = atmosphere(h)
    # print( "rho -----", rho)
    # Calculate drag and lift for the spacecraft
    a_drag, a_lift, CD, CL, pro_area = calculate_aerodynamic_forces(v_rel, rho , surface_properties, M, T , entity_data,AOA)
    # print("a_drag",a_drag)
    # print("a_lift",a_lift)
    return a_drag, a_lift, CL, CD, pro_area

# Main function to compute forces for multiple spacecraft
def compute_forces_for_entities(data, loaded_polynomials, alpha_list, vv, rr):
    # print("alpha_list",alpha_list, "vv",vv, "rr",rr)
    # vv and rr are matrix
    forces = []
    for i, alpha in enumerate(alpha_list):
        entity_data = data # we need to add two spacecraft details
        a_drag, a_lift, CL, CD, pro_area = compute_aerodynamic_forces(entity_data, loaded_polynomials, alpha, vv[i], rr[i])
        # print(a_drag)
        # print(a_lift)
        rel_f = np.matmul(C1(alpha),np.array(a_drag + a_lift))
        # F_frenet_l = np.matmul(C1(alpha),np.array(a_lift))
        # F_LVLH_l = np.matmul(Frenet2LVLH(rr[i],vv[i]), F_frenet_l)
        # F_frenet_D = np.matmul(C1(alpha),np.array(a_drag))
        # F_LVLH_D = np.matmul(Frenet2LVLH(rr[i],vv[i]),F_frenet_D)
        F_LVLH_l = np.matmul(Frenet2LVLH(rr[i],vv[i]), np.array(rel_f))



    return F_LVLH_l #, a_lift , a_drag ,CL, CD, pro_area ### ENABLE ME IF YOU WANT TO RUN THIS FILE

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




# Example usage:
if __name__ == "__main__":
    # Load the polynomial coefficients from a saved file
    loaded_polynomials = load_polynomials("C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nanosat_FF_mission\\helper_files\\polynomials.pkl")
    
    # Assume `data`, `vv`, `rr`, `alpha_list`, and `h_list` are defined
    # `alpha_list` contains angles of attack for each entity (chief, deputy, or others)
    # `h_list` contains the altitudes for each entity

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
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
        "sat": [0.0412, 0.0412,1.2],  # Moment of inertia for each satellite
        "T_period": 2000.0,  # Period of the sine wave
    }

    print("Parameters initialized.")

    satellite_key = f"deputy_1"  # Access deputy names dynamically
    satellite_properties = data["satellites"][satellite_key]    
    data_deputy = {}
    data_deputy['Primary'] = data['Primary']
    data_deputy['S/C'] = [satellite_properties["mass"], satellite_properties["area"]]

    # Example data for one or more entities
    r1 = np.array([2347.3093079 , -3943.69432762 , 4716.01627407])
    print(np.linalg.norm(r1))
    v1 = np.array([2.0969191, 6.24384464, 4.16757144])

    r2 = np.array([82.15330852, -5684.43257548, -3114.50144513])
    v2 = np.array([0.05410418, -3.74362997, 6.90235197])

    print("Loading density model...")


    print("Computing forces for entities...")
    print(np.linalg.norm(r1))

    # vv = np.vstack([v1, v2])
    # rr = np.vstack([r1, r2])

    vv = np.vstack([v1])
    rr = np.vstack([r1])
    h_list = [200]  # Altitude for each entity
    # # alpha_list = [5]  # Angle of attack for each entity
    # # combined_force_lvlh, L, D, CL, CD, pro_area = compute_forces_for_entities(data_deputy, loaded_polynomials, alpha_list, vv, rr)
    # # print(f"Combined force in LVLH frame: {combined_force_lvlh}")
    # # print(f"Lift force: {L}")
    # # print(f"Drag force: {D}")
    # # print(f"Lift coefficient: {CL}")
    # # print(f"Drag coefficient: {CD}")
    # # print(f"Projected area: {pro_area}")
    # # exit()


    # # Assuming forces is a 2xN array where the first row is drag and the second row is lift
    # for i in range(forces.shape[1]):
    #     a_drag = forces[0, i]  # First row corresponds to drag
    #     a_lift = forces[1, i]  # Second row corresponds to lift
    #     print(f"Entity {i+1} drag acceleration: {a_drag}")
    #     print(f"Entity {i+1} lift acceleration: {a_lift}")

    # Define alpha range from 0 to 360 degrees (in radians)
    alpha_degrees = np.arange(0, 360 , 1)
    alpha_list = alpha_degrees  # Convert degrees to radians

    # Initialize arrays to store the drag and lift forces and their directions
    drag_forces = []
    lift_forces = []
    area_pro = []
    CD_list = []
    CL_list = []
    drag_directions = []
    lift_directions = []

    # Initialize arrays to store the combined forces in the LVLH frame
    combined_forces_magnitudes = []
    combined_forces_directions = []

    # Loop over each alpha and calculate the forces
    for alpha in alpha_list:

        combined_force_lvlh, L, D, CL, CD, pro_area = compute_forces_for_entities(data_deputy, loaded_polynomials, [alpha], vv, rr)


        # Calculate magnitude of the combined force
        combined_force_magnitude = np.linalg.norm(combined_force_lvlh)
        combined_forces_magnitudes.append(combined_force_magnitude)


        # Calculate unit vector (direction) for the combined force
        combined_force_direction = combined_force_lvlh / combined_force_magnitude if combined_force_magnitude > 0 else np.array([0, 0, 0])
        combined_forces_directions.append(combined_force_direction)

        # Calculate magnitudes of drag and lift forces
        drag_magnitude = np.linalg.norm(D)
        lift_magnitude = np.linalg.norm(L)
        
        # Append magnitudes to lists
        drag_forces.append(drag_magnitude)
        lift_forces.append(lift_magnitude)
        area_pro.append(pro_area)
        CD_list.append(CD)
        CL_list.append(CL)
        print("Angle " , alpha, " CL ",CL, " CD ",CD)

        # Calculate unit vectors (directions)
        drag_direction = D / drag_magnitude if drag_magnitude > 0 else np.array([0, 0, 0])
        lift_direction = L / lift_magnitude if lift_magnitude > 0 else np.array([0, 0, 0])
        
        # Append directions to lists
        drag_directions.append(drag_direction)
        lift_directions.append(lift_direction)

    # exit()

    # Convert lists to numpy arrays for easier plotting
    drag_forces = np.array(drag_forces)
    lift_forces = np.array(lift_forces)
    area_pro = np.array(area_pro)
    drag_directions = np.array(drag_directions)
    lift_directions = np.array(lift_directions)



    # Convert lists to numpy arrays for easier plotting
    combined_forces_magnitudes = np.array(combined_forces_magnitudes)
    combined_forces_directions = np.array(combined_forces_directions)

    # Plot combined force magnitude with respect to alpha
    plt.figure()
    plt.plot(alpha_degrees, combined_forces_magnitudes, label="Combined Force Magnitude (LVLH)")
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Force Magnitude (N)")
    plt.title("Combined Force Magnitude in LVLH Frame vs. Angle of Attack")
    plt.legend()
    plt.grid()

    # Plot combined force direction components over angles
    plt.figure()
    plt.plot(alpha_degrees, combined_forces_directions[:, 0], label="Combined Force Direction X (LVLH)")
    plt.plot(alpha_degrees, combined_forces_directions[:, 1], label="Combined Force Direction Y (LVLH)")
    plt.plot(alpha_degrees, combined_forces_directions[:, 2], label="Combined Force Direction Z (LVLH)")
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Combined Force Direction (Unit Vector Components)")
    plt.title("Combined Force Direction Components in LVLH Frame vs. Angle of Attack")
    plt.legend()
    plt.grid()


    # Plot drag and lift forces with respect to alpha
    plt.figure()
    plt.plot(alpha_degrees, drag_forces, label="Drag Force Magnitude")
    plt.plot(alpha_degrees, lift_forces, label="Lift Force Magnitude")
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Force (N)")
    plt.title("Drag and Lift Force Magnitudes vs. Angle of Attack")
    plt.legend()
    plt.grid()

    # Plot projected area with respect to alpha
    plt.figure()
    plt.plot(alpha_degrees, area_pro, label="Projected Area")
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Projected Area (m^2)")
    plt.title("Projected Area vs. Angle of Attack")
    plt.legend()
    plt.grid()

    # Plot drag and lift coefficients with respect to alpha
    plt.figure()
    plt.plot(alpha_degrees, CD_list, label="Drag Coefficient")
    plt.plot(alpha_degrees, CL_list, label="Lift Coefficient")
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Coefficient")
    plt.title("Drag and Lift Coefficients vs. Angle of Attack")
    plt.legend()
    plt.grid()

    # Plot drag direction components over angles
    plt.figure()
    plt.plot(alpha_degrees, drag_directions[:, 0], label="Drag Direction X")
    plt.plot(alpha_degrees, drag_directions[:, 1], label="Drag Direction Y")
    plt.plot(alpha_degrees, drag_directions[:, 2], label="Drag Direction Z")
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Drag Direction (Unit Vector Components)")
    plt.title("Drag Force Direction Components vs. Angle of Attack")
    plt.legend()
    plt.grid()

    # Plot lift direction components over angles
    plt.figure()
    plt.plot(alpha_degrees, lift_directions[:, 0], label="Lift Direction X")
    plt.plot(alpha_degrees, lift_directions[:, 1], label="Lift Direction Y")
    plt.plot(alpha_degrees, lift_directions[:, 2], label="Lift Direction Z")
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Lift Direction (Unit Vector Components)")
    plt.title("Lift Force Direction Components vs. Angle of Attack")
    plt.legend()
    plt.grid()

    plt.show()
