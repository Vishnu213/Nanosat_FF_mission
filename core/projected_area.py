"""
Nanosat Formation Flying Project

Modified Sentman's equation for CL and CD 

Author:
    Vishnuvardhan Shakthibala
    
"""
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

# Define the rotation matrix around the z-axis for the angle of attack (yaw motion)
def rotation_matrix_z(alpha):
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([[cos_alpha, -sin_alpha, 0],
                     [sin_alpha, cos_alpha, 0],
                     [0, 0, 1]])

# Surface normals in the body frame
surface_normals = {
    'front': np.array([1, 0, 0]),    # Along x-axis
    'back': np.array([-1, 0, 0]),    # Opposite x-axis
    'top': np.array([0, 1, 0]),      # Along y-axis
    'bottom': np.array([0, -1, 0]),  # Opposite y-axis
    'left': np.array([0, 0, 1]),     # Along z-axis
    'right': np.array([0, 0, -1])    # Opposite z-axis
}

# Function to calculate transformed normals based on angle of attack (rotation around z-axis)
def transformed_normals(surface_normals, alpha):
    R_z = rotation_matrix_z(alpha)
    transformed = {}
    for key, normal in surface_normals.items():
        transformed[key] = np.dot(R_z, normal)
    return transformed

# Function to determine which surfaces are facing the velocity direction and calculate projected areas
def surfaces_facing_velocity(transformed_normals, velocity_vector):
    facing_surfaces = {}
    for key, normal in transformed_normals.items():
        dot_product = np.dot(normal, velocity_vector)
        if dot_product > 0:
            if dot_product > 1:
                dot_product = 1
            # Calculate the projected area as the dot product (magnitude of the projection along velocity)
            facing_surfaces[key] = {'normal': normal, 'projected_area': dot_product}
    return facing_surfaces

# Function to generate lookup table for the entire range of angle of attack (0 to 360 degrees)
def generate_lookup_table(min_angle, max_angle, step_size):
    lookup_table = {}
    velocity_vector = np.array([1, 0, 0])  # Assuming velocity along the x-axis
    
    for alpha_deg in range(min_angle, max_angle + 1, step_size):
        alpha_rad = np.radians(alpha_deg)
        transformed = transformed_normals(surface_normals, alpha_rad)
        facing_surfaces = surfaces_facing_velocity(transformed, velocity_vector)
        lookup_table[alpha_deg] = facing_surfaces  # Use integer angles as keys
    
    return lookup_table

# Function to fit polynomials to the lookup table data
def fit_polynomials(lookup_table):
    angles = np.array(list(lookup_table.keys()), dtype=float)
    
    # Initialize storage for the polynomial coefficients for each surface
    poly_coeffs = {
        'front': {'normal_x': [], 'normal_y': [], 'normal_z': [], 'area': []},
        'back': {'normal_x': [], 'normal_y': [], 'normal_z': [], 'area': []},
        'top': {'normal_x': [], 'normal_y': [], 'normal_z': [], 'area': []},
        'bottom': {'normal_x': [], 'normal_y': [], 'normal_z': [], 'area': []},
        'left': {'normal_x': [], 'normal_y': [], 'normal_z': [], 'area': []},
        'right': {'normal_x': [], 'normal_y': [], 'normal_z': [], 'area': []}
    }
    
    for surface in poly_coeffs.keys():
        # Prepare lists for fitting
        normal_x = []
        normal_y = []
        normal_z = []
        areas = []
        
        for angle in angles:
            angle = int(angle)  # Ensure angle is used as an integer to match the lookup_table keys
            if surface in lookup_table[angle]:  # Accessing integer key directly
                normal = lookup_table[angle][surface]['normal']
                area = lookup_table[angle][surface]['projected_area']
                
                normal_x.append(normal[0])
                normal_y.append(normal[1])
                normal_z.append(normal[2])
                if area > 1:
                    area = 1
                areas.append(area)
            else:
                # If the surface is not valid at this angle, use 0 for normal and area
                normal_x.append(0)
                normal_y.append(0)
                normal_z.append(0)
                areas.append(0)
        
        # Fit polynomials for each component
        poly_coeffs[surface]['normal_x'] = np.polyfit(angles, normal_x, deg=5)
        poly_coeffs[surface]['normal_y'] = np.polyfit(angles, normal_y, deg=5)
        poly_coeffs[surface]['normal_z'] = np.polyfit(angles, normal_z, deg=5)
        poly_coeffs[surface]['area'] = np.polyfit(angles, areas, deg=5)
    
    return poly_coeffs

# Function to save the polynomial coefficients for future use
def save_polynomials(poly_coeffs, filename='polynomials.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(poly_coeffs, f)

# Function to load the polynomial coefficients
def load_polynomials(filename='polynomials.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Define a function that returns the normal vectors and projected areas for a given angle of attack
def lookup_surface_properties(angle, poly_coeffs):
    surfaces_data = []
    
    for surface, coeffs in poly_coeffs.items():
        normal_x = np.polyval(coeffs['normal_x'], angle)
        normal_y = np.polyval(coeffs['normal_y'], angle)
        normal_z = np.polyval(coeffs['normal_z'], angle)
        projected_area = np.polyval(coeffs['area'], angle)
        
        if projected_area > 0:  # Only consider valid surfaces with positive projected area
            surfaces_data.append([normal_x, normal_y, normal_z, projected_area])
    
    return np.array(surfaces_data)

# Convert NumPy arrays to Python lists for JSON serialization
def convert_to_serializable(lookup_table):
    serializable_table = {}
    for angle, surfaces in lookup_table.items():
        serializable_table[angle] = {}
        for surface, data in surfaces.items():
            serializable_table[angle][surface] = {
                'normal': data['normal'].tolist(),  # Convert NumPy array to list
                'projected_area': data['projected_area']  # Projected area is already a float
            }
    return serializable_table

# Define a function that returns the normal vectors and projected areas for a given angle of attack
def lookup_table_surface_properties(angle, lookup_table):
    surfaces_data = []
    angle_str = str(int(angle))  # Convert angle to a string to match the keys in lookup_table

    # Check if the angle exists in the lookup table
    if angle_str in lookup_table:
        for surface, properties in lookup_table[angle_str].items():
            normal = properties['normal']
            projected_area = properties['projected_area']
            
            if projected_area > 0:  # Only consider valid surfaces with positive projected area
                surfaces_data.append([normal[0], normal[1], normal[2], projected_area])
    else:
        print(f"Angle {angle_str} not found in lookup table.")
    
    return np.array(surfaces_data)

        # Load the JSON lookup table
def load_lookup_table(filename):
        with open(filename, 'r') as f:
            return json.load(f)


# Generate lookup table for angles from 0 to 360 degrees with a step size of 1 degree
lookup_table_projected_area = generate_lookup_table(0, 360, 1)

# Example usage: Generate lookup table, fit polynomials, and save for future use
if __name__ == "__main__":
    # Generate lookup table for angles from 0 to 360 degrees with a step size of 1 degree
    lookup_table = generate_lookup_table(0, 360, 1)
    
    # Convert the lookup table to a JSON-serializable format
    serializable_lookup_table = convert_to_serializable(lookup_table)
    
    # Save the serializable lookup table to a JSON file
    with open('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nano_sat_casadi\\Nanosat_FF_mission\\helper_files\\lookup_table_projected_area.json', 'w') as f:
        json.dump(serializable_lookup_table, f, indent=4)
    
    # # Fit polynomials to the lookup table
    # poly_coeffs = fit_polynomials(lookup_table)
    
    # # Save the polynomial functions to a file
    # save_polynomials(poly_coeffs, 'C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nano_sat_casadi\\Nanosat_FF_mission\\helper_files\\polynomials_projected_area_test.pkl')
    
    # # Later, you can load the polynomial functions from the file
    # loaded_polynomials = load_polynomials('C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nano_sat_casadi\\Nanosat_FF_mission\\helper_files\\polynomials_projected_area.pkl')



    # Example usage after loading
    lookup_table_path = 'C:\\Users\\vishn\\Desktop\\My_stuffs\\Projects\\SDCS group\\Research\\Nano_sat_casadi\\Nanosat_FF_mission\\helper_files\\lookup_table_projected_area.json'
    loaded_lookup_table = load_lookup_table(lookup_table_path)

    # Test the lookup for an angle of attack (in degrees, e.g., 10 degrees)
    test_angle = 0
    surface_properties = lookup_table_surface_properties(test_angle, loaded_lookup_table)
    
    print(f"Surface properties for angle of attack = {test_angle} degrees:")
    print("Normal vectors and projected areas:")
    print(surface_properties)
    exit()

    # Initialize lists to store the real and fitted values for each surface
    real_projected_areas_front = []
    fitted_projected_areas_front = []
    real_back_projected_areas = []
    fitted_back_projected_areas = []
    angles = np.linspace(0, 360, 361)
    # Loop through each angle to extract real values and compute fitted values for the front surface
    for angle in angles:
        # Real value from the lookup table
        if 'front' in lookup_table[angle]:  # Check if 'front' surface data is available at this angle
            real_projected_areas_front.append(lookup_table[angle]['front']['projected_area'])
        else:
            real_projected_areas_front.append(0)  # Use 0 if the surface is not facing the velocity direction

        if 'back' in lookup_table[angle]:
            real_back_projected_areas.append(lookup_table[angle]['back']['projected_area'])
        else:
            real_back_projected_areas.append(0)


        # Fitted value using polynomial coefficients
        fitted_projected_area = np.polyval(poly_coeffs['front']['area'], angle)
        fitted_projected_areas_front.append(fitted_projected_area)

        fitted_back_projected_area = np.polyval(poly_coeffs['back']['area'], angle)
        fitted_back_projected_areas.append(fitted_back_projected_area)

    # Plot real and fitted values for the front surface
    plt.figure()
    plt.plot(angles, real_projected_areas_front, label='Real Projected Area (Front)', linestyle='--')
    plt.plot(angles, fitted_projected_areas_front, label='Fitted Projected Area (Front)', linestyle='-')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Projected Area')
    plt.title('Comparison of Real and Fitted Projected Areas for Front Surface')
    plt.legend()

    plt.figure()
    # Plot real and fitted values for the front surface
    plt.plot(angles, real_back_projected_areas, label='Real Projected Area (BACK)', linestyle='--')
    plt.plot(angles, fitted_back_projected_areas, label='Fitted Projected Area (BACK)', linestyle='-')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Projected Area')
    plt.title('Comparison of Real and Fitted Projected Areas for back Surface')
    plt.legend()

    plt.show()



    # # Optional: plot the projected area as a function of angle for a given surface
    # angles = np.linspace(0, 360, 361)
    # projected_areas_front = [np.polyval(loaded_polynomials['front']['area'], angle) for angle in angles]
    # projected_areas_back = [np.polyval(loaded_polynomials['back']['area'], angle) for angle in angles]
    

    # plt.plot(angles, projected_areas_front, label='Front Surface Projected Area')
    # plt.plot(angles, projected_areas_back, label='Back Surface Projected Area')
    # plt.xlabel('Angle of Attack (degrees)')
    # plt.ylabel('Projected Area')
    # plt.title('Projected Area vs Angle of Attack for Front Surface')
    # plt.legend()
    # plt.show()
