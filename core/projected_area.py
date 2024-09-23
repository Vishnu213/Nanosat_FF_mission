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


# Example usage: Generate lookup table, fit polynomials, and save for future use
if __name__ == "__main__":
    # Generate lookup table for angles from 0 to 360 degrees with a step size of 1 degree
    lookup_table = generate_lookup_table(0, 360, 1)
    
    # Convert the lookup table to a JSON-serializable format
    serializable_lookup_table = convert_to_serializable(lookup_table)
    
    # Save the serializable lookup table to a JSON file
    with open('../helper_files/lookup_table_projected_area.json', 'w') as f:
        json.dump(serializable_lookup_table, f, indent=4)
    
    # Fit polynomials to the lookup table
    poly_coeffs = fit_polynomials(lookup_table)
    
    # Save the polynomial functions to a file
    save_polynomials(poly_coeffs, '../helper_files/polynomials_projected_area.pkl')
    
    # Later, you can load the polynomial functions from the file
    loaded_polynomials = load_polynomials('../helper_files/polynomials_projected_area.pkl')
    
    # Test the lookup for an angle of attack (in degrees, e.g., 10 degrees)
    test_angle = 10
    surface_properties = lookup_surface_properties(test_angle, loaded_polynomials)
    
    print(f"Surface properties for angle of attack = {test_angle} degrees:")
    print("Normal vectors and projected areas:")
    print(surface_properties)
    
    # Optional: plot the projected area as a function of angle for a given surface
    angles = np.linspace(0, 360, 361)
    projected_areas_front = [np.polyval(loaded_polynomials['front']['area'], angle) for angle in angles]
    
    plt.plot(angles, projected_areas_front, label='Front Surface Projected Area')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Projected Area')
    plt.title('Projected Area vs Angle of Attack for Front Surface')
    plt.legend()
    plt.show()
