import numpy as np

# Define the rotation matrix around the y-axis for the angle of attack
def rotation_matrix_y(alpha):
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([[cos_alpha, 0, sin_alpha],
                     [0, 1, 0],
                     [-sin_alpha, 0, cos_alpha]])

# Surface normals in the body frame
surface_normals = {
    'front': np.array([1, 0, 0]),
    'back': np.array([-1, 0, 0]),
    'top': np.array([0, 1, 0]),
    'bottom': np.array([0, -1, 0]),
    'left': np.array([0, 0, 1]),
    'right': np.array([0, 0, -1])
}

# Function to calculate transformed normals based on angle of attack
def transformed_normals(surface_normals, alpha):
    R_y = rotation_matrix_y(alpha)
    transformed = {}
    for key, normal in surface_normals.items():
        transformed[key] = np.dot(R_y, normal)
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
        lookup_table[alpha_deg] = facing_surfaces
    
    return lookup_table

# Example usage
if __name__ == "__main__":
    # Generate lookup table from 0 to 360 degrees with a step size of 1 degree
    lookup_table = generate_lookup_table(0, 360, 1)
    
    # Example: Query the lookup table for a specific angle of attack (e.g., 10 degrees)
    angle_of_attack = 10
    result = lookup_table.get(angle_of_attack)
    
    print(f"Lookup Table Results for Angle of Attack = {angle_of_attack} degrees:")
    if result:
        for surface, info in result.items():
            print(f"Surface: {surface}, Normal: {info['normal']}, Projected Area: {info['projected_area']}")
    else:
        print("No surfaces are valid at this angle.")
    
    # # Optionally: Save the lookup table to a file
    # import json
    # with open('lookup_table.json', 'w') as f:
    #     json.dump(lookup_table, f, indent=4)
