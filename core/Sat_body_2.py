"""
Nanosat Formation Flying Project

Modified Sentman's equation for CL and CD 

Author:
    Vishnuvardhan Shakthibala
    
"""

import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the lookup table from the JSON file (generated from the previous code)
with open('../helper_files/lookup_table.json', 'r') as f:
    lookup_table = json.load(f)

# Define a function to fit a polynomial for each component of the normal vector and projected area
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
            if surface in lookup_table[str(int(angle))]:
                normal = lookup_table[str(int(angle))][surface]['normal']
                area = lookup_table[str(int(angle))][surface]['projected_area']
                
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

# Generate the polynomial fits for the lookup table data
poly_coeffs = fit_polynomials(lookup_table)

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

# Example usage: Query surface properties for a specific angle of attack
if __name__ == "__main__":
    # Test the lookup for an angle of attack (in degrees, e.g., 10 degrees)
    test_angle = 10
    surface_properties = lookup_surface_properties(test_angle, poly_coeffs)
    
    print(f"Surface properties for angle of attack = {test_angle} degrees:")
    print("Normal vectors and projected areas:")
    print(surface_properties)
    
    # Optional: plot the projected area as a function of angle for a given surface
    angles = np.linspace(0, 360, 361)
    projected_areas_front = [np.polyval(poly_coeffs['front']['area'], angle) for angle in angles]
    
    plt.plot(angles, projected_areas_front, label='Front Surface Projected Area')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Projected Area')
    plt.title('Projected Area vs Angle of Attack for Front Surface')
    plt.legend()
    plt.show()
