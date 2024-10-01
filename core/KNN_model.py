from scipy.interpolate import griddata
import numpy as np
import pickle
import matplotlib.pyplot as plt
from nrlmsise00 import msise_model
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.neighbors import KDTree
import time
from scipy.interpolate import NearestNDInterpolator


# Load the flattened arrays
# data = np.load('flattened_arrays.npz')
# u_flat = data['u_flat']
# r_flat = data['r_flat']
# i_flat = data['i_flat']
# density_flat = data['density_ref_flat']
# M_flat = data['M_flat']
# T_flat = data['T_flat']

# Constants
Re = 6371.0  # Earth's radius in km

# Function to get density from NRLMSISE-00
def get_density_from_nrlmsise00(alt, lat, lon, date):
    densities, _ = msise_model(
        time=date, 
        alt=alt, 
        lat=lat,  
        lon=lon,  
        f107a=150.0,  
        f107=150.0,  
        ap=4.0  
    )
    return densities[5]

# Generate B-spline model for density, molar mass, and temperature using griddata
def create_b_spline_model(r_flat, u_flat, i_flat, density_flat, M_flat, T_flat):
    points = np.vstack([r_flat, u_flat, i_flat]).T  # Stack points for interpolation
    # Create interpolators using griddata
    density_interp = lambda r, u, i: griddata(points, density_flat, (r, u, i), method='linear')
    M_interp = lambda r, u, i: griddata(points, M_flat, (r, u, i), method='linear')
    T_interp = lambda r, u, i: griddata(points, T_flat, (r, u, i), method='linear')
    return density_interp, M_interp, T_interp

# B-spline-like interpolation lookup function using griddata
def lookup_b_spline(density_interp, M_interp, T_interp, r_target, u_target, i_target):
    density = density_interp(r_target, u_target, i_target)
    molar_mass = M_interp(r_target, u_target, i_target)
    temperature = T_interp(r_target, u_target, i_target)
    return density, molar_mass, temperature


# Function to query the KDTree and get values
def query_knn(r_target, u_target, i_target, kdtree, density_flat, M_flat, T_flat):
    # Find the nearest neighbor using KDTree
    dist, idx = kdtree.query([[r_target, u_target, i_target]], k=1)  # k=1 for nearest neighbor

    # Use the found index to get the corresponding values
    interpolated_density = density_flat[idx[0][0]]
    interpolated_M = M_flat[idx[0][0]]
    interpolated_T = T_flat[idx[0][0]]

    return interpolated_density, interpolated_M, interpolated_T

# Function to compare density with B-spline interpolation and NRLMSISE-00
def compare_density_b_spline_nrlmsise(lat, lon, altitude, argument_of_latitude, inclination, 
                                        density_flat , M_flat, T_flat,kdtree):
    density_ref_100_days = []
    density_spline_100_days = []
    dates = []

    start_date = datetime(2024, 8, 30, 12, 0, 0)

    for day in range(3):
        current_date = start_date + timedelta(days=day)
        alt = altitude
        u = argument_of_latitude
        i = inclination

        # Get density from NRLMSISE-00
        density_ref = get_density_from_nrlmsise00(alt, lat, lon, current_date)
        density_ref_100_days.append(density_ref)

        # Get density from B-spline interpolation
        r_target = alt + Re  # Altitude in km
        # Find the nearest point for a given (r_target, u_target, i_target)
        # Use the found index to get r, u, and i values
        start_date = datetime.now()

        # Start timing the lookup table extraction
        start_time = time.time()  # Start the timer
        # Find the nearest neighbor using KDTree
        interpolated_density, interpolated_M, interpolated_T = query_knn(r_target, u, i, kdtree, density_flat, M_flat, T_flat)
        
        # End timing the lookup table extraction
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Lookup table extraction took {elapsed_time:.6f} seconds for 365 days.")
        density_spline_100_days.append(interpolated_density)

        dates.append(current_date)

    return dates, density_ref_100_days, density_spline_100_days




# Main function
if __name__ == "__main__":


    # # Combine r, u, i into a points matrix for KDTree
    # points = np.vstack([r_flat, u_flat, i_flat]).T

    # # Create a KDTree for fast k-nearest neighbor searches
    # kdtree = KDTree(points)

    # # Save the KDTree and data arrays using pickle
    # with open('knn_model.pkl', 'wb') as f:
    #     pickle.dump(kdtree, f)  # Save the KDTree
    #     pickle.dump(density_flat, f)  # Save the density values
    #     pickle.dump(M_flat, f)  # Save the molar mass values
    #     pickle.dump(T_flat, f)  # Save the temperature values

    # print("KNN model and data saved successfully.")

    # Load the saved KDTree and associated data
    with open('knn_model.pkl', 'rb') as f:
        kdtree = pickle.load(f)  # Load the KDTree
        density_flat = pickle.load(f)  # Load the density values
        M_flat = pickle.load(f)  # Load the molar mass values
        T_flat = pickle.load(f)  # Load the temperature values


    # Parameters for comparison
    altitude = 350  # Altitude in km
    argument_of_latitude = np.pi / 4  # Example argument of latitude
    inclination = np.pi / 6  # Example inclination
    lat = 0.0  # Latitude
    lon = 0.0  # Longitude

    # Compare density from NRLMSISE-00 and B-spline for 100 days
    dates, density_ref_100_days, density_spline_100_days = compare_density_b_spline_nrlmsise(
        lat, lon, altitude, argument_of_latitude, inclination, density_flat, M_flat, T_flat, kdtree)

    # Plot the results for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(dates, density_ref_100_days, label='NRLMSISE-00 Density', color='b')
    plt.plot(dates, density_spline_100_days, label='B-Spline Interpolated Density', color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Density (kg/m^3)')
    plt.title('Density Comparison between NRLMSISE-00 and B-Spline Interpolation over 100 Days')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()