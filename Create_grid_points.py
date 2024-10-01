import numpy as np
from nrlmsise00 import msise_model
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pymap3d as pm
import pickle
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta
import math

deg2rad = np.pi / 180

def C1(theta):
    C=np.array([[1,0,0],
                  [0,np.cos(theta),np.sin(theta)],
                  [0,-np.sin(theta),np.cos(theta)]])
    return C

def C2(theta):
    C=np.array([[np.cos(theta),0,-np.sin(theta)],
                  [0,1,0],
                  [np.sin(theta),0,np.cos(theta)]])
    return C

def C3(theta):
    C=np.array([[np.cos(theta),np.sin(theta),0],
                  [-np.sin(theta),np.cos(theta),0],
                  [0,0,1]])
    return C

# references frame transformations

# Orbit Perifocal to Earth centric inertial frame
def PQW2ECI(OM,om,i):
    C=np.matmul(C3(-OM),np.matmul(C1(-i),C3(-om)))
    return C

def kep2car(COE, mu):
    
    a = COE[0]  # Specific angular momentum
    e = COE[1]  # Eccentricity
    i = COE[2]  # Inclination
    OM = COE[3]  # Right Ascension of the Ascending Node (RAAN)
    u = COE[4]   # Argument of latitude
    om = 0  # Set to 0 for simplicity, as it doesn't affect the result
    TA = u  # Argument of latitude equals the true anomaly for circular orbits
    h=np.sqrt(mu*a*(1-e**2))
    # Position and velocity in perifocal frame
    rp = (h ** 2 / mu) * (1 / (1 + e * np.cos(TA))) * (np.cos(TA) * np.array([1, 0, 0])
        + np.sin(TA) * np.array([0, 1, 0]))
    
    vp = (mu / h) * (-np.sin(TA) * np.array([1, 0, 0]) 
        + (e + np.cos(TA)) * np.array([0, 1, 0]))

    # Convert to ECI (Earth-Centered Inertial) frame
    RR = np.matmul(PQW2ECI(OM, om, i), np.transpose(rp))
    VV = np.matmul(PQW2ECI(OM, om, i), np.transpose(vp))

    return RR, VV


def ecef_to_latlon(x, y, z):
    # Longitude calculation
    lon = np.arctan2(y, x)  # In radians

    # Latitude calculation (spherical Earth approximation)
    hyp = np.sqrt(x**2 + y**2)  # Hypotenuse in the xy-plane
    lat = np.arctan2(z, hyp)  # In radians

    # Convert to degrees
    lon = np.degrees(lon)
    lat = np.degrees(lat)

    return lat, lon

mu = 398600.4418  # Earth's gravitational parameter in km^3/s^2
# COE = [7000, 0.2, 55*deg2rad , 20*deg2rad, 0]  # Circular orbit at 7000 km altitude
# t_epoch = datetime(2022, 1, 1, 0, 0, 0)  # Epoch time
# rr,vv=kep2car(COE, mu)
# rr_ecef = np.zeros(3)
# rr_ecef[0],rr_ecef[1],rr_ecef[2] = pm.eci2ecef(rr[0],rr[1],rr[2], t_epoch)

# Constants
Re = 6371.0  # Earth's radius in km
ee = 0.0167  # Earth's orbital eccentricity

# Sidereal day length
SIDEREAL_DAY_SECONDS = 86164.1  # In seconds, about 4 minutes shorter than a solar day

# The analytic density model to fit the data with a lower bound
def analytic_density_model(u, r, i, A, B, C, D):

    Re = 6378.137  # Earth's radius
    ee = 0.0167  # Earth's eccentricity
    lower_bound=1e-18
    upper_bound=1e-6
    # Exponential term
    exp_arg = (r - Re) * np.sqrt(1 - ee**2 * np.sin(i)**2 * np.sin(u)**2) / D
    exp_arg = np.clip(exp_arg, -1, 1)  # Prevent overflow

    # Density model
    density = A * (1 + B * np.cos(u - C)) * np.exp(exp_arg)

    # Clip the density to enforce the lower bound
    return density  # None means no upper bound
# Objective function for MSE with log scale

# def mse_loss(params, u, r, i, density_ref):
#     A, B, C, D = params
#     density_pred = analytic_density_model(u, r, i, A, B, C, D)
#     mse = np.mean((np.log(density_pred) - np.log(density_ref)) ** 2)  # Log-scale fitting
#     return mse

# Molar masses of different gases (in g/mol)
molar_masses = {
    'He': 4.0026,  # Helium
    'O': 16.0,     # Oxygen atom
    'N2': 28.0134, # Nitrogen molecule
    'O2': 32.0,    # Oxygen molecule
    'Ar': 39.948,  # Argon
    'H': 1.008,    # Hydrogen atom
}

# Function to get density from NRLMSISE-00
def get_density_from_nrlmsise00(alt, lat, lon, date):
    doy = date.timetuple().tm_yday
    sec = date.hour * 3600 + date.minute * 60 + date.second
    lst = calculate_lst(lon, date)
    densities, t = msise_model(
        time=date, 
        alt=alt, 
        lat=lat,  
        lon=lon,  
        f107a=150.0,  
        f107=150.0,  
        ap=4.0,  
    )


    # Densities (in g/cm³)
    density_He = densities[0]
    density_O = densities[1]
    density_N2 = densities[2]
    density_O2 = densities[3]
    density_Ar = densities[4]
    density_H = densities[6]

    # Calculate total number density
    total_density = density_He + density_O + density_N2 + density_O2 + density_Ar + density_H

    # Calculate mean molar mass in kg/mol
    mean_molar_mass = (density_He * molar_masses['He'] + density_O * molar_masses['O'] + density_N2 * molar_masses['N2'] +
                        density_O2 * molar_masses['O2'] + density_Ar * molar_masses['Ar'] + density_H * molar_masses['H']) / total_density


    # Temperature at the altitude (in K)
    temp_alt =t[1]  # Temperature at the given altitude (not exospheric temperature)

    return densities[5], mean_molar_mass , temp_alt

# Function to calculate Local Sidereal Time (LST) for a given longitude and time
def calculate_lst(longitude, time):
    # Greenwich Mean Sidereal Time (GMST) calculation
    jd = (time - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / (24 * 3600) + 2451545.0  # Julian date
    t = (jd - 2451545.0) / 36525.0  # Centuries since J2000.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + t**2 * (0.000387933 - t / 38710000)
    gmst = gmst % 360.0  # GMST in degrees
    lst = (gmst + longitude) % 360.0  # Convert to local sidereal time
    return lst

def generate_grid_points_and_convert_with_raan(
    ecc_points=5, inc_points=5, u_points=5, a_points=5, raan_points=5, time_points=50):
    """
    Generate grid points and convert them into altitude, latitude, longitude, and date.
    
    Parameters:
    - ecc_points (int): Number of grid points for eccentricity.
    - inc_points (int): Number of grid points for inclination.
    - u_points (int): Number of grid points for argument of latitude.
    - a_points (int): Number of grid points for semi-major axis.
    - raan_points (int): Number of grid points for RAAN.
    - time_points (int): Number of time steps over the simulation period.
    
    Returns:
    - (grid1, grid2): Tuple containing generated grids (e.g., eccentricity, inclination, etc.) and converted values (altitude, latitude, longitude, dates).
    """
    # Define constants and parameters
    Re = 6378.137  # Earth's radius in km
    mu = 398600.4418  # Earth's gravitational parameter in km^3/s^2

    # Define grid ranges
    eccentricities = np.linspace(0, 0.99, ecc_points)  # Eccentricity grid
    inclinations = np.linspace(0.01,np.pi, inc_points)  # Inclination grid (radians)
    arguments_of_latitude = np.linspace(0, 2 * np.pi, u_points)  # Argument of latitude (radians)
    a = np.linspace(Re + 100, Re + 500, a_points)  # Semi-major axis (150 km to 600 km above Earth's surface)
    raans = np.linspace(0, 2 * np.pi, raan_points)  # RAAN grid (radians)
    time_steps = np.linspace(0, 2 * 365 * 24 * 3600, time_points)  # Time steps over 5 years (seconds)

    initial_date = datetime.now()

    # Create meshgrid for all combinations of eccentricity, inclination, argument of latitude, altitude, RAAN, and time
    e_grid, i_grid, u_grid, a_grid, omega_grid, t_grid = np.meshgrid(
        eccentricities, inclinations, arguments_of_latitude, a, raans, time_steps, indexing='ij'
    )

    # Initialize lists for valid grid points
    valid_altitude = []
    valid_latitude = []
    valid_longitude = []
    valid_dates = []
    valid_e_grid = []
    valid_i_grid = []
    valid_u_grid = []
    valid_a_grid = []
    valid_omega_grid = []
    valid_r_grid = []



    initial_date = datetime.now()

    # Loop over the grid to calculate altitude, latitude, longitude, date, and r
    for idx, _ in np.ndenumerate(e_grid):
        # Argument of latitude, eccentricity, semi-major axis, RAAN
        u = u_grid[idx]
        e = e_grid[idx]
        a = a_grid[idx]

        # Calculate the position vector magnitude using the orbit equation
        r = a * (1 - e**2) / (1 + e * np.cos(u))
        omega = omega_grid[idx]
        i = i_grid[idx]
        t = initial_date+timedelta(seconds=t_grid[idx])


        COE = [a, e, i, omega, u]  # Circular orbit at 7000 km altitude
        t_epoch = datetime(2022, 1, 1, 0, 0, 0)  # Epoch time
        rr,vv=kep2car(COE, mu)
        rr_ecef = np.zeros(3)
        rr_ecef[0],rr_ecef[1],rr_ecef[2] = pm.eci2ecef(rr[0],rr[1],rr[2], t)

        # Convert ECEF to latitude and longitude
        lat, lon = ecef_to_latlon(rr_ecef[0], rr_ecef[1], rr_ecef[2])

        # Update the altitude (r - Earth's radius)
        alt = np.linalg.norm(rr_ecef) - Re 

        # Check if the altitude is above 150 km
        if alt >= 150:

            # Store valid grid points
            valid_altitude.append(alt)
            valid_latitude.append(lat)
            valid_longitude.append(lon)
            valid_dates.append(t)
            valid_e_grid.append(e)
            valid_i_grid.append(i)
            valid_u_grid.append(u)
            valid_a_grid.append(a)
            valid_omega_grid.append(omega)
            valid_r_grid.append(r)

    # Convert lists to numpy arrays
    altitude_filtered = np.array(valid_altitude)
    latitude_filtered = np.array(valid_latitude)
    longitude_filtered = np.array(valid_longitude)
    dates_filtered = np.array(valid_dates)
    e_grid_filtered = np.array(valid_e_grid)
    i_grid_filtered = np.array(valid_i_grid)
    u_grid_filtered = np.array(valid_u_grid)
    a_grid_filtered = np.array(valid_a_grid)
    omega_grid_filtered = np.array(valid_omega_grid)
    r_grid_filtered = np.array(valid_r_grid)

    return (e_grid_filtered, i_grid_filtered, u_grid_filtered, a_grid_filtered, omega_grid_filtered, r_grid_filtered), (altitude_filtered, latitude_filtered, longitude_filtered, dates_filtered)


# Objective function for MSE with log scale
def mse_loss(params, u, r, i, density_ref):
    scaling_factor = 1e13  # Scaling factor for density values - to make it have reasonable values
    A, B, C, D = params
    density_pred = analytic_density_model(u, r, i, A, B, C, D)

    # Apply the scaling factor to the predicted density
    density_pred_scaled = density_pred 

    # Replace `NaN` or `inf` in predicted density
    density_pred_scaled = np.nan_to_num(density_pred_scaled, nan=0.0, posinf=1e12, neginf=1e12)

    # Calculate the log-scale MSE with scaled values
    #mse = np.mean((np.log(density_pred_scaled) - np.log(density_ref)) ** 2)
    # Calculate the log-scale MSE with scaled values
    mse = np.mean((density_pred_scaled - density_ref) ** 2)
    return mse

# Function to save coefficients to a pickle file
def save_coefficients_to_pickle(coefficients, filename='coefficients.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(coefficients, f)

# Function to load coefficients from a pickle file
def load_coefficients_from_pickle(filename='coefficients.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Generate densities for 10 days to compare reference and analytic model
def compare_density_10_days(lat, lon, altitude, argument_of_latitude, inclination, A_fit, B_fit, C_fit, D_fit):
    density_ref_10_days = []
    density_analytic_10_days = []
    dates = []
    
    start_date = datetime(2024, 8, 30, 12, 0, 0)  # Start at solar peak date
    
    for day in range(10):
        current_date = start_date + relativedelta(years=day)
        alt = altitude  # Altitude in km
        u = argument_of_latitude  # Argument of latitude
        i = inclination  # Inclination in radians
        
        # Get density from NRLMSISE-00
        density_ref = get_density_from_nrlmsise00(alt, lat, lon, current_date)
        density_ref_10_days.append(density_ref)
        
        # Calculate density using analytic model with fitted parameters
        density_analytic = analytic_density_model(u, alt + Re, i, A_fit, B_fit, C_fit, D_fit)
        density_analytic_10_days.append(density_analytic)
        
        # Save the date for plotting
        dates.append(current_date)

    return dates, density_ref_10_days, density_analytic_10_days


def load_coefficients_from_pickle(file_path='coefficients.pkl'):
    """
    Load coefficients from a pickle file.

    Parameters:
    file_path (str): The path to the pickle file. Default is 'coefficients.pkl'.

    Returns:
    object: The loaded coefficients.
    """
    with open(file_path, 'rb') as file:
        coefficients = pickle.load(file)
    return coefficients

def debug_callback(x):
    print("Current guess:", x)
    loss_value = mse_loss(x, u_flat, r_flat, i_flat, density_ref_flat)
    print("Current loss:", loss_value)


# The analytic density model to fit the data
def analytic_density_model_for_fit(xdata, A, B, C, D):
    r, u, i = xdata  # Unpack r, u, i from the tuple
    # Earth's radius and eccentricity constants
    Re = 6378.137
    ee = 0.0167

    # Exponential term
    exp_arg = (r - Re) * np.sqrt(1 - ee**2 * np.sin(i)**2 * np.sin(u)**2) / D
    exp_arg = np.clip(exp_arg, -100, 100)  # Prevent overflow in the exponent

    # Density model
    return A * (1 + B * np.cos(u - C)) * np.exp(exp_arg) 

    # Function to calculate density with the fitted parameters
def predict_density(r, u, i, A, B, C, D):
    Re = 6378.137  # Earth's radius
    ee = 0.0167  # Earth's eccentricity
    scaling_factor = 1e13  # Scaling factor for density values
    # Exponential term
    exp_arg = (r - Re) * np.sqrt(1 - ee**2 * np.sin(i)**2 * np.sin(u)**2) / D
    exp_arg = np.clip(exp_arg, -100, 100)  # Prevent overflow

    # Density model
    return A * (1 + B * np.cos(u - C)) * np.exp(exp_arg) 





# Example of using the fit and saving the coefficients
if __name__ == "__main__":

#######################################################################################
# generating grid points and saving it as numpy array
    
    print("Generating grid points and converting to altitude, latitude, longitude, and date...")
    # Flatten the arrays for fitting

    # Generate the grid and converted values with RAAN
    dat1, dat2 = generate_grid_points_and_convert_with_raan( ecc_points=5, inc_points=5, u_points=30, a_points=8, raan_points=5, time_points=5)

    print("Grid points generated and converted.")

    altitude, latitude, longitude, dates = dat2
    e_grid, i_grid, u_grid,a_grid,omega_grid, r_grid= dat1

    density_ref = np.zeros_like(altitude)
    total_molar_density = np.zeros_like(altitude)
    temps = np.zeros_like(altitude)

    for idx, _ in np.ndenumerate(density_ref):
        alt = altitude[idx]  # Use the altitude directly from the generated grid
        lat = latitude[idx]  # Use latitude from the generated grid
        lon = longitude[idx]  # Use longitude from the generated grid
        date = dates[idx]  # Use the date from the generated grid

        # Call NRLMSISE-00 to get density
        density_ref[idx], total_molar_density[idx], temps[idx] = get_density_from_nrlmsise00(alt, lat, lon, date)



    print("Reference densities generated.")
    # Flatten the arrays for fitting
    u_flat = u_grid.flatten()
    a_flat = a_grid.flatten()
    i_flat = i_grid.flatten()
    omega_flat = omega_grid.flatten()
    e_flat = e_grid.flatten()
    r_flat = r_grid.flatten()


    density_ref_flat = density_ref.flatten()
    M_flat = total_molar_density.flatten()
    T_flat = temps.flatten()
    dates_flat = dates.flatten()

    print("Flattened arrays for fitting.")

    np.savez('flattened_arrays_30_u.npz', u_flat=u_flat, a_flat=a_flat, i_flat=i_flat,
    omega_flat=omega_flat, e_flat=e_flat, r_flat=r_flat ,
    density_ref_flat=density_ref_flat, M_flat=M_flat, T_flat=T_flat)


# ######################################################################
# ######## Load the flattened arrays for fitting - fittig the mode here

#     scaling_factor = 1e13 # Scaling factor for density values - to make it have reasonable values
#     print("Loading flattened arrays for fitting...")
#     # Load the flattened arrays
#     data = np.load('flattened_arrays.npz')

#     # Retrieve individual arrays
#     u_flat = data['u_flat']
#     a_flat = data['a_flat']
#     r_flat = data['r_flat']
#     i_flat = data['i_flat']
#     omega_flat = data['omega_flat']
#     e_flat = data['e_flat']
#     density_ref_flat = data['density_ref_flat']

#     density_ref_flat = density_ref_flat 
#     density_ref_flat = np.nan_to_num(density_ref_flat, nan=0.0, posinf=1e12, neginf=1e12)

#     print(np.isnan(density_ref_flat).any(), np.isinf(density_ref_flat).any())
#     print(np.isnan(r_flat).any(), np.isinf(r_flat).any())
#     print(np.isnan(i_flat).any(), np.isinf(i_flat).any())
#     print(np.isnan(u_flat).any(), np.isinf(u_flat).any())

#     # Initial guess for parameters
#     load_coefficients = load_coefficients_from_pickle('./coefficients.pkl')
#     initial_guess = [load_coefficients['A'], load_coefficients['B'], load_coefficients['C'], load_coefficients['D']]
#     initial_guess = [0.1, 0.1, 0.5, 0.5]
#     print("Initial guess for parameters:", initial_guess)

#     # Initial guess for A, B, C, D
#     initial_guess = [1.0e-9, 0.1, 0.5, 0.5]  # You can adjust these
#     print("estimating the parameters using curve_fit...")
#     # Fitting function using curve_fit
# # Fitting function using curve_fit

#     # Example: A should be between 1e-12 and 1e-6, B between 0 and 1, C between 0 and 2π, D between 1e-1 and 1e3
#     lower_bounds = [-1, 0, 0, -20]
#     upper_bounds = [1, 1, 2*np.pi, 20]

#     # Perform curve fitting with bounds
#     popt, pcov = curve_fit(
#         analytic_density_model_for_fit, 
#         (r_flat, u_flat, i_flat),  # Pass r, u, i as a tuple
#         density_ref_flat, 
#         p0=initial_guess, 
#         bounds=(lower_bounds, upper_bounds)  # Apply the bounds here
#     )

#     # Extract the estimated parameters
#     A_fit, B_fit, C_fit, D_fit = popt

#     print(f"Initial guess for minimizer: A={A_fit}, B={B_fit}, C={C_fit}, D={D_fit}")

#     intial_guess = [A_fit, B_fit, C_fit, D_fit]

#     initial_guess = [1.0e-6, 0.1, 0.5, 0.5]  # You can adjust these

#     # Use the fitted values to predict density
#     predicted_density = predict_density(r_flat, u_flat, i_flat, A_fit, B_fit, C_fit, D_fit)

#     # Calculate Mean Squared Error (MSE)
#     mse = mean_squared_error(density_ref_flat, predicted_density)
#     print(f"Mean Squared Error: {mse}")

#     # Plot the actual vs predicted density values
#     plt.figure(figsize=(10, 6))
#     plt.plot(density_ref_flat, label="Actual Density", color='b', marker='o')
#     plt.plot(predicted_density, label="Predicted Density", color='r', linestyle='--')
#     plt.xlabel('Data Point Index')
#     plt.ylabel('Density')
#     plt.title('Actual vs Predicted Density')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     bounds = [(-1, 1), (-20, 20), (0, 2*np.pi), (-20, 20)]

#     print("Fitting the analytic model using log-scale MSE and L-BFGS-B...")
#     # Fitting the analytic model using log-scale MSE and L-BFGS-B
#     result = minimize(mse_loss, initial_guess, args=(u_flat, r_flat, i_flat, density_ref_flat),
#                         method='TNC', bounds = bounds, callback=debug_callback,
#                         options={'disp': True}) # method='L-BFGS-B'

#     # Extract fitted coefficients
#     A_fit, B_fit, C_fit, D_fit = result.x

#     print("Fitted coefficients:")
#     # Save the fitted coefficients to a pickle file
#     coefficients = {'A': A_fit, 'B': B_fit, 'C': C_fit, 'D': D_fit}
#     save_coefficients_to_pickle(coefficients)

# ######################################################################################
# # Comparision - example
# # Use the co-efficients saved as coefficients.pkl to generate the density values for 10 days




#     load_coefficients = load_coefficients_from_pickle('./coefficients.pkl')
#     print("Fitted coefficients saved to pickle file.")
#     A_fit = load_coefficients['A']
#     B_fit = load_coefficients['B']
#     C_fit = load_coefficients['C']
#     D_fit = load_coefficients['D']

    
#     # Example: Altitude 400 km, argument of latitude π/4 rad, inclination 45°
#     dates, density_ref_10_days, density_analytic_10_days = compare_density_10_days(
#         lat=0, lon=0, altitude=250, argument_of_latitude=np.pi/4, inclination=np.radians(45),
#         A_fit=A_fit, B_fit=B_fit, C_fit=C_fit, D_fit=D_fit)


#     # print a table of densities
#     for i in range(len(dates)):
#         print(f"{dates[i].strftime('%Y-%m-%d')} | {density_ref_10_days[i]:.4e} | {density_analytic_10_days[i]:.4e}")



#     # Plot the reference density
#     plt.figure(figsize=(10, 6))
#     plt.plot(dates, density_ref_10_days, label='NRLMSISE-00 Reference Density', color='b')
#     plt.xlabel('Date')
#     plt.ylabel('Density (g/cm^3)')
#     plt.title('NRLMSISE-00 Reference Density Over 10 Days (Solar Peak)')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # Plot the analytic density model
#     plt.figure(figsize=(10, 6))
#     plt.plot(dates, density_analytic_10_days, label='Analytic Density Model', color='r', linestyle='--')
#     plt.xlabel('Date')
#     plt.ylabel('Density (g/cm^3)')
#     plt.title('Analytic Density Model Over 10 Days (Solar Peak)')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # Plot both together in one graph
#     plt.figure(figsize=(10, 6))
#     plt.plot(dates, density_ref_10_days, label='NRLMSISE-00 Reference Density', color='b')
#     plt.plot(dates, density_analytic_10_days, label='Analytic Density Model', color='r', linestyle='--')
#     plt.xlabel('Date')
#     plt.ylabel('Density (g/cm^3)')
#     plt.title('Density Comparison Over 10 Days (Solar Peak)')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

