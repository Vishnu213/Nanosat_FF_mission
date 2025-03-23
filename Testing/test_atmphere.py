import casadi as ca

def atmosphere(h):
    """
    Computes the atmospheric density using an exponential model (CasADi version).

    Parameters:
        h (casadi.SX or casadi.MX): Altitude from Earth's surface [km] (CasADi symbolic variable)

    Returns:
        casadi.SX or casadi.MX: Atmospheric density [kg/m^3] (CasADi symbolic expression)
    """
    # Reference altitude vector [km]
    h_0 = [0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
           180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]

    # Reference density vector [kg/m^3]
    rho_0 = [1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4,
             8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8,
             8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10, 2.798e-10, 7.248e-11,
             2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13,
             3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15]

    # Scale height vector [km]
    H = [7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382,
         5.877, 7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546,
         53.628, 53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64,
         181.05, 268.00]

    # # Use CasADi's conditional logic for efficient piecewise function definition.
    # rho = ca.SX.zeros(1)  # Initialize rho as a CasADi symbolic variable
    # for k in range(len(h_0) - 1):
    #     rho = ca.if_else(h_0[k] <= h,  #Condition
    #                     ca.if_else(h <= h_0[k + 1],rho_0[k] * ca.exp(-(h - h_0[k]) / H[k]),rho), #True
    #                      rho) #False

    # # For altitudes above the highest reference point, use the last interval's parameters.
    # rho = ca.if_else(h > h_0[-1], rho_0[-1] * ca.exp(-(h - h_0[-1]) / H[-1]), rho)

    # Initialize rho. We'll use ca.if_else to update it.
    rho = rho_0[-1] * ca.exp(-(h - h_0[-1]) / H[-1])  # Default to the highest altitude

    for k in range(len(h_0) - 1):
        # Use ca.logic_and to combine conditions for MX compatibility
        condition = ca.logic_and(h_0[k] <= h, h <= h_0[k + 1])
        rho = ca.if_else(condition, rho_0[k] * ca.exp(-(h - h_0[k]) / H[k]), rho)


    return rho
    
def atmosphere_numpy(h):
    """
    Computes the atmospheric density using an exponential model.

    Parameters:
    h (float): Altitude from Earth's surface [km]

    Returns:
    float: Atmospheric density [kg/m^3]
    """
    import numpy as np
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
    j = 27   # Default to the highest range if altitude is greater than the last value
    for k in range(27):
        if h_0[k] <= h <= h_0[k + 1]:
            j = k
            break

    # Compute density [kg/m^3] using exponential model
    rho = rho_0[j] * np.exp(-(h - h_0[j]) / H[j])

    return rho



if __name__ == '__main__':
    # Example usage with CasADi
    h = ca.SX.sym('h')  # Define a symbolic variable for altitude
    rho_sym = atmosphere(h)
    print("Symbolic density expression:\n", rho_sym)

    # Create a CasADi function to evaluate the density
    density_func = ca.Function('density', [h], [rho_sym])

    # Test with a specific altitude (e.g., 50 km)
    altitude_km = 50.0
    density_at_50km = density_func(altitude_km)
    print(f"\nDensity at {altitude_km} km (CasADi): {density_at_50km} kg/m^3")
    
    density_at_50km_np = atmosphere_numpy(altitude_km)
    print(f"Density at {altitude_km} km (numpy): {density_at_50km_np} kg/m^3")


    # Test with another altitude (e.g., 250 km)
    altitude_km = 250.0
    density_at_250km = density_func(altitude_km)
    print(f"\nDensity at {altitude_km} km (CasADi): {density_at_250km} kg/m^3")    
    density_at_250km_np = atmosphere_numpy(altitude_km)
    print(f"Density at {altitude_km} km (numpy): {density_at_250km_np} kg/m^3")

    # Example using MX for more complex expressions (if needed)
    h_mx = ca.MX.sym('h_mx')
    rho_mx = atmosphere(h_mx)
    density_func_mx = ca.Function('density_mx', [h_mx], [rho_mx])
    # You would typically use MX within larger optimization problems.

    #Test vector inputs
    altitudes = [0, 25, 50, 150, 300, 600, 900, 1000, 1100]
    for alt in altitudes:
        print(f"Density at {alt} km (CasADi): {density_func_mx(alt)} kg/m^3")
        print(f"Density at {alt} km (numpy): {atmosphere_numpy(alt)} kg/m^3")