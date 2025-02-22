import numpy

def LVLHframe(rr, vv):
    """
    Calculates the rotation matrix from the Earth-Centered Inertial (ECI) frame
    to the Local Vertical Local Horizontal (LVLH) frame.

    Args:
        rr (numpy.ndarray): Position vector in the ECI frame (3-element array).
        vv (numpy.ndarray): Velocity vector in the ECI frame (3-element array).

    Returns:
        numpy.ndarray: 3x3 rotation matrix from ECI to LVLH.  The columns of this
                      matrix are the unit vectors of the LVLH frame expressed in
                      the ECI frame.
    """
    # Calculate the unit vector in the radial (x) direction.
    x_unit = rr / numpy.linalg.norm(rr)

    # Calculate the unit vector in the out-of-plane (z) direction (normal to the orbital plane).
    z_unit = numpy.cross(rr, vv) / numpy.linalg.norm(numpy.cross(rr, vv))

    # Calculate the unit vector in the along-track (y) direction.
    y_unit = numpy.cross(z_unit, x_unit)

    # Create the rotation matrix by stacking the unit vectors as columns.
    Rot_LVLH = numpy.column_stack([x_unit, y_unit, z_unit])
    return Rot_LVLH

def Frenetframe(rr, vv):
    """
    Calculates the rotation matrix from the ECI frame to the Frenet-Serret frame.

    Args:
        rr (numpy.ndarray): Position vector in the ECI frame (3-element array).
        vv (numpy.ndarray): Velocity vector in the ECI frame (3-element array).

    Returns:
        numpy.ndarray: 3x3 rotation matrix from ECI to the Frenet-Serret frame.
    """
    # Calculate the unit tangent vector (T).
    T_unit = vv / numpy.linalg.norm(vv)

    # Calculate the unit vector in the out-of-plane (W) direction (same as LVLH's z-axis).
    W_unit = numpy.cross(rr, vv) / numpy.linalg.norm(numpy.cross(rr, vv))

    # Calculate the unit normal vector (N) - the principal normal.
    N_unit = numpy.cross(T_unit, W_unit)

    # Create the rotation matrix.
    Rot_FrenetFrame = numpy.column_stack([T_unit, N_unit, W_unit])
    return Rot_FrenetFrame

def Frenet2LVLH(rr, vv):
    """
    Calculates the rotation matrix from the Frenet-Serret frame to the LVLH frame.

    Args:
        rr (numpy.ndarray): Position vector in the ECI frame (3-element array).
        vv (numpy.ndarray): Velocity vector in the ECI frame (3-element array).

    Returns:
        numpy.ndarray: 3x3 rotation matrix from Frenet-Serret to LVLH.
    """
    RR_Frenet = Frenetframe(rr, vv)
    RR_LVLH = LVLHframe(rr, vv)
    # The rotation from Frenet to LVLH is the product of the inverse (transpose
    # for rotation matrices) of the LVLH transformation and the Frenet transformation.
    Rot_F2LVLH = numpy.matmul(RR_LVLH.transpose(), RR_Frenet)
    return Rot_F2LVLH

# Example Usage (and testing)
if __name__ == '__main__':
    # Example ECI position and velocity vectors (you can replace with your own data).
    r_eci = numpy.array([7000000.0, 0.0, 0.0])  # Example: 7000 km in the x-direction
    v_eci = numpy.array([0.0, 7500.0, 0.0])  # Example: 7.5 km/s in the y-direction

    # LVLH Frame
    R_eci_to_lvlh = LVLHframe(r_eci, v_eci)
    print("ECI to LVLH Rotation Matrix:\n", R_eci_to_lvlh)

    # Frenet Frame
    R_eci_to_frenet = Frenetframe(r_eci, v_eci)
    print("\nECI to Frenet Rotation Matrix:\n", R_eci_to_frenet)

    # Frenet to LVLH
    R_frenet_to_lvlh = Frenet2LVLH(r_eci, v_eci)
    print("\nFrenet to LVLH Rotation Matrix:\n", R_frenet_to_lvlh)


    # Verification:  A rotation from ECI->Frenet and then Frenet->LVLH should be the same as ECI->LVLH
    #  R_eci_to_lvlh  ?=  R_frenet_to_lvlh @ R_eci_to_frenet
    test_result = numpy.allclose(R_eci_to_lvlh, R_frenet_to_lvlh @ R_eci_to_frenet)
    print("\nVerification (ECI->LVLH == Frenet->LVLH @ ECI->Frenet):", test_result)

    # Test case 2:  A circular orbit in the x-y plane.
    r_eci2 = numpy.array([0.0, 7000000.0, 0.0])
    v_eci2 = numpy.array([-7500.0, 0.0, 0.0])

    R_eci_to_lvlh2 = LVLHframe(r_eci2, v_eci2)
    print("\nECI to LVLH Rotation Matrix (Test Case 2):\n", R_eci_to_lvlh2)
    R_frenet_to_lvlh2 = Frenet2LVLH(r_eci2,v_eci2)
    print("\nFrenet to LVLH Rotation Matrix (Test Case 2):\n", R_frenet_to_lvlh2)


    # Test case 3: An orbit with some inclination.
    r_eci3 = numpy.array([5000000.0, 5000000.0, 2000000.0])
    v_eci3 = numpy.array([-5000.0, 5500.0, 1000.0])
    R_eci_to_lvlh3 = LVLHframe(r_eci3, v_eci3)
    print("\nECI to LVLH Rotation Matrix (Test Case 3):\n", R_eci_to_lvlh3)
    R_frenet_to_lvlh3 = Frenet2LVLH(r_eci3,v_eci3)
    print("\nFrenet to LVLH Rotation Matrix (Test Case 3):\n", R_frenet_to_lvlh3)


    # Test case 4:  Check for near-zero vectors (to test robustness)
    r_eci4 = numpy.array([1e-10, 1e-10, 1e-10])  # Almost zero position
    v_eci4 = numpy.array([1e-5, 1e-5, 1e-5])  # Almost zero velocity
    R_eci_to_lvlh4 = LVLHframe(r_eci4, v_eci4)  # Expect a warning, but it should still run
    print("\nECI to LVLH Rotation Matrix (Test Case 4 - near zero):\n", R_eci_to_lvlh4)
    #Ideally a check like:
    if numpy.linalg.norm(r_eci4) < 1e-6:  # Or some appropriate small threshold
        print("Warning: Position vector is very small.  LVLH frame may be ill-defined.")
    #And similarly for the velocity vector
    R_frenet_to_lvlh4 = Frenet2LVLH(r_eci4,v_eci4)
    print("\nFrenet to LVLH Rotation Matrix (Test Case 4):\n", R_frenet_to_lvlh4)