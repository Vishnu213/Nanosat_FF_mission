import numpy
from scipy import integrate,optimize
import matplotlib.pyplot as plt
import os
import sys
import math

from TwoBP import car2kep, kep2car, twobp_cart, gauss_eqn, Event_COE, theta2M, guess_nonsingular, M2theta, Param2NROE, guess_nonsingular_Bmat, lagrage_J2_diff, absolute_NSROE_dynamics ,NSROE2car

def car2NNSOE(r, v, mu):
    """
    Convert from Cartesian coordinates (r, v) directly to non-singular elements.
    
    Parameters:
    r (numpy.ndarray): Position vector [x, y, z] (km)
    v (numpy.ndarray): Velocity vector [vx, vy, vz] (km/s)
    mu (float): Standard gravitational parameter (km^3/s^2)
    
    Returns:
    Non-singular elements [a, l, i, q1, q2, Omega]
    """
    
    # Step 1: Get classical orbital elements (COEs) from position and velocity
    COE_temp= car2kep(r, v, mu)
    

    a = COE_temp[0]**2 / (mu * (1 - COE_temp[1]**2))
    COE = numpy.array([a, COE_temp[1], COE_temp[2], COE_temp[3], COE_temp[4], COE_temp[5]])

    a = COE[0]
    M=theta2M(COE[5],COE[1])
    l = M+COE[4]
    i = COE[2]
    q1 = COE[1]*numpy.sin(COE[4])
    q2 = COE[1]*numpy.cos(COE[4])
    OM = COE[3]
    # Return non-singular elements
    return numpy.array([a, l, i, q1, q2, OM])

def car2NNSOE_density(r, v, mu):
    """
    Convert from Cartesian coordinates (r, v) directly to non-singular elements.
    
    Parameters:
    r (numpy.ndarray): Position vector [x, y, z] (km)
    v (numpy.ndarray): Velocity vector [vx, vy, vz] (km/s)
    mu (float): Standard gravitational parameter (km^3/s^2)
    
    Returns:
    Non-singular elements [a, l, i, q1, q2, Omega]
    """
    
    # Step 1: Get classical orbital elements (COEs) from position and velocity
    _,COE_temp= car2kep(r, v, mu)
    

    a = (COE_temp[0]**2 )/ (mu * (1 - COE_temp[1]**2))
    COE = numpy.array([a, COE_temp[1], COE_temp[2], COE_temp[3], COE_temp[4], COE_temp[5]])

    a = COE[0]
    M=theta2M(COE[5],COE[1])
    l = M+COE[4]
    u = COE[5] + COE[4]
    i = COE[2]
    q1 = COE[1]*numpy.sin(COE[4])
    q2 = COE[1]*numpy.cos(COE[4])
    OM = COE[3]
    # Return non-singular elements
    return numpy.array([a, l, i, u])