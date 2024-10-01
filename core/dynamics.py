import numpy

from TwoBP import (
    car2kep, 
    kep2car, 
    twobp_cart, 
    gauss_eqn, 
    Event_COE, 
    theta2M, 
    M2theta, 
    Param2NROE, 
    guess_nonsingular_Bmat, 
    lagrage_J2_diff,
    Lagrange_deri, 
    NSROE2car,
    NSROE2LVLH,
    NSROE2LVLH_2)
from lift_drag import compute_forces_for_entities, loaded_polynomials


def yaw_dynamics_N(t, yy, param):
    N_deputies = param["N_deputies"]  # Number of deputies (including chief)
    Iz = param["sat"]  # Assume that the moment of inertia is provided for each satellite

    y_dot = numpy.zeros(N_deputies + 1)  # Initialize yaw derivatives for all spacecraft
    u = numpy.zeros((N_deputies + 1, 1))  # Control input (can be updated based on control logic)

    # Loop over each spacecraft (including chief)
    for i in range(N_deputies + 1):
        y_dot[i] = -Iz[i] * u[i]  # Yaw dynamics for each spacecraft

    return y_dot

def yaw_dynamics(t, yy, param):
    Izc = param["sat"][0]
    Izd = param["sat"][1]

    y_dot = numpy.zeros((2,))
    u = numpy.zeros((2, 1))

    y_dot[0] = -Izc * u[0]
    y_dot[1] = -Izd * u[1]

    return y_dot

def Dynamics_N(t, yy, data):
    N_deputies = data["N_deputies"]  # Number of deputies
    #print("N_deputies",N_deputies)
    mu = data["Primary"][0]

    # Chief satellite dynamics
    chief_start_idx = 6 * N_deputies
    yaw_start = 6 * (N_deputies + 1) 
    chief_state = yy[chief_start_idx:chief_start_idx + 6]

    y_dot_chief,u_c= absolute_NSROE_dynamics(t, chief_state, data,yy)  # Chief dynamics

    y_dot_deputies = []



    # Loop over each deputy for relative dynamics
    for d in range(N_deputies):
        start_idx = d * 6
        delta_NSROE = yy[start_idx:start_idx + 6]  # Relative orbital elements of deputy
        
        # Calculate the absolute orbital elements by summing the deltas with the chief NSROE
        #print("yy",yy)
        deputy_NSOE = chief_state + delta_NSROE
        #print("deputy_NSOE",deputy_NSOE)
        #print("chief_state",chief_state)
        #print("delta_NSROE",delta_NSROE)
        rr_deputy, vv_deputy = NSROE2car(deputy_NSOE,data)

        rr_1 = numpy.vstack([rr_deputy])
        vv_1 = numpy.vstack([vv_deputy])   
        # Prepare data for each deputy
        satellite_key = f"deputy_{d + 1}"  # Access deputy names dynamically
        satellite_properties = data["satellites"][satellite_key]    
        data_deputy = {}
        data_deputy['Primary'] = data['Primary']
        data_deputy['S/C'] = [satellite_properties["mass"], satellite_properties["area"]]
        #print("rr_1",rr_1)
        #print("vv_1",vv_1)

        # Compute forces for the deputy

        u_deputy = compute_forces_for_entities(data_deputy, loaded_polynomials, [yy[yaw_start+d]],vv_1, rr_1)
        u_deputy = numpy.zeros((3))

        # calculate the differential aerodynamic forces
        u = u_c - u_deputy
        #print("u_c",u_c)
        # Compute the Lagrange matrix (A) and B-matrix for the deputy
        A_deputy = Lagrange_deri(t, deputy_NSOE, data)
        B_deputy = guess_nonsingular_Bmat(t, deputy_NSOE,data, numpy.array(yy[yaw_start],yy[yaw_start+d]))  # Yaw specific to each deputy
        #print("A_deputy",A_deputy.shape)
        #print("B_deputy",B_deputy.shape)
        #print("u",u.shape)
        #print("delta_NSROE",delta_NSROE.shape)
        
        y_dot_deputy = numpy.matmul(A_deputy, delta_NSROE) + numpy.matmul(B_deputy, u)
        #print(numpy.matmul(A_deputy, delta_NSROE).shape)
        #print(numpy.matmul(B_deputy, u).shape)
        #print("y_dot_deputy",y_dot_deputy.shape)
        #print("Deputy dynamics size:", y_dot_deputy.size)
        
        # Apply relative dynamics for each deputy (implement relative dynamics here)
        y_dot_deputies.append(y_dot_deputy)


    # Yaw dynamics for chief + deputies (one yaw state per spacecraft)
    yaw_start = 6 * (N_deputies + 1)  # Yaw states start after the chief orbital elements
    yaw_states = yy[yaw_start:]  # Extract yaw states (one per spacecraft: chief + deputies)

    # Calculate yaw dynamics
    yaw_dot = yaw_dynamics_N(t, yaw_states, data)  # Yaw dynamics for all spacecraft



    y_dot_deputies = numpy.array([y_dot_deputies]).flatten()
    #print("Deputy dynamics size:", y_dot_deputies.shape)

    #print("Deputy dynamics size:", y_dot_deputies.shape)
    #print("Chief dynamics size:", y_dot_chief.size)
    #print("Yaw dynamics size:", yaw_dot.size)
    # Now concatenate deputies' dynamics, chief's dynamics, and yaw dynamics
    y_dot_total = numpy.concatenate([y_dot_deputies, y_dot_chief, yaw_dot])
    #print("y_dot_total",y_dot_total.shape)
    #print("y_dot_total",y_dot_total)
    return y_dot_total

def absolute_NSROE_dynamics(t, yy, param,yy_o):
    print("inside the abs yy0",yy_o)
    print("inside the abs",yy)
    if numpy.isnan(yy).any():
        print("inside the abs",yy)
    A = lagrage_J2_diff(t, yy, param)
    B = guess_nonsingular_Bmat(t, yy, param, yy_o[12:14])
    #print("B",B)
    #print("A",A)
    #print("inside the abs",yy)
    # convert the NSROE to ECI frame to get the aerodynamic forces
    rr, vv = NSROE2car(yy,param)
    data = {}
    data['Primary'] = param['Primary']
    data['S/C'] = [param["satellites"]["chief"]["mass"], param["satellites"]["chief"]["area"]]
    #print(yy_o[13])
    #print(rr)
    rr_1 = numpy.vstack([rr])
    vv_1 = numpy.vstack([vv])   
    u_chief=compute_forces_for_entities(data, loaded_polynomials,yy_o[12:13], vv_1, rr_1)
    u_chief_scale = u_chief * 1e12
    #u_chief = numpy.zeros((3))
    y_dot = A + numpy.matmul(B, u_chief_scale/1e12)

    return y_dot, u_chief

def absolute_NSROE_dynamics_N(t, yy, param,yy_o):
    A = lagrage_J2_diff(t, yy, param)
    B = guess_nonsingular_Bmat(t, yy, param, yy_o[12:14])
    #print("B",B)
    #print("A",A)
    #print("inside the abs",yy)
    # convert the NSROE to ECI frame to get the aerodynamic forces
    rr, vv = NSROE2car(yy[0:6],param)
    data = {}
    data['Primary'] = param['Primary']
    data['S/C'] = [param["satellites"]["chief"]["mass"], param["satellites"]["chief"]["area"]]
    #print(yy_o[13])
    #print(rr)
    rr_1 = numpy.vstack([rr])
    vv_1 = numpy.vstack([vv])   
    u_chief=compute_forces_for_entities(data, loaded_polynomials,yy_o[18:19], vv_1, rr_1)
    u_chief = numpy.zeros((3))
    y_dot = A + numpy.matmul(B, u_chief)

    return y_dot, u_chief

def Dynamics(t, yy, param):
    start_idx = 0
    chief_state = yy[6:12]
    y_dot_chief, u_c = absolute_NSROE_dynamics(t, chief_state, param,yy)

    delta_NSROE = yy[start_idx:start_idx + 6]  # Relative orbital elements of deputy
    
    # Calculate the absolute orbital elements by summing the deltas with the chief NSROE
    #print("yy",yy)
    deputy_NSOE = chief_state + delta_NSROE
    #print("deputy_NSOE",deputy_NSOE)
    #print("chief_state",chief_state)
    #print("delta_NSROE",delta_NSROE)
    rr_deputy, vv_deputy = NSROE2car(deputy_NSOE,param)

    rr_1 = numpy.vstack([rr_deputy])
    vv_1 = numpy.vstack([vv_deputy])   
    # Prepare data for each deputy
    satellite_key = f"deputy_1"  # Access deputy names dynamically
    satellite_properties = param["satellites"][satellite_key]    
    data_deputy = {}
    data_deputy['Primary'] = param['Primary']
    data_deputy['S/C'] = [satellite_properties["mass"], satellite_properties["area"]]
    #print("rr_1",rr_1)
    #print("vv_1",vv_1)

    # Compute forces for the deputy

    u_deputy = compute_forces_for_entities(data_deputy, loaded_polynomials, [yy[13]],vv_1, rr_1)
    #u_deputy = numpy.zeros((3))

    # calculate the differential aerodynamic forces
    u =  u_deputy*1e12 - u_c
    #print("u_c",u_c)
    # Compute the Lagrange matrix (A) and B-matrix for the deputy
    A_deputy = Lagrange_deri(t, chief_state, param)
    B_deputy = guess_nonsingular_Bmat(t, chief_state, param,yy[12:14])  # Yaw specific to each deputy
    #print("A_deputy",A_deputy.shape)
    #print("B_deputy",B_deputy.shape)
    #print("u",u.shape)
    #print("delta_NSROE",delta_NSROE.shape)
    
    y_dot_deputy = numpy.matmul(A_deputy, delta_NSROE) + numpy.matmul(B_deputy, u/1e12)

    
    y_dot_yaw = yaw_dynamics(t, yy[12:14], param)
    

    y = numpy.concatenate((y_dot_deputy, y_dot_chief, y_dot_yaw))

    return y
