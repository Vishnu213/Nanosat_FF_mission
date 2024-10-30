import numpy
import numpy as np
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

uu_ind = []
uu_log =[]


def yaw_dynamics_N(t, yy, param):
    N_deputies = param["N_deputies"]  # Number of deputies (including chief)
    Iz = param["sat"]  # Assume that the moment of inertia is provided for each satellite

    y_dot = numpy.zeros(N_deputies + 1)  # Initialize yaw derivatives for all spacecraft
    u = numpy.zeros((N_deputies + 1, 1))  # Control input (can be updated based on control logic)

    # Loop over each spacecraft (including chief)
    for i in range(N_deputies + 1):
        y_dot[i] = -Iz[i] * u[i]  # Yaw dynamics for each spacecraft

    return y_dot

# def yaw_dynamics(t, yy, param, uu):
#     Izc = param["sat"][0]  # Moment of inertia for the chief satellite
#     Izd = param["sat"][1]  # Moment of inertia for the deputy satellite

#     # Initialize y_dot and y_ddot for both chief and deputy satellites
#     # y_dot: yaw rate (angular velocity) -> yy[14], yy[15]
#     # y_ddot: yaw acceleration
#     y_dynamics = numpy.zeros((4,))

#     # Extract angular velocities (yaw rates)
#     y_dot_c = yy[14]  # Angular velocity for chief
#     y_dot_d = yy[15]  # Angular velocity for deputy

#     T = param["T_period"]
#     U_c=23.6e-6 * np.sin(2 * np.pi * t / T)
#     u_c = U_c
#     print("TIME--------- ",t)
#     print("u_c",u_c)

#     # Yaw dynamics for chief satellite
#     y_dynamics[0] = y_dot_c  # Derivative of yaw angle = angular velocity for chief
#     y_dynamics[1] = y_dot_d  # Derivative of yaw angle = angular velocity for deputy
#     y_dynamics[2] = (Izc) * u_c #uu[0]  # Derivative of yaw rate (angular acceleration) for chief
#     y_dynamics[3] = (Izd) * uu[1]  # Derivative of yaw rate (angular acceleration) for deputy


#     # Yaw dynamics for chief satellite
#     # # y_dynamics[0] = y_dot_c  # Derivative of yaw angle = angular velocity for chief
#     # # y_dynamics[1] = y_dot_d  # Derivative of yaw angle = angular velocity for deputy
#     # y_dynamics[0] = (-Izc) * uu[0]  # Derivative of yaw rate (angular acceleration) for chief
#     # y_dynamics[1] = (-Izd) * uu[1]  # Derivative of yaw rate (angular acceleration) for deputy


#     return y_dynamics

# Define custom wave function
def custom_wave(t, period, high_value, low_value, transition_fraction, total_orbits):
    # Adjust the period to span over the desired number of orbits
    total_period = total_orbits * period

    # Normalize the time to the total period (to span over multiple orbits)
    t_mod = np.mod(t, total_period)

    # Determine the time period for the first and second halves (first half high, second half low)
    first_half_period = (total_orbits / 2) * period
    second_half_period = total_period - first_half_period
    transition_time = transition_fraction * period

    # Define different phases of the wave
    stay_high = np.where(t_mod <= first_half_period - 0.5 * transition_time, high_value, 0)
    stay_low = np.where(t_mod >= first_half_period + 0.5 * transition_time, low_value, 0)

    # Smooth transition from high to low
    transition_down = np.where(
        np.logical_and(t_mod > first_half_period - 0.5 * transition_time, t_mod <= first_half_period),
        high_value * 0.5 * (1 + np.cos(np.pi * (t_mod - (first_half_period - 0.5 * transition_time)) / transition_time)),
        0
    )

    # Smooth transition from low to high
    transition_up = np.where(
        np.logical_and(t_mod > total_period - 0.5 * transition_time, t_mod <= total_period),
        high_value * 0.5 * (1 - np.cos(np.pi * (t_mod - (total_period - 0.5 * transition_time)) / transition_time)),
        0
    )

    # Combine all phases
    wave = stay_high + stay_low + transition_down + transition_up
    return wave



def yaw_dynamics(t, yy, param, uu):
    # Extracting parameters for inertia
    Izc = param["sat"][0]  # Chief satellite's moment of inertia
    Izd = param["sat"][1]  # Deputy satellite's moment of inertia

    # Initialize state derivatives and control input
    y_dynamics = np.zeros(4)

    # Normalize yaw angles (chief and deputy) to stay within 0 and 2*pi
    if yy[12] > 2*np.pi:
        yy[12] = 2*np.pi - yy[12]
    elif yy[12] < -2*np.pi:
        yy[12] = 2 * np.pi + yy[12]

    if yy[13] > 2*np.pi:
        yy[13] = 2*np.pi - yy[13]
    elif yy[13] < -2*np.pi:
        yy[13] = 2 * np.pi + yy[13]
    


    # Extract angular velocities (yaw rates) from yy (assuming yy[14] and yy[15] are angular velocities)
    y_dot_c = yy[14]  # Chief satellite angular velocity
    y_dot_d = yy[15]  # Deputy satellite angular velocity

    T = param["T_period"]
    
    # Control gains
    Kp = 100
    Kd = 20
    
    # Control limits (min and max torque values)
    control_min = -23e-6
    control_max = 23e-6

    # Custom wave applied to control yaw angle (90 degrees to 0 degrees with smooth transitions)
    wave_output = custom_wave(t, T, 0*90 * np.pi / 180, 0, 0.5,10)
    wave_output_1 = custom_wave(t, T, 0*90 * np.pi / 180, 0, 0.5,10)
    
    # PID control law for the chief satellite
    e_current = wave_output - yy[12]  # Current error for chief
    derivative = -y_dot_c
    control_input = Kp * e_current + Kd * derivative

    # Clip the control input for chief to the specified range
    control_input_clipped = np.clip(control_input, control_min, control_max)

    # Chief satellite yaw dynamics
    y_dynamics[0] = y_dot_c  # Derivative of yaw angle (yaw rate) for chief
    y_dynamics[2] = Izc * control_input_clipped  # Derivative of yaw rate (angular acceleration) for chief

    # PID control law for the deputy satellite
    e_current_1 = wave_output_1 - yy[13]  # Current error for deputy
    derivative_1 = -y_dot_d
    control_input_1 = Kp * e_current_1 + Kd * derivative_1

    # Clip the control input for deputy to the specified range
    control_input_1_clipped = np.clip(control_input_1, control_min, control_max)

    # Deputy satellite yaw dynamics
    y_dynamics[1] = y_dot_d  # Derivative of yaw angle (yaw rate) for deputy
    y_dynamics[3] = Izd * control_input_1_clipped  # Derivative of yaw rate (angular acceleration) for deputy



    # Uncomment to print control inputs and time for debugging
    # print("u_c:", control_input_clipped)
    # print("u_d:", control_input_1_clipped)
    # print("time:", t)

    return y_dynamics




def Dynamics_N(t, yy, data):
    N_deputies = data["N_deputies"]  # Number of deputies
    #print("N_deputies",N_deputies)
    mu = data["Primary"][0]

    # Chief satellite dynamics
    chief_start_idx = 6 * N_deputies
    yaw_start = 6 * (N_deputies + 1) 
    chief_state = yy[chief_start_idx:chief_start_idx + 6]

    y_dot_chief,u_c= absolute_NSROE_dynamics_N(t, chief_state, data,yy)  # Chief dynamics

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
        # u_deputy = numpy.zeros((3))

        # calculate the differential aerodynamic forces
        u = u_deputy - u_c
        #print("u_c",u_c)
        # Compute the Lagrange matrix (A) and B-matrix for the deputy
        A_deputy = Lagrange_deri(t, chief_state, data)
        B_deputy = guess_nonsingular_Bmat(t, chief_state,data, numpy.array(yy[yaw_start])) # ,yy[yaw_start+d] # Yaw specific to each deputy
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
    # print("inside the abs yy0",yy_o)
    # if numpy.isnan(yy).any():
    #     print("inside the abs",yy)
    A = lagrage_J2_diff(t, yy, param)
    B = guess_nonsingular_Bmat(t, yy, param,yy_o[12:14]) # , yy_o[12:14]
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
    # print("rr_1 cheiffff",rr_1)
    # print("vv_1 chieffff",vv_1)   
    u_chief=compute_forces_for_entities(data, loaded_polynomials,yy_o[12:13], vv_1, rr_1)
    uu_ind.append(u_chief)
    # print("u_ind",uu_ind)
    # print("u_chief-----",u_chief)
 
    # u_chief = numpy.zeros((3))
    # u_chief = 0*np.array([1e-6,1e-6,0.01e-6])
    y_dot = A + numpy.matmul(B, u_chief)

    return y_dot, u_chief

def absolute_NSROE_dynamics_N(t, yy, param,yy_o):
    global uu_ind, uu_log
    A = lagrage_J2_diff(t, yy, param)
    B = guess_nonsingular_Bmat(t, yy, param) # yy_o[12:14]
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
    # u_chief = numpy.zeros((3))
    y_dot = A + numpy.matmul(B, u_chief)

    return y_dot, u_chief



def Dynamics(t, yy, param,uu):
    global uu_ind, uu_log
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
    # print("rr_1",rr_1)
    # print("vv_1",vv_1)

    # Compute forces for the dep
    u_deputy = compute_forces_for_entities(data_deputy, loaded_polynomials, [yy[13]],vv_1, rr_1)
    # print("u_deputy",u_deputy)
    # print("u_c",uu_ind)
    uu_ind.append(u_deputy)
    uu_log.append(uu_ind)
    uu_ind = []
    # u_deputy = numpy.zeros((3))
    # calculate the differential aerodynamic forces
    u =  u_deputy - u_c
    # print("u _differential",u)
    #print("u_c",u_c)
    # Compute the Lagrange matrix (A) and B-matrix for the deputy
    A_deputy = Lagrange_deri(t, chief_state, param)
    B_deputy = guess_nonsingular_Bmat(t, chief_state, param,yy[12:14]) # yy[12:14] # Yaw specific to each deputy
    #print("A_deputy",A_deputy.shape)h
    #print("B_deputy",B_deputy.shape)
    #print("u",u.shape)
    #print("delta_NSROE",delta_NSROE.shape)
    # u = np.array([1e-6,1e-6,0.01e-6])
    y_dot_deputy = numpy.matmul(A_deputy, delta_NSROE) + numpy.matmul(B_deputy, u)

    
    y_dot_yaw = yaw_dynamics(t, yy, param,uu)
    

    y = numpy.concatenate((y_dot_deputy, y_dot_chief, y_dot_yaw))


    return y


def absolute_NSROE_dynamics_density(t, yy, param):
    # print("inside the abs yy0",yy_o)
    # if numpy.isnan(yy).any():
    #     print("inside the abs",yy)
    A = lagrage_J2_diff(t, yy, param)
    # print("A",A)
    B = guess_nonsingular_Bmat(t, yy, param, numpy.zeros((2,1)))
    # print("B",B)
    #print("B",B)
    #print("A",A)
    #print("inside the abs",yy)
    # convert the NSROE to ECI frame to get the aerodynamic forces

    u_chief = numpy.zeros((3))
    # print("product",numpy.matmul(B, u_chief))
    y_dot = A + numpy.matmul(B, u_chief)

    return y_dot