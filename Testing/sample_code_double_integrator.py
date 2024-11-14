import numpy as np
import matplotlib.pyplot as plt

# Define parameters
T = 50000  # Total time
T_half = T / 2
K = 1/0.0412   # System property, affecting the dynamics: x_ddot = K * u
Kp = -0.000005 # Proportional gain
Ki = -0  # Integral gain
Kd = np.sqrt(4*(-Kp)/K)  # Derivative gain
print((Kd**2))
print(4*Kp)

# Define desired signal
def desired_signal(t):
    return 3 if t <= T_half else 6

# Define the double integrator dynamics with PID control
def double_integrator_pid(t, y, integral_error):
    # Current state
    position, velocity = y

    # Desired position
    x_d = desired_signal(t)
    
    # Calculate error and integral of the error
    error = x_d - position
    integral_error += error  # Update integral of error
    derivative_error = velocity  # Derivative of position is the velocity



    
    # PID control law (no adjustment for K in the control law itself)
    u = - Kp * error  - Kd * derivative_error - Ki * integral_error

    u_saturated = np.clip(u, -23e-6, 23e-6)  # Saturate the control input    

    # Dynamics of double integrator with property K
    d_position = velocity
    d_velocity = K*u_saturated  # Incorporate K directly in the dynamics

    return [d_position, d_velocity], integral_error

# Set up time array for integration
time_span = (0, T)
time_eval = np.linspace(0, T, 1000)

# Initialize integral error and history lists
integral_error = 0
position_history = []
velocity_history = []
error_history = []
time_history = []

# Initial conditions
y0 = [0, 0.]  # Initial position and velocity

# Perform numerical integration
for t in time_eval:
    # Solve for small step
    sol, integral_error = double_integrator_pid(t, y0, integral_error)
    
    # Update initial state for the next step
    y0[0] += sol[0] * (time_eval[1] - time_eval[0])  # Position update
    y0[1] += sol[1] * (time_eval[1] - time_eval[0])  # Velocity update
    
    # Log histories
    position_history.append(y0[0])
    velocity_history.append(y0[1])
    error_history.append(desired_signal(t) - y0[0])
    time_history.append(t)

# Plot results
plt.figure(figsize=(12, 8))

# Plot the position tracking
plt.subplot(3, 1, 1)
plt.plot(time_eval, [desired_signal(t) for t in time_eval], 'r--', label="Desired Position")
plt.plot(time_history, position_history, 'b', label="Tracked Position")
plt.title("Position Tracking with PID Control")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()

# Plot the velocity
plt.subplot(3, 1, 2)
plt.plot(time_history, velocity_history, 'g', label="Velocity")
plt.title("Velocity")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend()

# Plot the error
plt.subplot(3, 1, 3)
plt.plot(time_history, error_history, 'm', label="Tracking Error")
plt.title("Tracking Error")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()

plt.tight_layout()
plt.show()
