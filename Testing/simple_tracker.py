# Re-importing CasADi to ensure all necessary functions are available
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# Define the moment of inertia, proportional and derivative gains, and sine wave properties again
Izc = 1.2        # Moment of inertia
K_p = 1     # Proportional gain
K_d = 0.1         # Derivative gain

A = 1.0           # Amplitude of the sine wave
T_period = 2000.0  # Period of the sine wave

# Time variable
t = ca.MX.sym('t')  # Time

# Define differential variables
y = ca.MX.sym('y')     # Yaw angle (state)
yd = ca.MX.sym('yd')   # Yaw rate (derivative of yaw angle)
u = ca.MX.sym('u')     # Control input (algebraic variable)

# Define the desired yaw angle as a sine wave
theta_desired = A * ca.sin(2 * ca.pi / T_period * t)

# PD control law (algebraic equation)
control_eqn = u - K_p * (theta_desired - y) 
# Differential equation (yaw dynamics)
dyn_eqn = ca.vertcat(yd, -Izc * u)  # Now we include both y_dot = yd and yd_dot = -Izc*u

# Pack the differential and algebraic equations
dae = {
    'x': ca.vertcat(y, yd),  # Differential states: y and yd
    'p': ca.vertcat(t),      # Time parameter
    'z': ca.vertcat(u),      # Algebraic state (control input)
    'ode': dyn_eqn,          # Differential equations for y and yd
    'alg': control_eqn       # Algebraic equation (PD control law)
}


# Time variable
t = ca.MX.sym('t')  # Time

# Define differential variables
y = ca.MX.sym('y')     # Yaw angle (state)
yd = ca.MX.sym('yd')   # Yaw rate (derivative of yaw angle)
u = ca.MX.sym('u')     # Control input (algebraic variable)

# Define the desired yaw angle as a sine wave
theta_desired = A * ca.sin(2 * ca.pi / T_period * t)

# PD control law (algebraic equation)
control_eqn = u - K_p * (theta_desired - y)

# Differential equation (yaw dynamics)
dyn_eqn = ca.vertcat(yd, Izc * u)  # Now we include both y_dot = yd and yd_dot = -Izc*u

# Pack the differential and algebraic equations
dae = {
    'x': ca.vertcat(y, yd),  # Differential states: y and yd
    'p': ca.vertcat(t),      # Time parameter
    'z': ca.vertcat(u),      # Algebraic state (control input)
    'ode': dyn_eqn,          # Differential equations for y and yd
    'alg': control_eqn       # Algebraic equation (PD control law)
}

# Set up time points for simulation
tf = 1000  # Final time
dt = 2   # Time step
t_values = np.arange(0, tf + dt, dt)
# Define integration options
integrator_options = {
    'tf': tf,              # Final time of the integration
    'grid': t_values,  # Time grid with 100 intervals
    'reltol': 1e-8,               # Relative tolerance
    'abstol': 1e-10,               # Absolute tolerance
    'output_t0': True           # Include initial time in the output
}
integrator = ca.integrator('integrator', 'idas', dae,integrator_options)

# Redefine initial conditions for integration
y0 = [0.02, 0.0]  # Initial yaw angle and yaw rate
u0 = 0.02         # Initial control input
t0= 0.0           # Initial time
# Call the integrator once with the full time grid
res = integrator(x0=y0, z0=u0, p=t0)

# Extract results for yaw angle, control input, and desired yaw angle
y_values = res['xf'].full()[:, 0]  # Yaw angle
yd_values = res['xf'].full()[:, 1] # Yaw rate (optional, not plotted)
u_values = res['zf'].full().T       # Control input

y_values = res['xf'].full()[0, :]  # Yaw angle

# Desired yaw angle over time (sine wave)
theta_desired_values = A * np.sin(2 * np.pi / T_period * t_values)


# Check that all arrays have the same length
print(f"Shapes -> t_values: {t_values.shape}, y_values: {y_values.shape}, u_values: {u_values.shape}, theta_desired_values: {theta_desired_values.shape}")

# Plotting the results

plt.figure(figsize=(10, 6))

# Plot yaw angle (y) and desired yaw angle (theta_desired)
plt.subplot(2, 1, 1)
plt.plot(t_values, y_values, label='Yaw Angle (y)', color='b')
plt.plot(t_values, theta_desired_values, label='Desired Yaw Angle (theta_desired)', color='g', linestyle='--')
plt.title('Yaw Angle Tracking')
plt.ylabel('Angle (rad)')
plt.legend()

# Plot control input (u)
plt.subplot(2, 1, 2)
plt.plot(t_values, u_values, label='Control Input (u)', color='r')
plt.title('Control Input Over Time')
plt.ylabel('Control Input (u)')
plt.xlabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.show()