import numpy as np
from scipy import integrate

def RK5_step(f, t, y, h, *args):
    """
    Runge-Kutta 5th-order method for one step of integration.
    Args:
    - f: The function that defines the system dynamics f(t, y)
    - t: Current time
    - y: Current state vector
    - h: Time step
    - args: Additional arguments to be passed to f
    """
    k1 = h * f(t, y, *args)
    k2 = h * f(t + h/4, y + k1/4, *args)
    k3 = h * f(t + 3*h/8, y + 3*k1/32 + 9*k2/32, *args)
    k4 = h * f(t + 12/13*h, y + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3, *args)
    k5 = h * f(t + h, y + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4, *args)
    k6 = h * f(t + h/2, y - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5, *args)
    
    # Update y
    y_next = y + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6
    return y_next

def RK5_integrator(f, t_span, y0, h, args=()):
    """
    Custom RK5 integrator.
    Args:
    - f: The function defining the system dynamics
    - t_span: Tuple (t0, tf) defining the time span
    - y0: Initial state vector
    - h: Time step size
    - args: Additional arguments to pass to f
    """
    t0, tf = t_span
    t_values = [t0]
    y_values = [y0]
    
    t = t0
    y = y0
    
    while t < tf:
        y = RK5_step(f, t, y, h, *args)
        t += h
        
        t_values.append(t)
        y_values.append(y)
    
    return np.array(t_values), np.array(y_values).T


def RK4_step(f, t, y, h, *args):
    """
    Runge-Kutta 4th-order method for one step of integration.
    Args:
    - f: The function that defines the system dynamics f(t, y)
    - t: Current time
    - y: Current state vector
    - h: Time step
    - args: Additional arguments to be passed to f
    """
    k1 = h * f(t, y, *args)
    k2 = h * f(t + h/2, y + k1/2, *args)
    k3 = h * f(t + h/2, y + k2/2, *args)
    k4 = h * f(t + h, y + k3, *args)
    
    # Update y
    y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_next

def RK4_integrator(f, t_span, y0, h, args=()):
    """
    Custom RK4 integrator.
    Args:
    - f: The function defining the system dynamics
    - t_span: Tuple (t0, tf) defining the time span
    - y0: Initial state vector
    - h: Time step size
    - args: Additional arguments to pass to f
    """
    t0=t_span[0]
    tf=t_span[-1]    
    t_values = [t0]
    y_values = [y0]
    
    t = t0
    y = y0
    
    while t < tf:
        y = RK4_step(f, t, y, h, *args)
        t += h
        
        t_values.append(t)
        y_values.append(y)
    
    return np.array(t_values), np.array(y_values).T


def integrate_system(method, Dynamics, t_span, yy_o, h, data, uu):
    """
    Integrate the system dynamics using the specified method.
    Args:
    - method: 'RK4', 'RK5', or 'solve_ivp'
    - Dynamics: The system dynamics function
    - t_span: The time span for integration
    - yy_o: Initial state
    - h: Time step
    - data: Parameters for the dynamics
    - uu: Control inputs
    """
    if method == 'solve_ivp':
        sol = integrate.solve_ivp(Dynamics, t_span, yy_o, t_eval=t_span,
                                  method='DOP853', args=(data, uu), rtol=1e-10, atol=1e-12)
        t_values = sol.t
        y_values = sol.y
    elif method == 'RK4':
        
        t_values, y_values = RK4_integrator(Dynamics, t_span, yy_o, h, args=(data, uu))
    elif method == 'RK5':
        t_values, y_values = RK5_integrator(Dynamics, t_span, yy_o, h, args=(data, uu))
    else:
        raise ValueError("Unknown integration method: choose 'solve_ivp', 'RK4', or 'RK5'.")
    
    return t_values, y_values