"""
Nanosat Formation Flying Project

Testing the core libraries : Testing the absolute near non singular orbital elements dynamics

Author:
    Vishnuvardhan Shakthibala
    
"""
## Copy the following lines of code 
# FROM HERE
import numpy
from scipy import integrate
import matplotlib.pyplot as plt
import os
import sys
import math
## ADD the packages here if you think it is needed and update it in this file.

## Import our libraries here
Library= os.path.join(os.path.dirname(os.path.abspath(__file__)),"../core")
sys.path.insert(0, Library)

from TwoBP import MEANNSOE2OSCOE, car2kep, kep2car, twobp_cart, gauss_eqn, Event_COE, theta2M, guess_nonsingular, M2theta, Param2NROE, guess_nonsingular_Bmat, lagrage_J2_diff,NSROE2car

from dynamics import absolute_NSROE_dynamics_density
# Parameters that is of interest to the problem
data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
deg2rad = numpy.pi / 180

# CHECK Formation Establishment and Reconfiguration Using
# Differential Elements in J2-Perturbed Orbits and SENGUPTA
# Chaser spacecraft initial conditions
# orbital elements - non singular



# Deputy spacecraft relative orbital  elements/ LVLH initial conditions
# NOE_chief = numpy.array([a,lambda_0,i,q1,q2,omega])
NOE_chief = numpy.array([6800,0.1,90*deg2rad,0.001,0.003,270.828*deg2rad])
## MAKE SURE TO FOLLOW RIGHT orbital elements order


# Design parameters for the formation - Sengupta and Vadali 2007 Relative Motion and the Geometry of Formations in Keplerian Elliptic Orbits

rho_1 = 500 # [m]  - radial separation 
rho_2 = 200 # [m]  - along-track separation
rho_3 = 300 # [m]  - cross-track separation
alpha = 0  # [rad] - angle between the radial and along-track separation
beta = numpy.pi/2 # [rad] - angle between the radial and cross-track separation
vd = -10 # Drift per revolutions m/resolution

parameters=numpy.array([rho_1,rho_2,rho_3,alpha,beta,vd])

# Initial relative orbital elements
RNOE_0=Param2NROE(NOE_chief, parameters,data)

print("RELATIVE ORBITAL ELEMTNS INITIAL", RNOE_0)
print("CHIEF INTIIAL ORBITAL ELEMENTS", NOE_chief)


# feed it into dynamical system to get the output
yy_o=NOE_chief
# test for gauess equation
mu=data["Primary"][0]
Torb = 2*numpy.pi*numpy.sqrt(NOE_chief[0]**3/mu)    # [s]    Orbital period
n_revol_T = 24*60*60/Torb
n_revolution=10000#n_revol_T
T_total=n_revolution*Torb

t_span=[0,T_total]
teval=numpy.linspace(0, T_total, 200000)
# K=numpy.array([k1,k2])

sol=integrate.solve_ivp(absolute_NSROE_dynamics_density, t_span, yy_o,t_eval=teval,
                        method='RK23',args=(data,),rtol=1e-15, atol=1e-12)


# Convert from NROE to Carterian co-ordinates. 



rr_s=numpy.zeros((3,len(sol.y[0])))
vv_s=numpy.zeros((3,len(sol.y[0])))
NSCOE_OS=numpy.zeros((8,len(sol.y[0])))

normal_r = numpy.zeros((len(sol.y[0])))

for i in range(0,len(sol.y[0])):
    # if sol.y[5][i]>2*numpy.pi:
    #     sol.y[5][i]= 
    # if sol.y[1][i]>2000:
        # print("lambda",sol.y[1][i])
    rr_s[:,i],vv_s[:,i]=NSROE2car(numpy.array([sol.y[0][i],sol.y[1][i],sol.y[2][i],
                                               sol.y[3][i],sol.y[4][i],sol.y[5][i]]),data)
    NSCOE_OS[:,i]=MEANNSOE2OSCOE(numpy.array([sol.y[0][i],sol.y[1][i],sol.y[2][i],
                                               sol.y[3][i],sol.y[4][i],sol.y[5][i]]))
    # print("OSCOE",NSCOE_OS)
    normal_r[i]=numpy.linalg.norm(rr_s[:,i])
    # print(rr_s[:,i],vv_s[:,i])
    # print(sol.y[0][i],sol.y[1][i],sol.y[2][i], sol.y[3][i],sol.y[4][i],sol.y[5][i])
    rp = numpy.linalg.norm(rr_s[:,i])
    # print(rp,sol.y[0][i])
    if  rp< data["Primary"][1]:
        print("Satellite is inside the Earth's radius")
        exit()



    # h = COE[0]
    # e =COE[1]
    # i =COE[2]
    # OM = COE[3]
    # om =COE[4]
    # TA =COE[5]

    


# Spherical earth
# Setting up Spherical Earth to Plot
N = 50
phi = numpy.linspace(0, 2 * numpy.pi, N)
theta = numpy.linspace(0, numpy.pi, N)
theta, phi = numpy.meshgrid(theta, phi)

r_Earth = 6378.14  # Average radius of Earth [km]
X_Earth = r_Earth * numpy.cos(phi) * numpy.sin(theta)
Y_Earth = r_Earth * numpy.sin(phi) * numpy.sin(theta)
Z_Earth = r_Earth * numpy.cos(theta)

# draw the unit vectors of the ECI frame on the 3d plot of earth



# Plotting Earth and Orbit
fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)
# x-axis
ax.quiver(0, 0, 0, 1e4, 0, 0, color='r', label='X-axis')
# y-axis
ax.quiver(0, 0, 0, 0, 1e4, 0, color='g', label='Y-axis')
# z-axis
ax.quiver(0, 0, 0, 0, 0, 1e4, color='b', label='Z-axis')
# plotting
ax.plot3D(rr_s[0],rr_s[1],rr_s[2] , 'black', linewidth=2, alpha=1)

ax.set_title('two body trajectory')




fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, rr_s[0])
axs[0].set_title('x')

# Plot data on the second subplot
axs[1].plot(teval, rr_s[1])
axs[1].set_title('y')

axs[2].plot(teval, rr_s[2])
axs[2].set_title('z')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[0])
axs[0].set_title('semi major axis')
print(min(sol.y[0]),max(sol.y[0]))

# Plot data on the second subplot
axs[1].plot(teval, sol.y[1])
axs[1].set_title('mean true latitude')

axs[2].plot(teval, sol.y[2])
axs[2].set_title('inclination')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[3])
axs[0].set_title('q1')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[4])
axs[1].set_title('q2')

axs[2].plot(teval, sol.y[5])
axs[2].set_title('right ascenstion of ascending node')


fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Flatten the axes array for easy iteration
axs = axs.flatten()

for i in range(8):
    axs[i].plot(teval,NSCOE_OS[i])
    axs[i].set_title(f'NSCOE_O[{i}]')

# Plot "Normal R" data on the final
fig=plt.figure()
ax = plt.axes()
ax.plot(teval,normal_r)
ax.set_title('Normal R')
# Adjust layout for better spacing
plt.tight_layout()
plt.show()

