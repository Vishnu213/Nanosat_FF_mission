# Nanosat_FF_mission

This repository holds the code for the project aimed at developing guidance and control for nanosatellite formation flying mission aimed at space debris characterizating using aerodynamic force. In the following sections you will more details of the project starting from description, methodology, simulation results.

# Description of the project

Low earth orbit (LEO) and very low earth orbit(VLEO)  are important regions for space-based applications like Earth observation, Telecommunication, and Astrophysics. Understanding and modeling the Space debris in this region is of utmost importance for space situational awareness. Specifically, due to the limitations of ground-based sensor systems, the detection and characterization of sub-millimeter-level objects require an in-situ sensor system. In these regions, the effect of aerodynamic force is significant, as a result, utilization of this force for position control is very attractive, especially for nanosats. This will enable nanosats not to carry propulsion system leading to increased available size, weight, and power (SWAP) for mission outputs such as science data by increasing the payload quality and mission redundancy via extra sensors and actuators. 	
 
In this work, we are proposing a guidance and control methodology for a novel aerodynamics-based nanosat formation flying mission. As the size and shape of the debris are not known a priori, the multi-static synthetic aperture radar (MSSAR) technique is considered to detect and characterize the sub-millimeter debris particles. 


# Methodology

### Relative orbital dynamics with J2 pertubation in nearly non singular orbital element representation

Due to simplicity and its capabilities to represent orbits with arbitrary eccentricity, nearly non singular elements are used[Roscoe]. Specifically, we are making use of mean nearly non singular orbital elements - this choice is supported by the fact that these elements only take into account the secular effects and therefore nullifying these effects will be an important part of keeping the formation configuration bounded.

Furthermore, with the introduction of parameters that can be used to geometrically define the formation flying [sengupta], formation flying configurations can be obtained. 

## Design Parameters for Formation Flying

Based on **Sengupta and Vadali (2007)** - _"Relative Motion and the Geometry of Formations in Keplerian Elliptic Orbits"_:

| Parameter               | Symbol  | Value | Description                                             |
|-------------------------|---------|-------|---------------------------------------------------------|
| Radial Separation       | \( \rho_1 \) | 0.1 m | Separation along the radial axis                       |
| Cross-Track Separation  | \( \rho_3 \) | 0 m   | Separation along the cross-track axis                  |
| Radial and Along-Track Angle | \( \alpha \) | 0 rad | Angle between radial and along-track separation         |
| Radial and Cross-Track Angle | \( \beta \)  | \( \frac{\pi}{2} \) rad | Angle between radial and cross-track separation   |
| Drift per Revolution    | \( v_d \) | 0.000 m | Drift per orbital revolution                           |
| Along-Track Separation  | \( d \)  | -0.1 m | Separation along the orbital track                     |
| Adjusted Along-Track Separation | \( \rho_2 \) | \( \frac{2 \eta^2 d}{3 - \eta^2} \) m | Based on \( \eta \) |

### Diagram of Formation Geometry

![Formation Geometry](images/formation_geometry.png)

### Symbols and Formulae

- **\\( \\rho_1 \\):** Radial separation
- **\\( \\rho_3 \\):** Cross-track separation
- **\\( \\alpha \\):** Angle between radial and along-track separation
- **\\( \\beta \\):** Angle between radial and cross-track separation, given by \\( \\beta = \\alpha + 90^\\circ \\)
- **\\( v_d \\):** Drift per orbital revolution
- **\\( d \\):** Along-track separation
- **\\( \\rho_2 \\):** Adjusted along-track separation, calculated as:  
  \\[
  \\rho_2 = \\frac{2 \\eta^2 d}{3 - \\eta^2}
  \\]

These parameters are critical to accurately modeling the relative motion and geometry of satellite formations in elliptical orbits.


# Repo directory structure
```markdown
├── core
├── Testing
├── README.md
└── .gitignore
```

- **core** contains the functions and details related to core part of project that includes attitude dynamics models, kinematics, translation dynamics, etc.
- **Testing** contains the test of each function/modules.
# Simulation



# References

**1. Sengupta, P. and Vadali, S. (2007)**  
*Title:* Relative Motion and the Geometry of Formations in Keplerian Elliptic Orbits with Arbitrary Eccentricity  
*DOI:* [10.2514/1.25941](https://doi.org/10.2514/1.25941)  

**2. Traub, C., Fasoulas, S., and Herdrich, G. (2022)**  
*Title:* A planning tool for optimal three-dimensional formation flight maneuvers of satellites in VLEO using aerodynamic lift and drag via yaw angle deviations  
*DOI:* [10.1016/J.ACTAASTRO.2022.04.010](https://doi.org/10.1016/J.ACTAASTRO.2022.04.010)  

**3. Roscoe, C. W. T., Westphal, J. J., Griesbach, J. D., and Schaub, H. (2015)**  
*Title:* Formation Establishment and Reconfiguration Using Differential Elements in J2-Perturbed Orbits  
*Journal:* Journal of Guidance Control and Dynamics  
*DOI:* [10.2514/1.G000999](https://doi.org/10.2514/1.G000999)  

**4. Curtis, Howard D. (2020)**  
*Title:* Orbital Mechanics for Engineering Students: Revised Reprint  
*Publisher:* Butterworth-Heinemann  

**5. Hanspeter Schaub, John L. Junkins (2018)**  
*Title:* Analytical Mechanics of Space Systems  
*Publisher:* American Institute of Aeronautics and Astronautics, Incorporated  

**6. Vallado, D. A. (2001)**  
*Title:* Fundamentals of Astrodynamics and Applications, 4th ed.  
*Series:* Space Technology Library  
