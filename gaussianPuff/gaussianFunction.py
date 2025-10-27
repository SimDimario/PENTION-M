#gaussianFunction.py
import numpy as np
from scipy.special import erfcinv as erfcinv
from gaussianPuff.sigmaCalculation import calc_sigmas 
from numpy import sqrt

def gauss_func_plume(Q,u,dir1,x,y,z,xs,ys,H,STABILITY):
    """
    Calculate the Gaussian plume concentration at a point (x,y,z) from a stack
    located at (xs,ys) with height H, emitting a pollutant at rate Q
    with wind speed u and direction dir1.
    Param:
        Q: emission rate (kg/s)
        u: wind speed (m/s)
        dir1: wind direction (degrees, 0 is north)
        x, y, z: coordinates of the point where concentration is calculated (m)
        xs, ys: coordinates of the stack (m)
        H: height of the stack (m)
        STABILITY: stability class (ePasquill-Gifford stability class)
    Returns: 
        concentration at (x,y,z) (kg/m^3)
    """
    u1=u
    x1=x-xs # shift the coordinates so that stack is centre point
    y1=y-ys

    # components of u (wind) in x and y directions
    # -180 degrees the wind direct is wheret the wind is coming from
    # so we need to subtract 180 degrees to get the direction of the wind
    wx=u1*np.sin((dir1-180.)*np.pi/180.)
    wy=u1*np.cos((dir1-180.)*np.pi/180.)

    dot_product=wx*x1+wy*y1    # Angle between point x, y and the wind direction, so use scalar product:
    magnitudes=u1*np.sqrt(x1**2.+y1**2.)  # Magnitudes of the vectors
    subtended=np.arccos(dot_product/(magnitudes+1e-15)) # Avoid division by zero

    # distance to point x,y from stack
    hypotenuse=np.sqrt(x1**2.+y1**2.)

    # distance along the wind direction to perpendilcular line that intesects
    downwind=np.cos(subtended)*hypotenuse #x
    crosswind=np.sin(subtended)*hypotenuse #y

    ind=np.where(downwind>0.)
    C=np.zeros((len(x),len(y)))

    # calculate sigmas based on stability and distance downwind
    (sig_y,sig_z)=calc_sigmas(STABILITY,downwind)

    #sigma_y, sigma_z determnate the spread of the plume in the crosswind and vertical directions
    #sigma_y is the horizontal spread, sigma_z is the vertical spread
    #the concentration is highest at the stack and decreases with distance from the stack

    C[ind]=Q/(2.*np.pi*u1*sig_y[ind]*sig_z[ind]) \
        * np.exp(-crosswind[ind]**2./(2.*sig_y[ind]**2.))  \
        *(np.exp(-(z[ind]-H)**2./(2.*sig_z[ind]**2.)) + \
        np.exp(-(z[ind]+H)**2./(2.*sig_z[ind]**2.)) )
    return C

def gauss_func_puff(puff, x_grid, y_grid, z_grid, dt, stability, wind_speed, wind_dir):

    # Calcola sigmas in base al tempo passato
    downwind_dist = wind_speed * dt
    sig_y, sig_z = calc_sigmas(stability, np.array([downwind_dist]))

    # Coord. relative al puff
    x1 = x_grid - puff.x
    y1 = y_grid - puff.y
    z1 = z_grid - puff.z

    factor = puff.q / (2 * np.pi * sig_y * sig_z)
    C = factor * np.exp(-x1**2 / (2 * sig_y**2)) \
             * np.exp(-y1**2 / (2 * sig_y**2)) \
             * (np.exp(-(z1)**2 / (2 * sig_z**2)) + np.exp(-(z1 + 2*puff.z)**2 / (2 * sig_z**2)))
    
    return C