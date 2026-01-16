import numpy as np
from scipy.special import erfcinv as erfcinv
from gaussianPuff.sigmaCalculation import calc_sigmas
from numpy import sqrt


def gauss_func_plume(Q, u, dir1, x, y, z, xs, ys, H, STABILITY):
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
    u1 = u
    x1 = x - xs
    y1 = y - ys
    wx = u1 * np.sin((dir1 - 180.0) * np.pi / 180.0)
    wy = u1 * np.cos((dir1 - 180.0) * np.pi / 180.0)
    dot_product = wx * x1 + wy * y1
    magnitudes = u1 * np.sqrt(x1**2.0 + y1**2.0)
    subtended = np.arccos(dot_product / (magnitudes + 1e-15))
    hypotenuse = np.sqrt(x1**2.0 + y1**2.0)
    downwind = np.cos(subtended) * hypotenuse
    crosswind = np.sin(subtended) * hypotenuse
    ind = np.where(downwind > 0.0)
    C = np.zeros((len(x), len(y)))
    (sig_y, sig_z) = calc_sigmas(STABILITY, downwind)
    C[ind] = (
        Q
        / (2.0 * np.pi * u1 * sig_y[ind] * sig_z[ind])
        * np.exp(-crosswind[ind] ** 2.0 / (2.0 * sig_y[ind] ** 2.0))
        * (
            np.exp(-((z[ind] - H) ** 2.0) / (2.0 * sig_z[ind] ** 2.0))
            + np.exp(-((z[ind] + H) ** 2.0) / (2.0 * sig_z[ind] ** 2.0))
        )
    )
    return C


def gauss_func_puff(puff, x_grid, y_grid, z_grid, dt, stability, wind_speed, wind_dir):
    downwind_dist = wind_speed * dt
    sig_y, sig_z = calc_sigmas(stability, np.array([downwind_dist]))
    x1 = x_grid - puff.x
    y1 = y_grid - puff.y
    z1 = z_grid - puff.z
    factor = puff.q / (2 * np.pi * sig_y * sig_z)
    C = (
        factor
        * np.exp(-(x1**2) / (2 * sig_y**2))
        * np.exp(-(y1**2) / (2 * sig_y**2))
        * (
            np.exp(-((z1) ** 2) / (2 * sig_z**2))
            + np.exp(-((z1 + 2 * puff.z) ** 2) / (2 * sig_z**2))
        )
    )
    return C
