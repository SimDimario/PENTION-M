#gauss_model.py
import numpy as np
from scipy.special import erfcinv
from gaussianPuff.config import ModelConfig, StabilityType, WindType, NPS, nps_properties, OutputType, DispersionModelType
from gaussianPuff.gaussianFunction import gauss_func_plume, gauss_func_puff

class Puff:
    def __init__(self, x, y, z, q, t_release):
        self.x = x
        self.y = y
        self.z = z
        self.q = q  # massa del puff
        self.t_release = t_release

# Funzione di crescita igroscopica basata su Köhler
def apply_hygroscopic_growth(C1: np.ndarray, RH: float, dry_size: float, nps_type: NPS) -> np.ndarray:
    
    RH = np.clip(RH, 1e-3, 0.99)  # Avoid instable values

    Mw = 18e-3

    #NPS physical properties
    rho_s = nps_properties[nps_type]["rho_s"]
    Ms = nps_properties[nps_type]["Ms"]
    nu = nps_properties[nps_type]["nu"]

    # Dry mass and moles
    mass=np.pi/6.*rho_s*dry_size**3.
    moles=mass/Ms
    
    # Humidified mass and moles
    nw=RH*nu*moles/(1.-RH)
    mass2=nw*Mw+moles*Ms
    C1_humidified=C1*mass2/mass

    return C1_humidified

from typing import Optional, Tuple

def run_dispersion_model(config: ModelConfig, bounds: Optional[Tuple] = None):

    #Spatial grid setup
    """dxy = 5000 / 499 #grid size in meters
    dz = 10
    x = np.mgrid[-2500:2500 + dxy:dxy] #-2500 to 2500 with step dxy (meter)"""
    if bounds is None:
        L = 4999
        dxy = 10
        N = int(L / dxy) + 1
        x = np.linspace(-L/2, L/2, N)
        y = x.copy()
    else:
        x_min, y_min, x_max, y_max = bounds
        grid_size = config.grid_size
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
    
        dx = (x_max - x_min) / (grid_size - 1)
        dy = (y_max - y_min) / (grid_size - 1)
        dxy = (dx + dy) / 2  

    # Calcolo dz 
    dz = x[1] - x[0]
    z_max = 100  # altezza massima da considerare 
    z = np.arange(0, z_max + dz, dz)

    times = np.arange(1, config.days * 24 + 1) / 24.0

    # Stability setup
    if config.stability_profile == StabilityType.CONSTANT:
        stability = np.full_like(times, config.stability_value.value)
        stability_label = f"Stability {config.stability_value.value}"
    else:
        stability = np.round(2.5 * np.cos(times * 2 * np.pi / 365.) + 3.5)
        stability_label = "Annual cycle"

    # Output grid setup
    if config.output == OutputType.PLAN_VIEW or config.output == OutputType.SURFACE_TIME or config.output == OutputType.NO_PLOT:
        C1=np.zeros((len(x),len(y),len(times))); # array to store data, initialised to be zero
        x_grid,y_grid=np.meshgrid(x,y); # x and y defined at all positions on the grid
        z_grid=np.zeros(np.shape(x_grid));    # z is defined to be at ground level.
    elif config.output == OutputType.HEIGHT_SLICE:
        z=np.mgrid[0:500+dz:dz];       # z-grid
        C1=np.zeros((len(y),len(z),len(times))); # array to store data, initialised to be zero
        [y_grid,z_grid]=np.meshgrid(y,z); # y and z defined at all positions on the grid
        x_grid=x[config.x_slice]*np.ones(np.shape(y));    # x is defined to be x at x_slice       
   
    # Wind speed and direction setup
    wind_speed = config.wind_speed * np.ones_like(times) #m/s
    if config.wind_type == WindType.CONSTANT: #stable directions
        wind_dir = np.zeros_like(times)
        wind_label = "Constant wind"
    elif config.wind_type == WindType.FLUCTUATING: # variable directions
        wind_dir = 360. * np.random.rand(len(times))
        wind_label = "Fluctuating wind"
    elif config.wind_type == WindType.PREVAILING: #
        wind_dir = -np.sqrt(2.) * erfcinv(2. * np.random.rand(len(times))) * 40.
        wind_dir = np.mod(wind_dir, 360)
        wind_label = "Prevailing wind"
    else:
        raise ValueError("Unsupported wind type")
        
    if config.dispersion_model == DispersionModelType.PLUME:
        for t in range(len(times)):
            for (x_s, y_s, q, h) in config.stacks:
                C = gauss_func_plume(q, wind_speed[t], wind_dir[t], x_grid, y_grid, z_grid, #type:ignore
                            x_s, y_s, h, stability[t])
                C1[:, :, t] += C #type:ignore
    elif config.dispersion_model == DispersionModelType.PUFF:
        if config.config_puff is None:
            raise ValueError("config.config_puff must not be None when using the PUFF dispersion model.")
        puff_interval = config.config_puff.puff_interval 
        max_puff_age = config.config_puff.max_puff_age
        puffs = []  
        for t in range(len(times)):
                # Create a puff every puff_interval hours
                if t % puff_interval == 0:
                    for (x_s, y_s, q, h) in config.stacks:
                        puff = Puff(x_s, y_s, h, q, t)
                        puffs.append(puff)

                active_puffs = [p for p in puffs if 0 <= t - p.t_release <= max_puff_age]
                
                # Update puff positions based on wind speed and direction
                for puff in active_puffs:
                    dt = t - puff.t_release
                    if dt == 0:
                        continue  # no puff released yet

                    # Converti la direzione del vento (da cui proviene) in direzione del movimento (verso cui va)
                    theta_rad = np.radians(270 - wind_dir[t])
                    dx = wind_speed[t] * np.cos(theta_rad) * 3600  # distanza oraria in metri
                    dy = wind_speed[t] * np.sin(theta_rad) * 3600

                    puff.x += dx
                    puff.y += dy

                    C = gauss_func_puff(puff, x_grid, y_grid, z_grid, dt, stability[t], wind_speed[t], wind_dir[t]) #type:ignore
                    C1[:, :, t] += C #type:ignore

    if config.humidify:
        C1= apply_hygroscopic_growth(C1, config.RH, config.dry_size, config.aerosol_type) #type:ignore

    return C1, (x_grid, y_grid, z_grid), times, stability, wind_dir, stability_label, wind_label, (puffs if config.dispersion_model == DispersionModelType.PUFF else None) #, (sigma_y, sigma_z if config.dispersion_model == DispersionModelType.PUFF else None) #type:ignore

