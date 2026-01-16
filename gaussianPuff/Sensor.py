import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import RegularGridInterpolator
import os
import sys
from gaussianPuff.config import WindType, StabilityType, PasquillGiffordStability
import numpy as np
import pandas as pd

"""sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ClassificatoreNPS import service_clf_nps"""


class SensorSubstance:
    def __init__(
        self,
        sensor_id,
        x: float,
        y: float,
        z: float = 2.0,
        noise_level: float = 0.1,
        is_fault: bool = False,
    ):
        self.id = sensor_id
        self.x = x
        self.y = y
        self.z = z
        self.noise_level = noise_level
        self.concentrations = None
        self.noisy_concentrations = None
        self.times = None
        self.is_fault = is_fault

    def sample_substance(self, conc_field, x_grid, y_grid, t_grid):
        """
        Sample the 3D concentration field (x, y, t) at the sensor position.
        """

        if self.is_fault:
            print(f"Sensor {self.id} is faulty. No data sampled.")
            self.concentrations = np.array([], dtype=float)
            self.noisy_concentrations = np.array([], dtype=float)
            self.times = []
            return

        x_sorted = np.sort(np.unique(x_grid))
        y_sorted = np.sort(np.unique(y_grid))
        times = np.sort(np.unique(t_grid))
        self.times = times

        interpolator = RegularGridInterpolator(
            (x_sorted, y_sorted, times), conc_field, bounds_error=False, fill_value=0.0
        )
        coords = [(self.x, self.y, t) for t in times]
        self.concentrations = np.array([interpolator(c) for c in coords])

        if self.noise_level > 0.0:
            noise_std = self.noise_level * np.maximum(self.concentrations, 1e-6)
            noise = np.random.normal(0, noise_std)
            self.noisy_concentrations = np.clip(self.concentrations + noise, 0, None)
        else:
            self.noisy_concentrations = self.concentrations.copy()

    def sample_substance_synthetic(self, x_grid, y_grid, t_grid):
        """
        Generate synthetic concentration and mass spectral time series
        without knowing the actual source.
        """

        if self.is_fault:
            print(f"Sensor {self.id} is faulty. No data sampled.")
            return {
                "times": [],
                "concentrations": np.array([], dtype=float),
                "noisy_concentrations": np.array([], dtype=float),
                "mass_spectrum": None,
                "source_pos": None,
                "source_intensity": None,
            }
        src_x = np.random.uniform(x_grid.min(), x_grid.max())
        src_y = np.random.uniform(y_grid.min(), y_grid.max())
        src_intensity = np.random.uniform(0.1, 1.0)
        conc_field = np.zeros((len(x_grid), len(y_grid), len(t_grid)))
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                r2 = (x - src_x) ** 2 + (y - src_y) ** 2
                for k, t in enumerate(t_grid):
                    conc_field[i, j, k] = (
                        src_intensity * np.exp(-r2 / (2 * 0.01)) * np.exp(-0.1 * t)
                    )
        from scipy.interpolate import RegularGridInterpolator

        interpolator = RegularGridInterpolator(
            (x_grid, y_grid, t_grid), conc_field, bounds_error=False, fill_value=0.0
        )
        coords = [(self.x, self.y, t) for t in t_grid]
        self.concentrations = np.array([interpolator(c) for c in coords])
        self.times = t_grid
        if self.noise_level > 0:
            noise_std = self.noise_level * np.maximum(self.concentrations, 1e-6)
            noise = np.random.normal(0, noise_std)
            self.noisy_concentrations = np.clip(self.concentrations + noise, 0, None)
        else:
            self.noisy_concentrations = self.concentrations.copy()
        mean_conc = float(np.mean(np.asarray(self.noisy_concentrations, dtype=float)))
        num_bins = 600
        baseline = np.random.rand(num_bins) * 0.01
        spectrum = baseline.copy()
        peak_positions = np.random.choice(range(num_bins), size=3, replace=False)
        for pos in peak_positions:
            spectrum[pos] += np.random.uniform(0.1, 1.0) * mean_conc
        if self.noise_level > 0:
            spectrum += baseline * (1 + np.random.rand(num_bins))

        return {
            "times": self.times,
            "concentrations": self.concentrations,
            "noisy_concentrations": self.noisy_concentrations,
            "mass_spectrum": spectrum,
            "source_pos": (src_x, src_y),
            "source_intensity": src_intensity,
        }

    import os

    default_dataset_path = (
        "/PentionSystem/ClassificatoreNPS/datasetNPS/1-s2.0-S2468170923000358-mmc1.csv"
    )

    def _generate_mass_spectra(
        self,
        df=pd.read_csv(default_dataset_path, sep=",", header=0),
        n_generic=9,
        noise_level=0.01,
    ):
        """
        Generate mass spectra for a sensor based on a real dataset.

        Args:
            df (pd.DataFrame):
            n_generic (int): Number of generic spectra to generate in addition to the NPS spectrum.
            noise_level (float): Gaussian noise to be added to the spectra.
        Returns:
            list of np.array: List of ghosts
        """

        if self.is_fault:
            print(f"Sensor {self.id} is faulty. No mass spectrum generated.")
            return np.full(600, np.nan, dtype=float)

        df_nps = df[df["label"] != "Other"]
        row_nps = df_nps.sample(n=1).iloc[0]
        spectrum_nps = row_nps[df.columns[1:601]].values.astype(float)
        spectrum_nps += np.random.normal(0, noise_level, size=spectrum_nps.shape)

        mass_spectra = [spectrum_nps]

        for _ in range(n_generic):
            row_generic = df.sample(n=1).iloc[0]
            spectrum_generic = row_generic[df.columns[1:601]].values.astype(float)
            spectrum_generic += np.random.normal(
                0, noise_level, size=spectrum_generic.shape
            )
            mass_spectra.append(spectrum_generic)

        return mass_spectra

    def _simulate_mass_spectrum(self, nps=False):
        """
        Simulates a synthetic mass spectrum
        """
        num_bins = 600
        np.random.seed(None)
        baseline = np.random.rand(num_bins) * 0.01
        peak_positions = np.random.choice(range(num_bins), size=3, replace=False)
        spectrum = baseline.copy()

        """if self.noisy_concentrations is None or len(self.noisy_concentrations) == 0:
            mean_conc = 0.0
        else:
            mean_conc = float(np.mean(np.asarray(self.noisy_concentrations, dtype=float)))
        """

        if nps:
            peak_positions = np.random.choice(range(num_bins), size=3, replace=False)
            for pos in peak_positions:
                spectrum[pos] += np.random.uniform(0.1, 1.0)
        else:
            peak_positions = np.random.choice(range(num_bins), size=2, replace=False)
            for pos in peak_positions:
                spectrum[pos] += np.random.uniform(0.01, 0.05)

        spectrum += baseline * (1 + np.random.rand(num_bins))
        return spectrum

    def plot_timeseries(self, use_noisy=True):

        if self.times is None:
            raise ValueError("The sensor has not yet sampled data.")

        data = self.noisy_concentrations if use_noisy else self.concentrations

        if data is None:
            raise ValueError("Concentration data is not available for the sensor.")

        plt.plot(self.times, data, label=f"Sensor {self.id}")
        plt.xlabel("Time (h)")
        plt.ylabel("Concentration [μg/m³]")
        plt.title(f"Temporal evolution - Sensor {self.id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _fault_probability(self, wind_speed, stability_value, RH, wind_type):
        base_prob = 0.1
        if wind_speed > 6.0:
            base_prob += 0.2
        if stability_value in [
            PasquillGiffordStability.VERY_UNSTABLE,
            PasquillGiffordStability.VERY_STABLE,
        ]:
            base_prob += 0.15
        if RH > 0.8:
            base_prob += 0.2
        if wind_type == WindType.FLUCTUATING:
            base_prob += 0.1
        return min(base_prob, 0.75)

    def run_sensor(self, wind_speed, stability_value, RH, wind_type):
        """
        Performs sensor sampling.
        Samples meteorology, substance, and simulates the mass spectrum.
        If the sensor is in a fault state, it does not sample data.

        Returns:
            dict: Data sampled from the sensor, including time, mass spectrum,
                wind speed, wind type, stability type, stability value,
                humidity, dry size, and relative humidity (RH).
        """

        self.is_fault = np.random.rand() < self._fault_probability(
            wind_speed, stability_value, RH, wind_type
        )

        mass_spectra = self._generate_mass_spectra(noise_level=self.noise_level)

        return mass_spectra


class SensorAir:
    def __init__(self, sensor_id, x: float, y: float, z: float):
        self.id = sensor_id
        self.x = x
        self.y = y
        self.z = z

    def sample_meteorology(self):
        wind_type = random.choice(
            [WindType.CONSTANT, WindType.PREVAILING, WindType.FLUCTUATING]
        )
        stability_type = random.choice([StabilityType.CONSTANT, StabilityType.ANNUAL])
        if stability_type == StabilityType.CONSTANT:
            stability_value = random.choice(
                [
                    PasquillGiffordStability.VERY_UNSTABLE,
                    PasquillGiffordStability.MODERATELY_UNSTABLE,
                    PasquillGiffordStability.SLIGHTLY_UNSTABLE,
                    PasquillGiffordStability.NEUTRAL,
                    PasquillGiffordStability.MODERATELY_STABLE,
                    PasquillGiffordStability.VERY_STABLE,
                ]
            )
        else:
            stability_value = PasquillGiffordStability.NEUTRAL

        wind_speed = self._assign_wind_speed(stability_value)
        humidify = random.choice([True, False])
        dry_size = round(np.random.uniform(0.5, 2.5), 2)
        RH = round(np.random.uniform(0, 0.99), 2) if humidify else 0.0

        return (
            wind_speed,
            wind_type,
            stability_type,
            stability_value,
            humidify,
            dry_size,
            RH,
        )

    def _assign_wind_speed(self, stability: PasquillGiffordStability) -> float:
        """
        Returns a wind speed (m/s) consistent with atmospheric stability.
        The ranges are based on simplified meteorological literature.
        """
        if stability == PasquillGiffordStability.VERY_UNSTABLE:
            return round(random.uniform(2.0, 6.0), 2)
        elif stability == PasquillGiffordStability.MODERATELY_UNSTABLE:
            return round(random.uniform(2.0, 5.0), 2)
        elif stability == PasquillGiffordStability.SLIGHTLY_UNSTABLE:
            return round(random.uniform(3.0, 6.5), 2)
        elif stability == PasquillGiffordStability.NEUTRAL:
            return round(random.uniform(4.0, 8.0), 2)
        elif stability == PasquillGiffordStability.MODERATELY_STABLE:
            return round(random.uniform(1.0, 4.0), 2)
        elif stability == PasquillGiffordStability.VERY_STABLE:
            return round(random.uniform(0.5, 3.0), 2)
        else:
            return round(random.uniform(2.0, 6.0), 2)


"""if __name__ == "__main__":
    
    sensor_air = SensorAir(sensor_id=0, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()
    print(f"Wind Speed: {wind_speed} m/s, Wind Type: {wind_type}, Stability: {stability_value}, RH: {RH}")

    sensor = SensorSubstance(sensor_id=2, x=50.0, y=50.0, z=2.0, noise_level=0.05)
    mas=sensor.run_sensor(wind_speed, stability_value, RH, wind_type)
    print(sensor.is_fault)
    print(mas[0])
    print(type(mas[0]))
    print(type(mas))
    print(len(mas[0]))

    result=service_clf_nps.pipe_clf_dnn(mas)
    print(result)
    result=service_clf_nps.pipe_clf_brf(mas)
    print(result)"""
