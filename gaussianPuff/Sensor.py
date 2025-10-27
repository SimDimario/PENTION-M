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
    def __init__(self, sensor_id, x: float, y: float, z: float = 2.,
                 noise_level: float = 0.1, is_fault: bool = False):
        self.id = sensor_id
        self.x = x
        self.y = y
        self.z = z
        self.noise_level = noise_level
        self.concentrations = None
        self.noisy_concentrations = None
        self.times= None
        self.is_fault = is_fault

    def sample_substance(self, conc_field, x_grid, y_grid, t_grid):
        """
        Campiona il campo di concentrazione 3D (x, y, t) alla posizione del sensore.
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

        interpolator = RegularGridInterpolator((x_sorted, y_sorted, times), conc_field, bounds_error=False, fill_value=0.0)
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
        Genera serie temporali sintetiche di concentrazione e spettro di massa
        senza conoscere la sorgente reale.
        """

        if self.is_fault:
            print(f"Sensor {self.id} is faulty. No data sampled.")
            return {
                "times": [],
                "concentrations": np.array([], dtype=float),
                "noisy_concentrations": np.array([], dtype=float),
                "mass_spectrum": None,
                "source_pos": None,
                "source_intensity": None
            }

        # 1. Sorgente casuale nella zona
        src_x = np.random.uniform(x_grid.min(), x_grid.max())
        src_y = np.random.uniform(y_grid.min(), y_grid.max())
        src_intensity = np.random.uniform(0.1, 1.0)

        # 2. Genera campo di concentrazione sintetico (gaussiano centrato sulla sorgente)
        conc_field = np.zeros((len(x_grid), len(y_grid), len(t_grid)))
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                r2 = (x - src_x)**2 + (y - src_y)**2
                for k, t in enumerate(t_grid):
                    # decadimento con distanza e tempo
                    conc_field[i, j, k] = src_intensity * np.exp(-r2 / (2*0.01)) * np.exp(-0.1*t)

        # 3. Interpolazione per ottenere la serie temporale al sensore
        from scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator((x_grid, y_grid, t_grid), conc_field, 
                                            bounds_error=False, fill_value=0.0)
        coords = [(self.x, self.y, t) for t in t_grid]
        self.concentrations = np.array([interpolator(c) for c in coords])
        self.times = t_grid

        # 4. Aggiungi rumore
        if self.noise_level > 0:
            noise_std = self.noise_level * np.maximum(self.concentrations, 1e-6)
            noise = np.random.normal(0, noise_std)
            self.noisy_concentrations = np.clip(self.concentrations + noise, 0, None)
        else:
            self.noisy_concentrations = self.concentrations.copy()

        # 5. Genera spettro di massa basato sulla concentrazione media
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
            "source_pos": (src_x, src_y),  # opzionale, utile per debug
            "source_intensity": src_intensity
        }

    def _generate_mass_spectra(self, df=pd.read_csv(r"C:/Users/claud/Desktop/PENTION/ClassificatoreNPS/datasetNPS/1-s2.0-S2468170923000358-mmc1.csv", sep=',', header=0), n_generic=9, noise_level=0.01):
        """
        Genera spettri di massa per un sensore basandosi su un dataset reale.

        Args:
            df (pd.DataFrame): 
            n_generic (int): Numero di spettri generici da generare oltre allo spettro NPS.
            noise_level (float): Rumore gaussiano da aggiungere agli spettri.
        Returns:
            list of np.array: Lista di spettri
        """

        if self.is_fault:
            print(f"Sensor {self.id} is faulty. No mass spectrum generated.")
            return np.full(600, np.nan, dtype=float)

        df_nps = df[df['label'] != 'Other'] 
        row_nps = df_nps.sample(n=1).iloc[0]
        spectrum_nps = row_nps[df.columns[1:601]].values.astype(float)  
        spectrum_nps += np.random.normal(0, noise_level, size=spectrum_nps.shape)
        
        mass_spectra = [spectrum_nps]

        for _ in range(n_generic):
            row_generic = df.sample(n=1).iloc[0]
            spectrum_generic = row_generic[df.columns[1:601]].values.astype(float)
            spectrum_generic += np.random.normal(0, noise_level, size=spectrum_generic.shape)
            mass_spectra.append(spectrum_generic)

        return mass_spectra

    def _simulate_mass_spectrum(self, nps=False):
        """
        Simula uno spettro di massa sintetico
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
            # Picchi distintivi per una sostanza NPS
            peak_positions = np.random.choice(range(num_bins), size=3, replace=False)
            for pos in peak_positions:
                spectrum[pos] += np.random.uniform(0.1, 1.0)
        else:
            # Picchi casuali generici
            peak_positions = np.random.choice(range(num_bins), size=2, replace=False)
            for pos in peak_positions:
                spectrum[pos] += np.random.uniform(0.01, 0.05)  # più piccoli e diffusi
        
        spectrum += baseline * (1 + np.random.rand(num_bins))
        return spectrum

    def plot_timeseries(self, use_noisy=True):

        if self.times is None:
            raise ValueError("Il sensore non ha ancora campionato dati.")

        data = self.noisy_concentrations if use_noisy else self.concentrations

        if data is None:
            raise ValueError("I dati di concentrazione non sono disponibili per il sensore.")
        
        plt.plot(self.times, data, label=f"Sensor {self.id}")
        plt.xlabel("Tempo (h)")
        plt.ylabel("Concentrazione [μg/m³]")
        plt.title(f"Andamento temporale - Sensore {self.id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _fault_probability(self, wind_speed, stability_value, RH, wind_type):
        
        # Base probability
        base_prob = 0.1
        
        # Aumenta probabilità con vento forte
        if wind_speed > 6.0:
            base_prob += 0.2
        
        # Aumenta probabilità se stabilità molto instabile
        if stability_value in [PasquillGiffordStability.VERY_UNSTABLE, PasquillGiffordStability.VERY_STABLE]:
            base_prob += 0.15
            
        # Aumenta probabilità con alta umidità
        if RH > 0.8:
            base_prob += 0.2
            
        # Vento fluttuante aumenta la probabilità
        if wind_type == WindType.FLUCTUATING:
            base_prob += 0.1
            
        # Limita la probabilità al massimo di 0.75 per evitare valori troppo estremi
        return min(base_prob, 0.75)
        
    def run_sensor(self, wind_speed, stability_value, RH, wind_type):
        '''
        Esegue il campionamento del sensore.
        Campiona meteorologia, sostanza e simula lo spettro di massa.
        Se il sensore è in stato di guasto, non campiona dati.
        
        Returns:
            dict: Dati campionati dal sensore, inclusi tempi, spettro di massa,
                velocità del vento, tipo di vento, tipo di stabilità, valore di stabilità,
                umidità, dimensione secca e umidità relativa (RH).
        '''
        
        self.is_fault = np.random.rand() < self._fault_probability(wind_speed, stability_value, RH, wind_type)
        
        mass_spectra = self._generate_mass_spectra(noise_level=self.noise_level)

        return mass_spectra
    
class SensorAir:
    def __init__(self, sensor_id, x: float, y: float, z: float):
        self.id = sensor_id
        self.x = x
        self.y = y
        self.z = z

    def sample_meteorology(self):
        wind_type = random.choice([WindType.CONSTANT ,WindType.PREVAILING, WindType.FLUCTUATING])
        stability_type = StabilityType.CONSTANT
        if stability_type == StabilityType.CONSTANT:
            stability_value = random.choice([
                PasquillGiffordStability.VERY_UNSTABLE,
                PasquillGiffordStability.MODERATELY_UNSTABLE,
                PasquillGiffordStability.SLIGHTLY_UNSTABLE,
                PasquillGiffordStability.NEUTRAL,
                PasquillGiffordStability.MODERATELY_STABLE,
                PasquillGiffordStability.VERY_STABLE
            ])
        else:
            # Fallback to a neutral stability ensuring the correct enum type
            stability_value = PasquillGiffordStability.NEUTRAL
        
        wind_speed = self._assign_wind_speed(stability_value)  
        humidify = random.choice([True, False])
        dry_size = 1.0
        RH = round(np.random.uniform(0, 0.99), 2) if humidify else 0.0

        return wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH

    def _assign_wind_speed(self, stability: PasquillGiffordStability) -> float:
        """
        Restituisce una velocità del vento (m/s) coerente con la stabilità atmosferica.
        I range sono basati su letteratura meteorologica semplificata.
        """
        if stability == PasquillGiffordStability.VERY_UNSTABLE:  # A
            return round(random.uniform(2.0, 6.0), 2)
        elif stability == PasquillGiffordStability.MODERATELY_UNSTABLE:  # B
            return round(random.uniform(2.0, 5.0), 2)
        elif stability == PasquillGiffordStability.SLIGHTLY_UNSTABLE:  # C
            return round(random.uniform(3.0, 6.5), 2)
        elif stability == PasquillGiffordStability.NEUTRAL:  # D
            return round(random.uniform(4.0, 8.0), 2)
        elif stability == PasquillGiffordStability.MODERATELY_STABLE:  # E
            return round(random.uniform(1.0, 4.0), 2)
        elif stability == PasquillGiffordStability.VERY_STABLE:  # F
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

   