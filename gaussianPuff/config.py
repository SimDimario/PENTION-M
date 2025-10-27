#config.py
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

class DispersionModelType(Enum):
    PLUME = "plume"
    PUFF = "puff"

class ConfigPuff():
    def __init__(self, puff_interval: float = 1, max_puff_age: float = 6):
        self.puff_interval = puff_interval
        self.max_puff_age = max_puff_age

class OutputType(Enum):
    PLAN_VIEW = 1
    HEIGHT_SLICE = 2
    SURFACE_TIME = 3
    NO_PLOT = 4

    def from_string(output_type: str):
        try:
            return OutputType[output_type]
        except KeyError:
            raise ValueError(f"No OutpuType member matches '{output_type}'")

class WindType(Enum):
    CONSTANT = 1
    FLUCTUATING = 2
    PREVAILING = 3

    def from_string(wind_type: str):
        try:
            return WindType[wind_type]
        except KeyError:
            raise ValueError(f"No WindType member matches '{wind_type}'")

class PasquillGiffordStability(Enum):
    VERY_UNSTABLE = 1
    MODERATELY_UNSTABLE = 2
    SLIGHTLY_UNSTABLE = 3
    NEUTRAL = 4
    MODERATELY_STABLE= 5
    VERY_STABLE = 6

    def from_string(psg_stab: str):
        try:
            return PasquillGiffordStability[psg_stab]
        except KeyError:
            raise ValueError(f"No Pasquill-Gifford member matches '{psg_stab}'")

class StabilityType(Enum):
    CONSTANT = 1
    ANNUAL = 2

    def from_string(stab_type: str):
        try:
            return StabilityType[stab_type]
        except KeyError:
            raise ValueError(f"No StabilityType member matches '{stab_type}'")

class NPS(Enum):
    CANNABINOID_ANALOGUES = 0
    CATHINONE_ANALOGUES = 1
    PHENETHYLAMINE_ANALOGUES = 2
    PIPERAZINE_ANALOGUES = 3
    TRYPTAMINE_ANALOGUES = 4
    FENTANYL_ANALOGUES = 5
    OTHER_COMPOUNDS = 6

    @staticmethod
    def from_string(nps_string: str):
        nps = nps_string.replace(" ", "_").replace("-", "_").upper()
        try:
            return NPS[nps]
        except KeyError:
            raise ValueError(f"No NPS member matches '{nps_string}'")

# Definizione delle proprietà fisiche per ciascun tipo di NPS
nps_properties = {
    NPS.CANNABINOID_ANALOGUES: {"nu": 1, "rho_s": 1.2e3, "Ms": 314.46},
    NPS.CATHINONE_ANALOGUES: {"nu": 1, "rho_s": 1.3e3, "Ms": 149.23},
    NPS.PHENETHYLAMINE_ANALOGUES: {"nu": 2, "rho_s": 1.1e3, "Ms": 121.18},
    NPS.PIPERAZINE_ANALOGUES: {"nu": 2, "rho_s": 1.25e3, "Ms": 160.25},
    NPS.TRYPTAMINE_ANALOGUES: {"nu": 1, "rho_s": 1.18e3, "Ms": 204.27},
    NPS.FENTANYL_ANALOGUES: {"nu": 1, "rho_s": 1.24e3, "Ms": 336.47},
    NPS.OTHER_COMPOUNDS: {"nu": 1, "rho_s": 1.3e3, "Ms": 250.00},
}

@dataclass
class ModelConfig:
    days: int
    RH: float
    aerosol_type: NPS
    humidify: bool
    stability_profile: StabilityType
    stability_value: PasquillGiffordStability
    wind_type: WindType
    wind_speed: float
    output: OutputType
    stacks: List[Tuple[float, float, float, float]]  # List of (x, y, Q, H)
    dry_size: float = 60e-9
    x_slice: int = 26
    y_slice: int = 1
    grid_size: int= 500
    dispersion_model: DispersionModelType = DispersionModelType.PLUME
    config_puff: 'Optional[ConfigPuff]' = field(default_factory=ConfigPuff) if dispersion_model == DispersionModelType.PUFF else None

    def to_dict(self):
        d = asdict(self)

        d["aerosol_type"] = self.aerosol_type.name if self.aerosol_type else None
        d["stability_profile"] = self.stability_profile.name if self.stability_profile else None
        d["stability_value"] = self.stability_value.name if self.stability_value else None
        d["wind_type"] = self.wind_type.name if self.wind_type else None
        d["output"] = self.output.name if self.output else None
        d["dispersion_model"] = self.dispersion_model.name if self.dispersion_model else None

        if self.config_puff:
            d["config_puff"] = (
                self.config_puff.to_dict() if hasattr(self.config_puff, "to_dict") else asdict(self.config_puff)
            )

        return d