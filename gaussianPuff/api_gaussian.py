from fastapi import FastAPI
import os, sys
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import logging
import numpy as np
import random
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.config import (
    ModelConfig,
    WindType,
    StabilityType,
    PasquillGiffordStability,
    NPS,
    OutputType,
    DispersionModelType,
    ConfigPuff,
)
from gaussianPuff.plot_utils import plot_plan_view
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class ModelConfigRequest(BaseModel):
    days: int
    RH: float
    aerosol_type: str
    humidify: bool
    stability_profile: str
    stability_value: str
    wind_type: str
    wind_speed: float
    output: str
    stacks: List[Tuple[float, float, float, float]]
    dry_size: Optional[float] = 60e-9
    x_slice: Optional[int] = 26
    y_slice: Optional[int] = 1
    grid_size: Optional[int] = 500
    dispersion_model: str
    config_puff: Optional[dict] = None


class Payload(BaseModel):
    config: ModelConfigRequest
    bounds: List[float] = Field(..., min_items=4, max_items=4)


app = FastAPI()

from gaussianPuff.config import PasquillGiffordStability


@app.get("/get_meteo")
def get_meteo():
    """
    It returns pseudo-realistic and variable weather conditions
    for each simulation, without running GaussianPuff.
    """
    try:
        hour = datetime.utcnow().hour
        if 6 <= hour < 12:
            temperature = random.uniform(8.0, 16.0)
        elif 12 <= hour < 18:
            temperature = random.uniform(12.0, 24.0)
        else:
            temperature = random.uniform(4.0, 18.0)
        humidity = random.uniform(0.5, 0.9)
        base_speed = 4.0
        wind_speed = max(0.5, random.gauss(base_speed, 1.0))
        wind_dir_deg = (225.0 + random.uniform(-60.0, 60.0)) % 360.0
        classes = ["A", "B", "C", "D", "E", "F"]
        weights = [0.05, 0.10, 0.25, 0.30, 0.20, 0.10]
        stability_class = random.choices(classes, weights=weights, k=1)[0]

        return {
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "wind_speed": round(wind_speed, 2),
            "wind_dir_deg": round(wind_dir_deg, 2),
            "stability_class": stability_class,
        }

    except Exception as e:
        return {
            "temperature": 20.0,
            "humidity": 0.55,
            "wind_speed": 4.0,
            "wind_dir_deg": 225.0,
            "stability_class": "D",
            "error": str(e),
        }


@app.post("/start_simulation")
def start_simulation(payload: dict):
    logger.info("Request received /start_simulation")
    try:
        raw_config = payload.get("config", {})
        logger.info(raw_config)
        wind_type = WindType.from_string(raw_config["wind_type"])
        stability_type = StabilityType.from_string(raw_config["stability_profile"])
        output_type = OutputType.from_string(raw_config["output"])
        stability_value = PasquillGiffordStability.from_string(
            raw_config["stability_value"]
        )
        nps_type = NPS.from_string(raw_config["aerosol_type"])
        dispersion_model = (
            DispersionModelType(raw_config["dispersion_model"].lower())
            if "dispersion_model" in raw_config and raw_config["dispersion_model"]
            else DispersionModelType.PLUME
        )

        config = ModelConfig(
            days=raw_config["days"],
            RH=raw_config["RH"],
            aerosol_type=nps_type,
            humidify=raw_config["humidify"],
            stability_profile=stability_type,
            stability_value=stability_value,
            wind_type=wind_type,
            wind_speed=raw_config["wind_speed"],
            wind_dir_deg=raw_config.get("wind_dir_deg", 225.0),
            output=output_type,
            stacks=raw_config["stacks"],
            dry_size=raw_config["dry_size"],
            x_slice=raw_config["x_slice"],
            y_slice=raw_config["y_slice"],
            grid_size=raw_config.get("grid_size", 500),
            dispersion_model=dispersion_model,
            config_puff=(
                ConfigPuff(**raw_config["config_puff"])
                if raw_config.get("config_puff")
                else None
            ),
        )
        bounds = payload.get("bounds", None)
        logger.info(f"Template configuration created: {config}")
        logger.info(f"Bounds received: {bounds}")

        result = run_dispersion_model(config, bounds)
        logger.info("Simulation completed")

        C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = result
        logger.info("End gaussian model simulation")

        try:
            response = {
                "status": 200,
                "concentration": C1.tolist() if isinstance(C1, np.ndarray) else C1,
                "x": x.tolist() if isinstance(x, np.ndarray) else x,
                "y": y.tolist() if isinstance(y, np.ndarray) else y,
                "z": z.tolist() if isinstance(z, np.ndarray) else z,
                "times": times.tolist() if isinstance(times, np.ndarray) else times,
                "stability": (
                    stability.tolist()
                    if isinstance(stability, np.ndarray)
                    else str(stability)
                ),
                "wind_dir": (
                    wind_dir.tolist() if isinstance(wind_dir, np.ndarray) else wind_dir
                ),
                "stab_label": str(stab_label),
                "wind_label": str(wind_label),
                "puff": (
                    puff.tolist()
                    if isinstance(puff, np.ndarray)
                    else (str(puff) if puff is not None else None)
                ),
            }

            for k, v in response.items():
                logger.info(f"{k}: {type(v)}")

        except Exception as e:
            logger.exception("Error in data conversion")
            raise e

        logger.info(f"Simulation completed: returning")
        return response

    except Exception as error:
        logger.exception("Error during simulation")
        raise error


"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)"""
