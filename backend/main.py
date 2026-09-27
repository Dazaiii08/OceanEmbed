from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from preprocessing import prepare_input, get_available_dates, OCEAN_MASK
from model import predict, DEPTHS


app = FastAPI(
    title="OceanEmbed API",
    description="Subsurface ocean temperature reconstruction API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


prediction_cache = {}
available_dates_cache = get_available_dates()


class PredictionRequest(BaseModel):
    date: str


@app.get("/")
def root():
    return {
        "project": "OceanEmbed",
        "status": "running",
        "message": "OceanEmbed inference API is online."
    }


@app.get("/dates")
def available_dates():
    return {
        "dates": available_dates_cache
    }


def run_inference(date: str):
    if date in prediction_cache:
        return prediction_cache[date]

    try:
        model_input, _ = prepare_input(date)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    try:
        prediction = predict(model_input)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    prediction_cache[date] = prediction

    return prediction


def get_ocean_stats(temperature_map):
    """
    Calculate statistics using ocean cells only.
    Land cells are excluded using the trained ocean mask.
    """
    ocean_values = temperature_map[OCEAN_MASK]

    return {
        "min_temp_c": float(np.min(ocean_values)),
        "max_temp_c": float(np.max(ocean_values)),
        "mean_temp_c": float(np.mean(ocean_values))
    }


@app.post("/predict")
def predict_temperature(request: PredictionRequest):
    prediction = run_inference(request.date)

    depth_results = []

    for depth, temperature_map in zip(DEPTHS, prediction):

        stats = get_ocean_stats(temperature_map)

        depth_results.append({
            "depth_m": float(depth),
            **stats
        })

    return {
        "date": request.date,
        "grid": {
            "latitude_min": 5.5,
            "latitude_max": 24.25,
            "longitude_min": 80.5,
            "longitude_max": 99.25,
            "lat_points": 76,
            "lon_points": 76
        },
        "depths": depth_results
    }


@app.get("/map")
def get_temperature_map(
    date: str = Query(...),
    depth: float = Query(...)
):

    depth_matches = np.where(
        np.isclose(DEPTHS, depth)
    )[0]

    if len(depth_matches) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid depth.",
                "available_depths_m": [
                    float(d) for d in DEPTHS
                ]
            }
        )

    depth_index = int(depth_matches[0])

    prediction = run_inference(date)

    temperature_map = prediction[depth_index]

    stats = get_ocean_stats(temperature_map)

    return {
        "date": date,
        "depth_m": float(DEPTHS[depth_index]),
        "grid": {
            "latitude_min": 5.5,
            "latitude_max": 24.25,
            "longitude_min": 80.5,
            "longitude_max": 99.25,
            "lat_points": 76,
            "lon_points": 76
        },
        **stats,
        "temperature_map": temperature_map.tolist(),
        "ocean_mask": OCEAN_MASK.tolist()
    }