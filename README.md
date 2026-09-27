# OceanEmbed

## Subsurface Ocean Temperature Reconstruction from Sparse Surface Observations

OceanEmbed is a deep-learning-based system for reconstructing subsurface ocean temperature across the Bay of Bengal using surface-derived ocean observations.

The system uses three surface variables:

- Sea Surface Temperature (SST)
- Sea Surface Height (SSH)
- Sea Surface Salinity (SSS)

A convolutional encoder-decoder model maps these surface observations to temperature fields across multiple subsurface depths.

The current V1 implementation operates on historical data from **January 2020 through June 2020** and produces temperature reconstructions on a **76 × 76 spatial grid** across **14 depth levels**, extending from **0.5 m to 700 m**.

---

## Overview

Direct subsurface ocean observations are sparse across both space and depth. This makes it difficult to obtain continuous regional views of the upper and deep ocean thermal structure.

OceanEmbed investigates whether the information contained in surface ocean observations can be used to reconstruct the subsurface temperature structure over a regional ocean domain.

### Core Idea

```text
Surface Ocean Observations
        │
        ├── SST
        ├── SSH
        └── SSS
        │
        ▼
Spatial Preprocessing
        │
        ▼
Ocean-Domain Masking
        │
        ▼
CNN Encoder-Decoder
        │
        ▼
Subsurface Temperature Reconstruction
        │
        ▼
Interactive Temperature Maps
```

---

## Key Features

- Deep-learning-based subsurface ocean temperature reconstruction
- Uses SST, SSH and SSS as surface predictors
- Bay of Bengal regional domain
- 0.25° spatial resolution
- 76 × 76 reconstruction grid
- 14 subsurface depth levels
- Reconstruction from 0.5 m to 700 m
- Historical date-based inference
- Ocean-domain masking
- Ocean-only field statistics
- Interactive depth selection
- Interactive temperature-field visualization
- Dynamic temperature color scale
- FastAPI inference backend
- React + Vite frontend
- Prediction caching
- GLORYS holdout validation
- Independent CORA validation

---

## Problem Statement

Subsurface ocean temperature plays an important role in understanding ocean circulation, stratification, heat transport and the structure of the upper ocean.

However, direct subsurface measurements are much sparser than surface observations.

OceanEmbed explores a machine-learning approach where commonly available surface-derived measurements are used to reconstruct a temperature field throughout the subsurface ocean.

The goal is not to replace direct measurements, but to investigate whether surface observations contain enough information to produce useful regional estimates of subsurface thermal structure.

---

## Model Input

OceanEmbed takes three surface-derived ocean variables as model input.

| Input | Description |
|---|---|
| SST | Sea Surface Temperature |
| SSH | Sea Surface Height |
| SSS | Sea Surface Salinity |

### Input Tensor

The model input shape is:

```text
76 × 76 × 3
```

where:

```text
76 × 76 → spatial grid
3       → SST + SSH + SSS channels
```

The deployed inference pipeline converts the available surface datasets to the model grid, aligns the temporal observations, applies the trained ocean-domain mask, and prepares the resulting tensor for CNN inference.

---

## Model Output

The model reconstructs subsurface ocean temperature at 14 depth levels.

### Depth Levels
0.5 m
5 m
10 m
20 m
30 m
50 m
75 m
100 m
125 m
150 m
200 m
300 m
500 m
700 m


### Output Tensor

```text
76 × 76 × 14
```

where:

```text
76 × 76 → spatial grid
14      → reconstructed depth levels
```

Each output layer represents a temperature field for one selected depth.

---

## Spatial Domain

OceanEmbed currently operates over the Bay of Bengal.

```text
Latitude:     5.5°N  → 24.25°N
Longitude:   80.5°E → 99.25°E
Resolution:   0.25°
Grid:         76 × 76
```

### Domain Summary

| Parameter | Value |
|---|---|
| Region | Bay of Bengal |
| Latitude | 5.5°N – 24.25°N |
| Longitude | 80.5°E – 99.25°E |
| Grid spacing | 0.25° |
| Grid size | 76 × 76 |
| Number of depth levels | 14 |
| Maximum reconstruction depth | 700 m |

---

## Reconstruction Pipeline

```text
                 ┌───────────────┐
                 │      SST      │
                 └───────┬───────┘
                         │
                 ┌───────▼───────┐
                 │      SSH      │
                 └───────┬───────┘
                         │
                 ┌───────▼───────┐
                 │      SSS      │
                 └───────┬───────┘
                         │
                         ▼
                Spatial Regridding
                         │
                         ▼
              Temporal Alignment
                         │
                         ▼
               Ocean-Domain Mask
                         │
                         ▼
                CNN Encoder
                         │
                         ▼
                   Bottleneck
                         │
                         ▼
                CNN Decoder
                         │
                         ▼
          76 × 76 × 14 Temperature
                         │
                         ▼
              Interactive Dashboard
```

---

## Data

The current V1 system uses historical ocean data for the Bay of Bengal.

### Temporal Coverage

```text
January 2020 – June 2020
```

The current implementation is therefore a historical reconstruction system, not a real-time operational product.

### Surface Predictors

The model uses three surface-derived variables:

**Sea Surface Temperature**
SST provides information about the thermal state of the ocean surface.

**Sea Surface Height**
SSH provides information related to large-scale ocean dynamic structure and circulation.

**Sea Surface Salinity**
SSS provides additional information about surface hydrographic conditions.

### Subsurface Target

The primary reconstruction target is derived from the GLORYS ocean product.

The target contains temperature information across the model domain and depth levels.

### External Validation

CORA objective-analysis fields are used as an external validation product.

CORA is treated here as an observationally derived gridded objective-analysis dataset, rather than direct validation against individual raw ARGO profiles.

---

## Data Preprocessing

Before inference, the surface datasets are transformed into the representation expected by the trained CNN.

The preprocessing pipeline performs the following steps:

1. Load SST, SSH and SSS datasets
2. Regrid the variables onto the 0.25° OceanEmbed spatial grid
3. Align SSS to the daily SST timeline
4. Select the requested historical date
5. Stack SST, SSH and SSS into a three-channel input
6. Apply the ocean-domain mask
7. Handle masked/non-ocean cells in the same way as the training pipeline
8. Produce the final 76 × 76 × 3 tensor

**Important:** The deployed preprocessing was explicitly verified against the saved validation inputs used during model development. The current trained checkpoint expects the raw physical input representation produced by this preprocessing pipeline (no standardization/scaling is applied at inference time).

---

## Ocean-Domain Masking

OceanEmbed uses a spatial ocean mask to distinguish valid ocean cells from land/non-ocean cells.

The mask is:

```text
76 × 76
```

and is used to:

- Prevent land cells from being treated as valid ocean observations
- Match the training data representation
- Calculate ocean-only statistics
- Prevent masked cells from affecting reported minimum, maximum and mean values
- Visually distinguish non-ocean cells in the dashboard

Because the reconstruction grid is relatively coarse at 0.25°, the coastline representation is also coarse.

This can result in isolated dark cells or small irregular masked regions near coastal boundaries.

These cells are masked domain cells, not model prediction failures.

---

## Model Architecture

OceanEmbed uses a convolutional encoder-decoder architecture implemented in PyTorch.

The architecture follows an encoder → bottleneck → decoder structure with skip connections.

```text
Input
76 × 76 × 3
      │
      ▼
┌────────────────────┐
│ Encoder Block 1    │
│ 3 → 32 channels    │
│ 3×3 convolutions   │
└─────────┬──────────┘
          │
       MaxPool
          │
          ▼
┌────────────────────┐
│ Encoder Block 2    │
│ 32 → 64 channels   │
│ 3×3 convolutions   │
└─────────┬──────────┘
          │
       MaxPool
          │
          ▼
┌────────────────────┐
│ Bottleneck         │
│ 64 → 128 channels  │
│ 3×3 convolutions   │
└─────────┬──────────┘
          │
     Transposed Conv
          │
          ▼
┌────────────────────┐
│ Decoder Block 2    │
│ 128 → 64 channels  │
│ Skip connection    │
└─────────┬──────────┘
          │
     Transposed Conv
          │
          ▼
┌────────────────────┐
│ Decoder Block 1    │
│ 64 → 32 channels   │
│ Skip connection    │
└─────────┬──────────┘
          │
          ▼
     1×1 Output Conv
          │
          ▼
Output
76 × 76 × 14
```

The model contains approximately 467K trainable parameters.

---

## Training

The current V1 model was trained using a masked reconstruction objective.

A key aspect of the training pipeline is that invalid/non-ocean target cells are not treated as real zero-temperature observations.

Instead, valid target cells are used in the loss calculation through masking.

Conceptually:

```text
Prediction
    │
    ▼
Compare against valid target cells
    │
    ▼
Ignore invalid/non-ocean cells
    │
    ▼
Masked MSE Loss
```

This prevents the model from being optimized toward physically meaningless zero-temperature targets over masked regions.

---

## Validation

OceanEmbed has been evaluated using both a GLORYS holdout evaluation and an external CORA validation.

### GLORYS Holdout Validation

The final masked-loss model achieved:

| Metric | Result |
|---|---|
| RMSE | 1.420 °C |
| Climatology baseline RMSE | 1.619 °C |
| RMSE reduction vs baseline | ~12.3% |

The model therefore improves upon the climatology baseline in the current evaluation setup.

### Depth-wise GLORYS Performance

The model's performance varies with depth.

| Depth | RMSE (°C) | MAE (°C) |
|---|---|---|
| 0.5 m | 0.872 | 0.672 |
| 5 m | 0.825 | 0.629 |
| 10 m | 0.606 | 0.453 |
| 20 m | 0.609 | 0.431 |
| 30 m | 0.747 | 0.498 |
| 50 m | 1.499 | 1.182 |
| 75 m | 2.103 | 1.663 |
| 100 m | 2.433 | 1.994 |
| 125 m | 2.522 | 2.078 |
| 150 m | 2.047 | 1.674 |
| 200 m | 1.257 | 0.988 |
| 300 m | 0.766 | 0.609 |
| 500 m | 0.458 | 0.374 |
| 700 m | 0.843 | 0.670 |

The results show that reconstruction quality is depth-dependent.

In particular, spatial fine-scale skill is more limited in parts of the thermocline region, approximately around 75–150 m, where the predictions become more spatially smoothed and under-dispersed.

This is an important limitation of the current V1 model.

### Independent CORA Validation

CORA objective-analysis fields were used as an independent external validation product.

**May 2020**

| Metric | Result |
|---|---|
| RMSE | 0.975 °C |
| MAE | 0.708 °C |
| Pooled correlation | 0.994 |

**June 2020**

| Metric | Result |
|---|---|
| RMSE | 1.218 °C |
| MAE | 0.845 °C |
| Pooled correlation | 0.988 |

These results provide an external validation check beyond the primary GLORYS holdout evaluation.

Note: pooled correlation combines all depths together and is partly driven by the shared vertical temperature gradient; it should not be read as evidence of strong horizontal spatial skill at any single depth (see per-depth analysis in project notebooks).

The CORA results should be interpreted in the context of the current V1 evaluation period and dataset coverage.

---

## Frontend

The OceanEmbed dashboard is built using:

- React
- Vite
- CSS

The frontend provides a visual interface for historical reconstruction and depth exploration.

### Dashboard Features

**Historical Date Selection**
Users can select a date from the available historical dataset.

**Reconstruction**
Clicking Reconstruct sends the selected date to the FastAPI backend and runs the trained model.

**Depth Selection**
The interface provides:
- Depth slider
- Depth selection buttons
- Interactive switching between reconstructed depth levels

**Temperature Field**
The reconstructed temperature field is displayed as a 76 × 76 grid.

**Temperature Legend**
The dashboard dynamically displays a temperature color scale based on valid ocean cells.

**Field Statistics**
The interface reports:
- Mean temperature
- Minimum temperature
- Maximum temperature

These statistics are calculated over valid ocean cells rather than masked land cells.

**Reconstruction Summary**
The dashboard also communicates the current reconstruction depth and the three-input CNN setup.

---

## Backend

The inference backend is built using:

- Python
- FastAPI
- PyTorch
- NumPy
- xarray
- Uvicorn

The backend handles:

- Available-date discovery
- Input preparation
- Model inference
- Prediction caching
- Temperature-map retrieval
- Ocean-only statistics
- Frontend API requests

---

## API

### `GET /`

Returns the API status.

```json
{
  "project": "OceanEmbed",
  "status": "running",
  "message": "OceanEmbed inference API is online."
}
```

### `GET /dates`

Returns all historical dates available to the application.

GET /dates

Response:

```json
{
  "dates": [
    "2020-01-01",
    "2020-01-02",
    "2020-01-03"
  ]
}
```

### `POST /predict`

Runs model inference for a selected historical date.

Example request:

```json
{
  "date": "2020-05-25"
}
```

The response includes:

- Selected date
- Spatial grid information
- Available reconstructed depths
- Mean temperature
- Minimum temperature
- Maximum temperature for each depth

### `GET /map`

Returns the reconstructed temperature field for a specific date and depth.

The response contains:

- Date
- Depth
- Grid metadata
- Ocean-only minimum temperature
- Ocean-only maximum temperature
- Ocean-only mean temperature
- 76 × 76 temperature map
- 76 × 76 ocean mask

### API Architecture

```text
React Frontend
      │
      │ HTTP
      ▼
FastAPI Backend
      │
      ├── /dates
      ├── /predict
      └── /map
      │
      ▼
Preprocessing
      │
      ▼
OceanEmbed CNN
      │
      ▼
Temperature Reconstruction
```

---

## Project Structure

```text
OceanEmbed/
│
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   └── ...
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── ...
│   ├── package.json
│   └── ...
│
├── baseline_maskfix_v1.pth
│
├── glorys_train.nc
├── sst_train.nc
├── ssh_train.nc
├── sss_train.nc
│
├── ocean_mask.npy
├── ocean_mask_bool.npy
│
├── X_train.npy
├── X_val.npy
├── y_train.npy
├── y_val.npy
│
├── val_predictions_maskfix_v1.npy
│
├── model_training.ipynb
├── data_pipeline.ipynb
├── dataset_exploration.ipynb
├── argo_validation.ipynb
│
├── argo_oa_validation.nc
│
├── cora_scatter_june_20m.png
├── cora_scatter_june_100m.png
├── cora_depthwise_rmse_mae_june.png
├── oceanembed_results_table.csv
│
├── start_backend.bat
├── start_oceanembed.bat
│
└── README.md
```

---

## Running Locally

### Requirements

Recommended environment:

- Python 3.x
- Node.js
- npm
- PyTorch
- FastAPI
- Uvicorn
- NumPy
- xarray
- React
- Vite

### Backend Setup

Open a terminal in the project directory.

```bash
cd backend
```

Start FastAPI:

```bash
python -m uvicorn main:app --reload
```

The backend will be available at: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs


### Frontend Setup

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The dashboard will be available at: http://localhost:5173


### One-Click Startup

The repository also includes: start_oceanembed.bat


This launcher starts:

- FastAPI backend
- React frontend

and opens the local dashboard.

Typical workflow:

```text
Double-click start_oceanembed.bat
        │
        ├── Backend starts on port 8000
        │
        ├── Frontend starts on port 5173
        │
        └── Browser opens localhost:5173
```

---

## Example Usage

A typical OceanEmbed session looks like:
Start the application
Select a historical date
Click "Reconstruct"
Wait for model inference
Select a depth
Inspect the reconstructed temperature field
Review ocean-only field statistics


For example:

```text
Date:  2020-05-25
Depth: 100 m
Output: 76 × 76 temperature field
```

---

## Performance and Caching

Model inference can be more expensive than serving a simple API response because it requires:

- Loading/preparing the selected surface fields
- Running the CNN
- Generating the requested temperature field

The backend therefore maintains an in-memory prediction cache for dates that have already been reconstructed.

This allows subsequent depth requests for the same date to reuse the existing prediction instead of rerunning the CNN.

---

## Limitations

OceanEmbed V1 is a research and engineering prototype.

**1. Limited Temporal Coverage**
The current system is trained and evaluated only on January 2020 – June 2020. Performance outside this period has not been established.

**2. Not a Real-Time System**
The current application operates on historical data. It should not be described as a real-time operational ocean monitoring system.

**3. Regional Scope**
The current model is designed specifically around the selected Bay of Bengal spatial domain. Generalization to other ocean basins has not been demonstrated.

**4. Spatial Resolution**
The model operates on a 0.25° × 0.25° grid. This limits representation of very fine coastal structures and small-scale ocean features.

**5. Thermocline Reconstruction**
Spatial fine-scale reconstruction skill is more limited in parts of the thermocline, particularly around approximately 75–150 m. Predictions in this region can be more spatially smoothed and under-dispersed.

**6. External Validation Scope**
CORA provides an independent observationally derived gridded objective-analysis product for validation. It should not be interpreted as direct validation against individual raw ARGO profiles. Broader validation using additional independent observational datasets remains necessary.

**7. Generalization**
Additional multi-year testing is required before making broader claims about model robustness across different seasons, years, ocean states, and geographic regions.

---

## Future Work

Potential future extensions include:

- Multi-year training
- Multi-season validation
- Broader independent observational validation
- Near-real-time data ingestion
- Higher-resolution regional reconstruction
- Additional oceanographic predictors (winds, currents)
- Atmospheric predictors
- Improved thermocline representation
- Uncertainty estimation
- Ensemble reconstruction
- Probabilistic predictions
- Automated data ingestion
- Cloud deployment
- Public-facing inference API

---

## Scientific Interpretation

OceanEmbed should be interpreted as a machine-learning reconstruction system, not as a replacement for direct oceanographic observations.

The model learns statistical relationships between surface ocean conditions and the subsurface thermal structure represented in the training data.

The usefulness of the resulting field therefore depends on:

- Input data quality
- Training distribution
- Spatial resolution
- Temporal coverage
- Model generalization
- Independent validation

The current V1 results demonstrate the feasibility of the reconstruction approach within the evaluated 2020 period and spatial domain, while also highlighting limitations in fine-scale thermocline reconstruction.

---

## Reproducibility

The project contains the main components required to reproduce the current inference workflow:

```text
Raw / processed ocean datasets
        ↓
Preprocessing
        ↓
Saved trained checkpoint
        ↓
Inference
        ↓
Interactive visualization
```

The current deployed checkpoint is:
baseline_maskfix_v1.pth


The associated preprocessing pipeline is implemented in:

backend/preprocessing.py


The model architecture and checkpoint loading are implemented in:

backend/model.py
The API layer is implemented in:

backend/main.py


The web interface is implemented in:

frontend/src/App.jsx
frontend/src/App.css


---

## Research Artifacts

The repository also contains supporting artifacts from model development and validation, including:

- `model_training.ipynb`
- `data_pipeline.ipynb`
- `dataset_exploration.ipynb`
- `argo_validation.ipynb`

as well as validation outputs and figures:

- `cora_scatter_june_20m.png`
- `cora_scatter_june_100m.png`
- `cora_depthwise_rmse_mae_june.png`
- `oceanembed_results_table.csv`

---

## Current Status

**Functional V1 Research Prototype**

The current implementation provides a complete local pipeline:

```text
Surface Ocean Observations
          ↓
Data Preprocessing
          ↓
Spatial Regridding
          ↓
Ocean-Domain Masking
          ↓
CNN Inference
          ↓
14 Subsurface Temperature Fields
          ↓
FastAPI Backend
          ↓
React Dashboard
```

The current system can reconstruct historical Bay of Bengal temperature fields from January 2020 to June 2020, across 0.5 m to 700 m, using a 76 × 76 spatial grid.

---

## Roadmap

- [x] Data acquisition
- [x] Data preprocessing
- [x] Spatial regridding
- [x] Ocean-domain masking
- [x] CNN model development
- [x] Masked training objective
- [x] Model validation
- [x] CORA external validation
- [x] FastAPI backend
- [x] React dashboard
- [x] Interactive depth selection
- [x] Temperature legend
- [x] Ocean-only statistics
- [x] One-click local startup
- [x] Validation dashboard
- [ ] Multi-year training
- [ ] Broader independent validation
- [ ] Improved thermocline reconstruction
- [ ] Deployment
- [ ] Real-time / near-real-time ingestion

---

## Conclusion

OceanEmbed demonstrates an end-to-end deep-learning workflow for reconstructing subsurface ocean temperature from surface-derived observations.

The system combines:

- Oceanographic data
- Scientific preprocessing
- Convolutional deep learning
- Independent validation
- Interactive visualization

The current V1 implementation provides a functional historical reconstruction prototype for the Bay of Bengal and establishes a foundation for future work on larger temporal coverage, improved spatial resolution, stronger observational validation, and operational inference.

---

## License

Add the appropriate project license here before public release.

## Author

Developed as a machine-learning and oceanographic reconstruction project focused on using surface ocean observations to infer subsurface thermal structure.