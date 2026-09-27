from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

SST_FILE = PROJECT_DIR / "sst_train.nc"
SSH_FILE = PROJECT_DIR / "ssh_train.nc"
SSS_FILE = PROJECT_DIR / "sss_train.nc"

OCEAN_MASK_FILE = PROJECT_DIR / "ocean_mask.npy"
X_VAL_FILE = PROJECT_DIR / "X_val.npy"


# ============================================================
# EXACT MODEL GRID
# ============================================================

LAT_GRID = np.arange(5.5, 24.26, 0.25)
LON_GRID = np.arange(80.5, 99.26, 0.25)

LAT_SIZE = len(LAT_GRID)
LON_SIZE = len(LON_GRID)


# ============================================================
# LOAD SAVED TRAINING OCEAN MASK
# ============================================================

OCEAN_MASK = np.load(
    OCEAN_MASK_FILE
)

if OCEAN_MASK.shape != (76, 76):
    raise ValueError(
        "Ocean mask shape mismatch: "
        f"expected (76, 76), got {OCEAN_MASK.shape}"
    )

OCEAN_MASK = OCEAN_MASK.astype(bool)


# ============================================================
# HELPERS
# ============================================================

def _get_variable(ds, name):
    """
    Return a requested variable from an xarray Dataset.
    """

    if name in ds:
        return ds[name]

    raise KeyError(
        f"Variable '{name}' not found. "
        f"Available variables: {list(ds.data_vars)}"
    )


def _get_coord_name(da, candidates):
    """
    Find latitude / longitude coordinate names.
    """

    for name in candidates:
        if name in da.coords or name in da.dims:
            return name

    raise KeyError(
        f"Could not find coordinate among {candidates}. "
        f"Dims={da.dims}, coords={list(da.coords)}"
    )


def _regrid_to_model_grid(da):
    """
    Interpolate a DataArray onto the exact 76 x 76
    OceanEmbed model grid.
    """

    lat_name = _get_coord_name(
        da,
        ["latitude", "lat"]
    )

    lon_name = _get_coord_name(
        da,
        ["longitude", "lon"]
    )

    return da.interp(
        {
            lat_name: LAT_GRID,
            lon_name: LON_GRID
        },
        method="linear"
    )


# ============================================================
# LOAD DATASETS
# ============================================================

def load_input_datasets():
    """
    Load the saved 2020 SST, SSH and SSS datasets.
    """

    sst_ds = xr.open_dataset(
        SST_FILE
    )

    ssh_ds = xr.open_dataset(
        SSH_FILE
    )

    sss_ds = xr.open_dataset(
        SSS_FILE
    )

    sst = _get_variable(
        sst_ds,
        "analysed_sst"
    )

    ssh = _get_variable(
        ssh_ds,
        "sla"
    )

    sss = _get_variable(
        sss_ds,
        "sss"
    )

    return sst, ssh, sss


# ============================================================
# AVAILABLE DATES
# ============================================================

def get_available_dates():
    """
    Return all dates available in the SST dataset.
    """

    sst, _, _ = load_input_datasets()

    return [
        str(
            np.datetime_as_string(
                date,
                unit="D"
            )
        )
        for date in sst.time.values
    ]


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(date):
    """
    Reproduce the trained model's input preparation.

    Pipeline:

        SST / SSH / SSS
              |
              v
        spatial regridding
              |
              v
        SSS forward-fill to SST daily timeline
              |
              v
        select requested date
              |
              v
        stack 3 channels
              |
              v
        apply saved ocean mask
              |
              v
        NaN -> 0
              |
              v
        float32 model input

    Returns
    -------
    model_input : np.ndarray
        Shape (76, 76, 3)

    raw_X : np.ndarray
        Shape (76, 76, 3)
    """

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    sst, ssh, sss = load_input_datasets()

    requested_date = np.datetime64(
        date
    )

    # --------------------------------------------------------
    # SPATIAL REGRIDDING
    # --------------------------------------------------------

    sst_regridded = _regrid_to_model_grid(
        sst
    )

    ssh_regridded = _regrid_to_model_grid(
        ssh
    )

    sss_regridded = _regrid_to_model_grid(
        sss
    )

    # --------------------------------------------------------
    # SSS DAILY ALIGNMENT
    #
    # SSS was originally weekly.
    # It was forward-filled onto the SST daily timeline.
    # --------------------------------------------------------

    sss_daily = sss_regridded.reindex(
        time=sst_regridded.time,
        method="ffill"
    )

    # --------------------------------------------------------
    # SELECT REQUESTED DATE
    # --------------------------------------------------------

    try:
        sst_day = sst_regridded.sel(
            time=requested_date
        )
    except KeyError:
        raise ValueError(
            f"Date {date} is not available in SST dataset."
        )

    try:
        ssh_day = ssh_regridded.sel(
            time=requested_date
        )
    except KeyError:
        raise ValueError(
            f"Date {date} is not available in SSH dataset."
        )

    try:
        sss_day = sss_daily.sel(
            time=requested_date
        )
    except KeyError:
        raise ValueError(
            f"Date {date} is not available after SSS alignment."
        )

    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    sst_array = np.squeeze(
        sst_day.values
    )

    ssh_array = np.squeeze(
        ssh_day.values
    )

    sss_array = np.squeeze(
        sss_day.values
    )

    # --------------------------------------------------------
    # SHAPE CHECK
    # --------------------------------------------------------

    expected_shape = (
        76,
        76
    )

    if sst_array.shape != expected_shape:
        raise ValueError(
            f"SST shape mismatch: "
            f"expected {expected_shape}, "
            f"got {sst_array.shape}"
        )

    if ssh_array.shape != expected_shape:
        raise ValueError(
            f"SSH shape mismatch: "
            f"expected {expected_shape}, "
            f"got {ssh_array.shape}"
        )

    if sss_array.shape != expected_shape:
        raise ValueError(
            f"SSS shape mismatch: "
            f"expected {expected_shape}, "
            f"got {sss_array.shape}"
        )

    # --------------------------------------------------------
    # STACK CHANNELS
    # --------------------------------------------------------

    raw_X = np.stack(
        [
            sst_array,
            ssh_array,
            sss_array
        ],
        axis=-1
    )

    # --------------------------------------------------------
    # APPLY EXACT TRAINING OCEAN MASK
    #
    # The saved model dataset masked land / invalid cells
    # before converting NaNs to zero.
    #
    # Broadcasting:
    #
    #   ocean_mask: (76, 76)
    #   raw_X:      (76, 76, 3)
    #
    # --------------------------------------------------------

    raw_X = np.where(
        OCEAN_MASK[:, :, None],
        raw_X,
        np.nan
    )

    # --------------------------------------------------------
    # SAME NaN HANDLING USED FOR TRAINING INPUT ARRAYS
    # --------------------------------------------------------

    raw_X = np.nan_to_num(
        raw_X,
        nan=0.0
    )

    # --------------------------------------------------------
    # MODEL INPUT
    #
    # IMPORTANT:
    # No normalization is applied.
    # The trained checkpoint expects the raw input scale.
    # --------------------------------------------------------

    model_input = raw_X.astype(
        np.float32
    )

    # --------------------------------------------------------
    # FINAL SHAPE CHECK
    # --------------------------------------------------------

    expected_final_shape = (
        76,
        76,
        3
    )

    if model_input.shape != expected_final_shape:
        raise ValueError(
            "Final model input shape mismatch: "
            f"expected {expected_final_shape}, "
            f"got {model_input.shape}"
        )

    return model_input, raw_X


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "OceanEmbed preprocessing test"
    )

    print(
        "--------------------------------"
    )

    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    print(
        "Project:",
        PROJECT_DIR
    )

    # --------------------------------------------------------
    # MODEL GRID
    # --------------------------------------------------------

    print(
        "\nModel grid:"
    )

    print(
        "Latitude :",
        LAT_GRID[0],
        "to",
        LAT_GRID[-1]
    )

    print(
        "Longitude:",
        LON_GRID[0],
        "to",
        LON_GRID[-1]
    )

    print(
        "Grid size:",
        LAT_SIZE,
        "x",
        LON_SIZE
    )

    # --------------------------------------------------------
    # OCEAN MASK
    # --------------------------------------------------------

    print(
        "\nOcean mask:"
    )

    print(
        "Shape:",
        OCEAN_MASK.shape
    )

    print(
        "Ocean cells:",
        int(OCEAN_MASK.sum())
    )

    print(
        "Total cells:",
        OCEAN_MASK.size
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    dates = get_available_dates()

    print(
        "\nAvailable dates:"
    )

    print(
        "First:",
        dates[0]
    )

    print(
        "Last :",
        dates[-1]
    )

    print(
        "Count:",
        len(dates)
    )

    # --------------------------------------------------------
    # VALIDATION SAMPLE
    #
    # 145 training samples
    # means X_val[0] = date index 145.
    # --------------------------------------------------------

    validation_index = 145

    test_date = dates[
        validation_index
    ]

    print(
        "\nTesting known validation date:",
        test_date
    )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    X, raw_X = prepare_input(
        test_date
    )

    print(
        "\nShapes:"
    )

    print(
        "Model input shape:",
        X.shape
    )

    print(
        "Raw input shape  :",
        raw_X.shape
    )

    # --------------------------------------------------------
    # RAW STATISTICS
    # --------------------------------------------------------

    channel_names = [
        "SST",
        "SSH",
        "SSS"
    ]

    print(
        "\nRaw channel ranges:"
    )

    for i, name in enumerate(
        channel_names
    ):

        channel = X[:, :, i]

        print(
            f"{name}: "
            f"min={channel.min():.4f}, "
            f"max={channel.max():.4f}, "
            f"mean={channel.mean():.4f}, "
            f"std={channel.std():.4f}"
        )

    # --------------------------------------------------------
    # LOAD SAVED VALIDATION SAMPLE
    # --------------------------------------------------------

    if X_VAL_FILE.exists():

        X_val = np.load(
            X_VAL_FILE
        )

        reference = np.nan_to_num(
            X_val[0],
            nan=0.0
        )

        print(
            "\nComparison against saved X_val[0]:"
        )

        print(
            "Reference date:",
            test_date
        )

        print(
            "Reference shape:",
            reference.shape
        )

        # ----------------------------------------------------
        # DIFFERENCE
        # ----------------------------------------------------

        diff = np.abs(
            X - reference
        )

        print(
            "\nOverall difference:"
        )

        print(
            "Max absolute difference:",
            f"{diff.max():.8f}"
        )

        print(
            "Mean absolute difference:",
            f"{diff.mean():.8f}"
        )

        # ----------------------------------------------------
        # DIFFERENCE BY CHANNEL
        # ----------------------------------------------------

        print(
            "\nDifference by channel:"
        )

        for i, name in enumerate(
            channel_names
        ):

            channel_diff = diff[:, :, i]

            print(
                f"{name}: "
                f"max={channel_diff.max():.8f}, "
                f"mean={channel_diff.mean():.8f}, "
                f"nonzero={(channel_diff > 1e-6).sum()}"
            )

        # ----------------------------------------------------
        # LARGEST DIFFERENCE
        # ----------------------------------------------------

        max_index = np.unravel_index(
            np.argmax(diff),
            diff.shape
        )

        lat_idx = max_index[0]
        lon_idx = max_index[1]
        channel_idx = max_index[2]

        print(
            "\nLargest mismatch:"
        )

        print(
            "Latitude :",
            LAT_GRID[lat_idx]
        )

        print(
            "Longitude:",
            LON_GRID[lon_idx]
        )

        print(
            "Channel  :",
            channel_names[channel_idx]
        )

        print(
            "Backend  :",
            X[max_index]
        )

        print(
            "Reference:",
            reference[max_index]
        )

        print(
            "Difference:",
            diff[max_index]
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if diff.max() < 1e-4:

            print(
                "\nSTATUS: PERFECT MATCH"
            )

            print(
                "Deployment preprocessing "
                "matches X_val[0]."
            )

        elif diff.max() < 1e-2:

            print(
                "\nSTATUS: VERY CLOSE MATCH"
            )

        else:

            print(
                "\nSTATUS: STILL MISMATCHED"
            )

    else:

        print(
            "\nX_val.npy not found."
        )

    print(
        "\nPreprocessing test completed."
    )