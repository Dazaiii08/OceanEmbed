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
    requested_date = np.datetime64(date)

    # SST
    with xr.open_dataset(SST_FILE) as ds:
        sst = _get_variable(ds, "analysed_sst")
        sst_day = sst.sel(time=requested_date)
        sst_regridded = _regrid_to_model_grid(sst_day)
        sst_array = np.asarray(
            sst_regridded.values,
            dtype=np.float32
        )

    # SSH
    with xr.open_dataset(SSH_FILE) as ds:
        ssh = _get_variable(ds, "sla")
        ssh_day = ssh.sel(time=requested_date)
        ssh_regridded = _regrid_to_model_grid(ssh_day)
        ssh_array = np.asarray(
            ssh_regridded.values,
            dtype=np.float32
        )

    # SSS
    with xr.open_dataset(SSS_FILE) as ds:
        sss = _get_variable(ds, "sss")
        sss_times = sss.time.values

        if requested_date < sss_times[0]:
            sss_day = xr.full_like(
                sss.isel(time=0, drop=True),
                np.nan
            )
        else:
            sss_day = sss.sel(
                time=requested_date,
                method="ffill"
            )

        sss_regridded = _regrid_to_model_grid(sss_day)
        sss_array = np.asarray(
            sss_regridded.values,
            dtype=np.float32
        )

    expected_shape = (76, 76)

    if sst_array.shape != expected_shape:
        raise ValueError(
            f"SST shape mismatch: expected {expected_shape}, got {sst_array.shape}"
        )

    if ssh_array.shape != expected_shape:
        raise ValueError(
            f"SSH shape mismatch: expected {expected_shape}, got {ssh_array.shape}"
        )

    if sss_array.shape != expected_shape:
        raise ValueError(
            f"SSS shape mismatch: expected {expected_shape}, got {sss_array.shape}"
        )

    raw_X = np.stack(
        [
            sst_array,
            ssh_array,
            sss_array
        ],
        axis=-1
    )

    raw_X = np.where(
        OCEAN_MASK[:, :, None],
        raw_X,
        np.nan
    )

    raw_X = np.nan_to_num(
        raw_X,
        nan=0.0
    )

    model_input = raw_X.astype(
        np.float32
    )

    expected_final_shape = (76, 76, 3)

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