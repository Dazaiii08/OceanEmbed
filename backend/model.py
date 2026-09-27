from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------
# OceanEmbed CNN
# ---------------------------------------------------------

class OceanCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=14):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        # 32 feature maps → 14 depth levels
        self.out = nn.Conv2d(
            32, out_channels, 1
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(
            torch.cat([u2, e2], dim=1)
        )

        u1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat([u1, e1], dim=1)
        )

        return self.out(d1)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

MODEL_PATH = PROJECT_DIR / "baseline_maskfix_v1.pth"


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

model = OceanCNN(
    in_channels=3,
    out_channels=14
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

DEPTHS = np.array([
    0.5,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200,
    300,
    500,
    700
])


def predict(input_data):
    """
    Run OceanEmbed inference.

    Expected input:
        numpy array with shape (76, 76, 3)

    Channel order:
        0 = SST
        1 = SSH/SLA
        2 = SSS

    Returns:
        numpy array with shape (14, 76, 76)
    """

    if input_data.shape != (76, 76, 3):
        raise ValueError(
            f"Expected input shape (76, 76, 3), "
            f"got {input_data.shape}"
        )

    # Convert HWC → CHW
    tensor = torch.tensor(
        input_data,
        dtype=torch.float32
    ).permute(2, 0, 1)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        prediction = model(tensor)

    # Remove batch dimension
    prediction = prediction.squeeze(0)

    # Move back to CPU / NumPy
    prediction = prediction.cpu().numpy()

    return prediction


if __name__ == "__main__":
    print("OceanEmbed model loaded successfully.")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Input shape: (76, 76, 3)")
    print(f"Output shape: (14, 76, 76)")
    print(f"Depths: {DEPTHS}")