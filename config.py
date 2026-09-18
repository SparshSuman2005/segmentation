import os
import torch


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 1e-4


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


TRAIN_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "train_img"
)

TRAIN_MASK_DIR = os.path.join(
    BASE_DIR,
    "train_lab"
)


TEST_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "test_img"
)

TEST_MASK_DIR = os.path.join(
    BASE_DIR,
    "test_lab"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "unet_crack.pth"
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)