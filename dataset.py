import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from config import (
    IMAGE_SIZE,
    VALID_EXTENSIONS
)


# ============================================================
# GET FILES
# ============================================================

def get_files(folder):

    files = []

    for file in os.listdir(folder):

        if file.lower().endswith(VALID_EXTENSIONS):

            files.append(file)

    files.sort()

    return files


# ============================================================
# CREATE IMAGE-MASK PAIRS
# ============================================================

def create_pairs(image_dir, mask_dir):

    image_files = get_files(image_dir)

    mask_files = get_files(mask_dir)


    # Dictionary containing masks

    mask_dictionary = {}

    for mask in mask_files:

        name = os.path.splitext(mask)[0]

        mask_dictionary[name] = mask


    pairs = []


    # Try matching filenames

    for image in image_files:

        image_name = os.path.splitext(image)[0]


        if image_name in mask_dictionary:

            image_path = os.path.join(
                image_dir,
                image
            )

            mask_path = os.path.join(
                mask_dir,
                mask_dictionary[image_name]
            )

            pairs.append(
                (image_path, mask_path)
            )


    # --------------------------------------------------------
    # If filenames don't match
    # --------------------------------------------------------

    if len(pairs) == 0:

        print(
            "\nWarning: Image and mask names do not match."
        )

        print(
            "Using sorted-order pairing."
        )


        if len(image_files) != len(mask_files):

            raise Exception(
                "Number of images and masks are different."
            )


        pairs = []


        for image, mask in zip(
            image_files,
            mask_files
        ):

            pairs.append(
                (
                    os.path.join(
                        image_dir,
                        image
                    ),

                    os.path.join(
                        mask_dir,
                        mask
                    )
                )
            )


    return pairs


# ============================================================
# CRACK DATASET
# ============================================================

class CrackDataset(Dataset):


    def __init__(self, pairs):

        self.pairs = pairs


    def __len__(self):

        return len(self.pairs)


    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )


        if image is None:

            raise Exception(
                f"Could not read image: {image_path}"
            )


        # BGR → RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )


        if mask is None:

            raise Exception(
                f"Could not read mask: {mask_path}"
            )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )


        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )


        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = image.astype(
            np.float32
        ) / 255.0


        # ----------------------------------------------------
        # Convert mask to binary
        # ----------------------------------------------------

        mask = mask.astype(
            np.float32
        ) / 255.0


        mask = (
            mask > 0.5
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # HWC → CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )


        # ----------------------------------------------------
        # Add channel to mask
        # ----------------------------------------------------

        mask = np.expand_dims(
            mask,
            axis=0
        )


        # ----------------------------------------------------
        # NumPy → PyTorch
        # ----------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )


        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )


        return image, mask