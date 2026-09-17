import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    BATCH_SIZE
)

from dataset import (
    create_pairs,
    CrackDataset
)


# ============================================================
# VISUALIZE PREDICTION
# ============================================================

def visualize_predictions(model):


    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_pairs = create_pairs(
        TEST_IMAGE_DIR,
        TEST_MASK_DIR
    )


    test_dataset = CrackDataset(
        test_pairs
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    images, masks = next(
        iter(test_loader)
    )


    images = images.to(
        DEVICE
    )

    masks = masks.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        outputs = model(
            images
        )


        predictions = torch.sigmoid(
            outputs
        )


    # --------------------------------------------------------
    # First image
    # --------------------------------------------------------

    image = images[0].cpu().numpy()


    image = np.transpose(
        image,
        (1, 2, 0)
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = (
        masks[0][0]
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = (
        predictions[0][0]
        .cpu()
        .numpy()
    )


    prediction = (
        prediction > 0.5
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 5)
    )


    # Original

    plt.subplot(
        1,
        3,
        1
    )


    plt.imshow(
        image
    )


    plt.title(
        "Original Image"
    )


    plt.axis(
        "off"
    )


    # Ground truth

    plt.subplot(
        1,
        3,
        2
    )


    plt.imshow(
        ground_truth,
        cmap="gray"
    )


    plt.title(
        "Ground Truth"
    )


    plt.axis(
        "off"
    )


    # Prediction

    plt.subplot(
        1,
        3,
        3
    )


    plt.imshow(
        prediction,
        cmap="gray"
    )


    plt.title(
        "Predicted Crack Mask"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    plt.show()