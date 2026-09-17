import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    MODEL_PATH
)

from dataset import (
    create_pairs,
    CrackDataset
)

from model import UNet


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model():


    print("\n==============================")

    print("LOADING TRAINING DATA")

    print("==============================\n")


    # --------------------------------------------------------
    # Create training pairs
    # --------------------------------------------------------

    train_pairs = create_pairs(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR
    )


    print(
        "Training pairs:",
        len(train_pairs)
    )


    if len(train_pairs) == 0:

        raise Exception(
            "No training data found."
        )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = CrackDataset(
        train_pairs
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet()

    model = model.to(
        DEVICE
    )


    print(
        "\nDevice:",
        DEVICE
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_losses = []


    print(
        "\n=============================="
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "==============================\n"
    )


    for epoch in range(
        EPOCHS
    ):


        model.train()


        running_loss = 0.0


        for images, masks in train_loader:


            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )


            # Clear gradients

            optimizer.zero_grad()


            # Forward pass

            outputs = model(
                images
            )


            # Calculate loss

            loss = criterion(
                outputs,
                masks
            )


            # Backpropagation

            loss.backward()


            # Update weights

            optimizer.step()


            running_loss += (
                loss.item()
            )


        epoch_loss = (
            running_loss /
            len(train_loader)
        )


        train_losses.append(
            epoch_loss
        )


        print(
            f"Epoch "
            f"[{epoch + 1}/{EPOCHS}] "
            f"Loss: "
            f"{epoch_loss:.4f}"
        )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )


    print(
        "\n=============================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=============================="
    )


    print(
        "\nModel saved at:"
    )

    print(
        MODEL_PATH
    )


    return model, train_losses