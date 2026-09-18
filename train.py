import os

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
# CHECKPOINT DIRECTORY
# ============================================================

# Create a checkpoints folder next to the model file

MODEL_DIR = os.path.dirname(
    MODEL_PATH
)

CHECKPOINT_DIR = os.path.join(
    MODEL_DIR,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


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
    # Training history
    # --------------------------------------------------------

    train_losses = []


    # Best training loss

    best_loss = float("inf")


    # ========================================================
    # START TRAINING
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "==============================\n"
    )


    for epoch in range(EPOCHS):


        model.train()


        running_loss = 0.0


        # ----------------------------------------------------
        # Train one epoch
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Calculate epoch loss
        # ----------------------------------------------------

        epoch_loss = (
            running_loss /
            len(train_loader)
        )


        train_losses.append(
            epoch_loss
        )


        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"[{epoch + 1}/{EPOCHS}] "
            f"Loss: "
            f"{epoch_loss:.4f}"
        )


        # ====================================================
        # SAVE MODEL AFTER EVERY EPOCH
        # ====================================================

        epoch_model_path = os.path.join(
            CHECKPOINT_DIR,
            f"epoch_{epoch + 1}.pth"
        )


        torch.save(
            model.state_dict(),
            epoch_model_path
        )


        print(
            f"Saved: epoch_{epoch + 1}.pth"
        )


        # ====================================================
        # SAVE LATEST MODEL
        # ====================================================

        latest_model_path = os.path.join(
            CHECKPOINT_DIR,
            "latest_model.pth"
        )


        torch.save(
            model.state_dict(),
            latest_model_path
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        # Here "best" means lowest TRAINING LOSS.
        #
        # We are not using validation loss yet because
        # this training script currently has no validation
        # loop.

        if epoch_loss < best_loss:


            best_loss = epoch_loss


            best_model_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )


            torch.save(
                model.state_dict(),
                best_model_path
            )


            print(
                "New best model saved!"
            )


        # ----------------------------------------------------
        # Safety message
        # ----------------------------------------------------

        print(
            f"Checkpoint saved successfully "
            f"after epoch {epoch + 1}.\n"
        )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_model_path = os.path.join(
        CHECKPOINT_DIR,
        "final_model.pth"
    )


    torch.save(
        model.state_dict(),
        final_model_path
    )


    # Also save to MODEL_PATH

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

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
        "\nModels saved in:"
    )

    print(
        CHECKPOINT_DIR
    )


    print(
        "\nAvailable models:"
    )


    print(
        "  epoch_1.pth"
    )


    if EPOCHS >= 2:

        print(
            "  epoch_2.pth"
        )


    if EPOCHS >= 3:

        print(
            "  epoch_3.pth"
        )


    if EPOCHS >= 4:

        print(
            "  epoch_4.pth"
        )


    if EPOCHS >= 5:

        print(
            "  epoch_5.pth"
        )


    print(
        "  latest_model.pth"
    )

    print(
        "  best_model.pth"
    )

    print(
        "  final_model.pth"
    )


    print(
        "\nMain model saved at:"
    )

    print(
        MODEL_PATH
    )


    return model, train_losses