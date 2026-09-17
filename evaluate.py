import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    BATCH_SIZE,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    MODEL_PATH
)

from dataset import (
    create_pairs,
    CrackDataset
)

from model import UNet

from metrics import calculate_metrics


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model():


    print(
        "\n=============================="
    )

    print(
        "EVALUATING MODEL"
    )

    print(
        "==============================\n"
    )


    # --------------------------------------------------------
    # Create test pairs
    # --------------------------------------------------------

    test_pairs = create_pairs(
        TEST_IMAGE_DIR,
        TEST_MASK_DIR
    )


    print(
        "Testing pairs:",
        len(test_pairs)
    )


    if len(test_pairs) == 0:

        raise Exception(
            "No testing data found."
        )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = CrackDataset(
        test_pairs
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = UNet()


    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    # --------------------------------------------------------
    # Metric storage
    # --------------------------------------------------------

    total_iou = 0.0

    total_dice = 0.0

    total_precision = 0.0

    total_recall = 0.0

    total_f1 = 0.0


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():


        for images, masks in test_loader:


            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )


            outputs = model(
                images
            )


            results = calculate_metrics(
                outputs,
                masks
            )


            total_iou += results["iou"]

            total_dice += results["dice"]

            total_precision += results[
                "precision"
            ]

            total_recall += results[
                "recall"
            ]

            total_f1 += results["f1"]


    # --------------------------------------------------------
    # Average
    # --------------------------------------------------------

    num_batches = len(
        test_loader
    )


    iou = total_iou / num_batches

    dice = total_dice / num_batches

    precision = (
        total_precision /
        num_batches
    )

    recall = (
        total_recall /
        num_batches
    )

    f1 = (
        total_f1 /
        num_batches
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=============================="
    )


    print(
        f"IoU       : {iou:.4f}"
    )

    print(
        f"Dice      : {dice:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    return model