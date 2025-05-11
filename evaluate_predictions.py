"""
Script Name: evaluate_predictions.py

Author: Faiz Chaudhry  
Company: AiLiveSim Ltd.

Purpose:
--------
This script calculates error metrics (MSE, RMSE, MAE) for normalized camera parameters derived from predictions.
The normalization includes:
- Conversion of fx to horizontal field of view (hfov) in radians.
- Conversion of cx and cy to normalized values relative to image width and height.

Parameters Evaluated:
- hfov: Horizontal field of view in radians
- norm_cx: cx as a normalized value (0 to 1) relative to width
- norm_cy: cy as a normalized value (0 to 1) relative to height
- k1, k2, k3, p1, p2: Distortion coefficients

Output:
--------
A CSV file containing the MSE, RMSE, and MAE for each parameter is saved in the specified output directory.

Usage:
-------
python evaluate_predictions.py --csv <path_to_predictions.csv> --width <image_width> --height <image_height> --output_dir <output_directory>
"""

import os
import pandas as pd
import numpy as np
import argparse
from typing import Tuple


def fx_to_hfov(fx: float, width: int) -> float:
    """
    Converts focal length (fx) to horizontal field of view (hfov) in radians.

    Args:
        fx (float): Focal length in pixels.
        width (int): Image width in pixels.

    Returns:
        float: Horizontal field of view in radians.
    """
    return 2 * np.arctan((width / 2) / fx)


def cx_to_norm(cx: float, width: int) -> float:
    """
    Converts cx to a normalized value relative to the image width.

    Args:
        cx (float): Principal point x-coordinate in pixels.
        width (int): Image width in pixels.

    Returns:
        float: cx as a normalized value (0 to 1).
    """
    return cx / width


def cy_to_norm(cy: float, height: int) -> float:
    """
    Converts cy to a normalized value relative to the image height.

    Args:
        cy (float): Principal point y-coordinate in pixels.
        height (int): Image height in pixels.

    Returns:
        float: cy as a normalized value (0 to 1).
    """
    return cy / height


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

    Args:
        gt (np.ndarray): Ground truth values.
        pred (np.ndarray): Predicted values.

    Returns:
        Tuple[float, float, float]: MSE, RMSE, MAE values.
    """
    error = pred - gt
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    return mse, rmse, mae


def run_normalized_metrics(csv_path: str, width: int, height: int, output_dir: str) -> None:
    """
    Reads a prediction CSV, computes normalized error metrics, and saves the results.

    Args:
        csv_path (str): Path to the predictions CSV file.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        output_dir (str): Directory to save the output CSV.

    Returns:
        None
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Normalize fx → hfov, cx → norm, cy → norm
    df["hfov_gt"] = 2 * np.arctan((width / 2) / df["fx_gt"])
    df["hfov_pred"] = 2 * np.arctan((width / 2) / df["fx_pred"])
    df["norm_cx_gt"] = df["cx_gt"] / width
    df["norm_cx_pred"] = df["cx_pred"] / width
    df["norm_cy_gt"] = df["cy_gt"] / height
    df["norm_cy_pred"] = df["cy_pred"] / height

    # Parameters to evaluate
    param_pairs = [
        ("norm_cx", "norm_cx_gt", "norm_cx_pred"),
        ("norm_cy", "norm_cy_gt", "norm_cy_pred"),
        ("hfov", "hfov_gt", "hfov_pred"),
        ("k1", "k1_gt", "k1_pred"),
        ("k2", "k2_gt", "k2_pred"),
        ("k3", "k3_gt", "k3_pred"),
        ("p1", "p1_gt", "p1_pred"),
        ("p2", "p2_gt", "p2_pred"),
    ]

    # Compute metrics for each parameter
    metrics = []
    for name, gt_col, pred_col in param_pairs:
        mse, rmse, mae = compute_metrics(df[gt_col].values, df[pred_col].values)
        metrics.append({"parameter": name, "MSE": mse, "RMSE": rmse, "MAE": mae})

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame(metrics)

    # Define output path
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(csv_path).replace("-predictions.csv", "_normalized_error_metrics.csv")
    output_path = os.path.join(output_dir, output_filename)

    # Save metrics to CSV
    metrics_df.to_csv(output_path, index=False)
    print(f"Normalized parameter metrics saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compute normalized error metrics from prediction CSV.")
    parser.add_argument("--csv", required=True, help="Path to the <dataset>-predictions.csv file.")
    parser.add_argument("--width", type=int, required=True, help="Image width in pixels.")
    parser.add_argument("--height", type=int, required=True, help="Image height in pixels.")
    parser.add_argument("--output_dir", type=str, default="results_vit", help="Directory to save the output CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_normalized_metrics(args.csv, args.width, args.height, args.output_dir)
