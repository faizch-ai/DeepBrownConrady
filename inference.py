"""
Script Name: inference.py

Author: Faiz Chaudhry  
Company: AiLiveSim Ltd.

Purpose:
--------
This script performs inference using a specified model on a folder of images, computes camera parameters, 
and saves both ground truth and predicted values in a CSV file. The output includes camera intrinsics 
and distortion parameters in both raw and normalized forms.

Parameters Computed:
- fx, fy: Focal lengths in pixels
- cx, cy: Principal point coordinates in pixels
- k1, k2, k3: Radial distortion coefficients
- p1, p2: Tangential distortion coefficients

Additional Features:
- Computes timing information for the inference process and saves it to a text file.
- Supports batch processing and multi-threading for improved performance.

Usage:
-------
python inference.py --folder_path <path_to_folder> --output_dir <output_directory> 
                    --model_type <model_type> --model_path <path_to_model>
                    --optim_path <path_to_optimizer_config> --device <device_type>
                    --scaling_factor <scaling_factor> --batch_size <batch_size>
                    --num_workers <num_workers> --grayscale <True/False> 
                    --num_channels <number_of_channels>
"""
import os
import json
import math
import yaml
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader
from DBC.datasets.cps import CPSImageDataset
from DBC.models.model_utils import get_model, normalize
from DBC.utils import transformation_utils
from tqdm import tqdm
import time
from typing import Dict, List, Union

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=6)


def log_timing_info(
    dataset_name: str, 
    num_images: int, 
    start_time: float, 
    end_time: float, 
    output_dir: str
) -> None:
    """
    Logs the timing information of the inference process.

    Args:
        dataset_name (str): Name of the dataset being processed.
        num_images (int): Number of images processed.
        start_time (float): Start time of the inference process.
        end_time (float): End time of the inference process.
        output_dir (str): Directory to save the timing information.

    Returns:
        None
    """
    elapsed = end_time - start_time
    avg_time = elapsed / num_images if num_images > 0 else 0

    os.makedirs(output_dir, exist_ok=True)
    timing_path = os.path.join(output_dir, f"{dataset_name}-timing.txt")

    with open(timing_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total images: {num_images}\n")
        f.write(f"Total inference time: {elapsed:.2f} seconds\n")
        f.write(f"Average time per image: {avg_time:.4f} seconds\n")

    print(f"\nTotal inference time: {elapsed:.2f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds")
    print(f"Timing info saved to: {timing_path}")


def interpret_camera_params(output_np: np.ndarray, width: float, height: float) -> Dict[str, Union[float, str]]:
    """
    Converts model outputs to camera parameters.

    Args:
        output_np (np.ndarray): The model output as a numpy array.
        width (float): Image width.
        height (float): Image height.

    Returns:
        dict: A dictionary containing camera parameters.
    """
    h_fov_rad = output_np[0]
    fx = (width / 2) / math.tan(h_fov_rad / 2)
    cx = output_np[-2] * width
    cy = output_np[-1] * height

    return {
        "fov_rad": h_fov_rad,
        "fx": fx,
        "fy": fx,
        "cx": cx,
        "cy": cy,
        "k1": output_np[1],
        "k2": output_np[2],
        "k3": output_np[3],
        "p1": output_np[4],
        "p2": output_np[5],
    }


def test_model_on_folder(
    model_type: str,
    model_path: str,
    num_outputs: int,
    device: torch.device,
    scaling_factor: float,
    folder_path: str,
    grayscale: bool,
    degrees: bool,
    batch_size: int,
    num_workers: int,
    transformations: Dict[str, A.Compose],
    num_channels: int,
    optim_path: str,
    output_dir: str
) -> None:
    """
    Runs inference on images in a specified folder and saves the results to a CSV file.

    Args:
        model_type (str): Type of the model architecture.
        model_path (str): Path to the model file.
        num_outputs (int): Number of output parameters.
        device (torch.device): Device to run the model (e.g., 'cpu' or 'cuda').
        scaling_factor (float): Scaling factor for input images.
        folder_path (str): Path to the input folder containing images.
        grayscale (bool): Whether to convert images to grayscale.
        degrees (bool): Whether to convert angles to degrees.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        transformations (dict): Dictionary containing image transformations.
        num_channels (int): Number of input channels.
        optim_path (str): Path to the optimizer configuration file.
        output_dir (str): Directory to save the output CSV file.

    Returns:
        None
    """
    print("Loading the model on:", device)
    model = get_model(model_type=model_type, num_outputs=num_outputs, device=device, num_channels=num_channels)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Load normalization parameters
    if model_type == "resnet50_extended_features":
        with open(optim_path, "r") as file:
            optimizer_configuration = yaml.safe_load(file)
            normalization_params = optimizer_configuration.get("normalization", {})
            max_image_height = normalization_params.get("max_image_height")
            max_image_width = normalization_params.get("max_image_width")

    # Collect image paths
    image_paths = [os.path.join(root, f) for root, _, files in os.walk(folder_path) 
                   for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    if not image_paths:
        print("No image files found in folder:", folder_path)
        return

    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(folder_path)).replace("-test-benchmark", "").capitalize()
    results_path = os.path.join(output_dir, f"{dataset_name}-predictions.csv")

    # Create dummy CSV
    dummy_csv_path = "temp_inference_dataset.csv"
    with open(dummy_csv_path, "w") as f:
        f.write("directory,image_path,distortion_model,focal_length,fov,width,height,cx,cy,k1,k2,k3,p1,p2\n")
        for img in image_paths:
            f.write(f"_,{img}," + ",".join([""] * 12) + "\n")

    dataset = CPSImageDataset(
        dataset=dummy_csv_path,
        grayscale=grayscale,
        degrees=degrees,
        transformation=transformations["None"],
        scaling_factor=scaling_factor,
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    predictions = []
    start_time = time.time()

    print(f"Predicting on {len(image_paths)} images from: {folder_path}")
    with torch.no_grad():
        for i, (image, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            width = image.shape[-1]
            height = image.shape[-2]
            image = image.to(device)

            if model_type == "resnet50_extended_features":
                norm_height, norm_width = normalize([height, width], [max_image_height, max_image_width])
                additional_features = torch.tensor([[norm_height, norm_width]], dtype=torch.float32).to(device)
                output = model(image, additional_features=additional_features)
            else:
                output = model(image)

            output_np = output.detach().cpu().numpy()[0]
            pred = interpret_camera_params(output_np, width / scaling_factor, height / scaling_factor)
            img_path = image_paths[i]
            gt_path = os.path.splitext(img_path)[0] + ".json"

            if not os.path.isfile(gt_path):
                print(f"Skipping {img_path} (missing GT JSON)")
                continue

            with open(gt_path, "r") as f:
                gt = json.load(f)

            row = {"image": os.path.basename(img_path)}
            for key in ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]:
                row[f"{key}_gt"] = gt.get(key, "")
                row[f"{key}_pred"] = pred.get(key, "")
            predictions.append(row)

    end_time = time.time()
    log_timing_info(dataset_name, len(image_paths), start_time, end_time, output_dir)

    df = pd.DataFrame(predictions)
    df.to_csv(results_path, index=False)
    print(f"Saved predictions to: {results_path}")
    if os.path.exists("temp_inference_dataset.csv"):
        os.remove("temp_inference_dataset.csv")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run model inference and save GT + prediction CSV.")
    parser.add_argument("--folder_path", "-p", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="results", help="Output directory to save results.")
    parser.add_argument("--model_type", default="resnet50_extended_features")
    parser.add_argument("--model_path", default="./model/dbc.pth")
    parser.add_argument("--optim_path", default="./model/optimizer_config.yml")
    parser.add_argument("--num_outputs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grayscale", type=bool, default=True)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--degrees", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--scaling_factor", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    transformation_mean = transformation_utils.IMAGE_NET_MEAN_GREY
    transformation_std = transformation_utils.IMAGE_NET_STD_GREY
    transformations = {
        "None": A.Compose([
            A.Normalize(mean=transformation_mean, std=transformation_std),
            A.pytorch.ToTensorV2(),
        ])
    }

    test_model_on_folder(
        model_type=args.model_type,
        model_path=args.model_path,
        num_outputs=args.num_outputs,
        device=torch.device(args.device),
        scaling_factor=args.scaling_factor,
        folder_path=args.folder_path,
        grayscale=args.grayscale,
        degrees=args.degrees,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transformations=transformations,
        num_channels=args.num_channels,
        optim_path=args.optim_path,
        output_dir=args.output_dir
    )
