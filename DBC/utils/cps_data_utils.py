import numpy as np
import pathlib
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Tuple, Optional
import json


def read_directory_into_dataframe(src: str) -> pd.DataFrame:
    """
    Iterates through a directory containing CPS generated data, returning a Pandas DataFrame
    with directory number, image paths, and the parameters used to create the images. Searches for images with 'png' extension.
    """
    print("Reading directory into dataframe...")
    src = pathlib.Path(src)
    data_rows = []

    # Iterate through each sub-directory
    for main_dir in tqdm(src.iterdir(), total=len(list(src.iterdir()))):
        if main_dir.is_dir():
            for sub_dir in main_dir.iterdir():
                if sub_dir.is_dir():
                    parameter_file_path = sub_dir / "calibration.json"

                    if not parameter_file_path.is_file():
                        raise FileNotFoundError(f"File '{str(parameter_file_path)}' does not exist")

                    with open(parameter_file_path, "r") as file:
                        parameters = json.load(file)

                    # Extract parameters from JSON
                    fov = parameters["fov"]
                    intrinsic_matrix = parameters["K"]
                    width = parameters["width"]
                    height = parameters["height"]
                    k1 = parameters["k1"]
                    k2 = parameters["k2"]
                    k3 = parameters["k3"]
                    p1 = parameters["p1"]
                    p2 = parameters["p2"]
                    cx = parameters["cx"]
                    cy = parameters["cy"]

                    # Get the images
                    image_files = list(sub_dir.glob("*.png"))

                    # Append data for each image to the data_rows list
                    for image_file in image_files:
                        row = [
                            main_dir.name,
                            str(image_file),
                            "Brown-Conrady",
                            intrinsic_matrix[0][0],
                            fov,
                            width,
                            height,
                            cx,
                            cy,
                            k1,
                            k2,
                            k3,
                            p1,
                            p2,
                        ]
                        data_rows.append(row)

    columns = [
        "directory",
        "image_path",
        "distortion_model",
        "focal_length",
        "fov",
        "width",
        "height",
        "cx",
        "cy",
        "k1",
        "k2",
        "k3",
        "p1",
        "p2",
    ]
    df = pd.DataFrame(data_rows, columns=columns)
    return df


def split_dataframe(
    dataframe: pd.DataFrame,
    training_ratio: float,
    validation_ratio: float,
    group_by: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the given dataframe, randomly, into training-, validation- and test sets. Sizes of the
    resulting data sets are defined by the training_ratio and validation_ratio parameters.
    Size of the test set is (1.0 - training_ratio - validation_ratio). Sum of the ratios cannot
    exceed 1.0. If the parameter group_by is given, then the dataframe is first grouped based on the group_by criteria,
    and the returned training-, validation- and test sets contain full groups. For example, if the the group_by is
    'directory_nr', then the returned sets contain groups of 'directory_nr'.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pandas dataframe that is split
    training_ratio : float
        Ratio of the training data set [0...1.0]
    validation_ratio : float
        Ratio of the validation data set [0...1.0]
    group_by : Optional[str], optional
        xxx, by default None
    seed : Optional[int], optional
        Seed for the random number generator, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Randomly split subsets
    """

    if training_ratio + validation_ratio > 1.0:
        raise ("Sum of training_ration and validation_ratio cannot exceed 1.0")

    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(dataframe))
    np.random.shuffle(indices)

    training_size = int(len(indices) * training_ratio)
    validation_size = int(len(indices) * validation_ratio)

    training_indices = indices[:training_size]
    validtion_indices = indices[training_size : training_size + validation_size]
    test_indices = indices[training_size + validation_size :]

    return dataframe.iloc[training_indices], dataframe.iloc[validtion_indices], dataframe.iloc[test_indices]


def main(src: str, dst: str) -> None:
    """
    Iterates through a directory that contains CPS generated data and writes a csv-file that contains paths
    to the images and the parameters that were used to generate these.

    Parameters
    ----------
    src : str
        Directory that contains the CPS dataset
    dst : str
        File path where the resulting csv file is written to.
    """
    df = read_directory_into_dataframe(src)
    print(f"Read {len(df)} images from {src}")

    if dst == "":
        dst = str(pathlib.Path(src) / "cps_dataset.csv")

    df.to_csv(dst, index=False)
    print(f"Dataset saved to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset CSV for CPS images and parameters.")
    parser.add_argument("--src", type=str, required=True, help="Path to the folder with the CPS dataset")
    parser.add_argument(
        "--dst",
        type=str,
        default="",
        help="Path where the dataset.csv file is written to. If empty, the file is written to <SRC>/cps_dataset.csv",
    )

    args = parser.parse_args()
    main(args.src, args.dst)
