import datetime
import torch
import os
from . import resnet32 as rs32
from . import resnet50 as rs50
from . import resnet50_extended_features as rs50_extended_features
from . import vit
from torch.nn import Module
from torch.optim import Optimizer
from torchvision import transforms

from typing import Any, Union, Tuple, Dict, Optional, List
import yaml
import pkgutil
from PIL import Image


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least loss, then save the
    model state.

    Args:
        output_dir: str
            The directory where the model will be saved.
        best_valid_loss: float
            The current best validation loss.

    Methods:
        __call__(current_valid_loss, epoch, model, optimizer, criterion):
            Save the model state if the current validation loss is less than the best_valid_loss.

    """

    def __init__(self, output_dir: str, model_version: str, best_valid_loss: float = float("inf")):
        """
        Initialize the SaveBestModel object.

        Args:
            output_dir (str):
                The directory where the model will be saved.
            model_version (str):
                The version of the model.
            best_valid_loss (float):
                The initial best validation loss. Default is float('inf'). Optional.
        """
        self.output_dir = output_dir
        self.best_valid_loss = best_valid_loss
        self.model_version = model_version

    def __call__(
        self,
        current_valid_loss: float,
        epoch: int,
        dataset: str,
        model_type: str,
        model: Module,
        optimizer: Optimizer,
        criterion: Any,
    ) -> None:
        """
        Save the model state if the current validation loss is less than the best_valid_loss.

        Args:
            current_valid_loss: float
                The current validation loss for the epoch.
            epoch: int
                The current epoch number.
            dataset: string
                Name of the dataset used.
            model_type: string
                Name of the used model.
            model: torch.nn.Module
                The PyTorch model to be saved.
            optimizer: torch.optim.Optimizer
                The optimizer used for training the model.
            criterion:
                The loss function used for training.

        """
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            # Get the current date and time
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d")  # Use a fixed timestamp for the first model

            # Create the file name with date, model type, and dataset name
            file_name = os.path.join(self.output_dir, f"best_model_{model_type}_{dataset}_{current_datetime}.pth")

            torch.save(
                {
                    "model_version": self.model_version,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                file_name,
            )
            return file_name
        return None


def save_model(
    model_version: str,
    epochs: int,
    dataset: str,
    model_type: str,
    model: Module,
    optimizer: Optimizer,
    criterion: Any,
    output_dir: str,
) -> None:
    """
    Function to save the trained model to disk.

    Args:
        model_version: str
            The version of the model
        epochs: int
            The total number of training epochs.
        dataset: string
            Name of the dataset used.
        model_type: string
            Name of the used model.
        model: torch.nn.Module
            The trained PyTorch model.
        optimizer: torch.optim.Optimizer
            The optimizer used for training the model.
        criterion:
            The loss function used for training.
        output_dir: str
            The directory where the model will be saved.

    """
    print("Saving final model...")

    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the file name with date, model type, and dataset name
    file_name = os.path.join(output_dir, f"final_model_{model_type}_{dataset}_{current_datetime}.pth")

    torch.save(
        {
            "model_version": model_version,
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        file_name,
    )
    return file_name


def get_model(
    model_type: str,
    num_outputs: int,
    device: torch.device,
    num_channels: int = 3,
    image_size: int = 0,
    model_configuration: str = None,
    additional_features_dim: int = 2,  # Add this parameter
    dropout_rate: float = 0.10,
) -> Module:
    """
    Get the specified model based on the given model_type and model configuration.

    Args:
        model_type: str
            The type of ResNet model to use. Choices: "resnet32","resnet50", "resnet50_scene",'resnet50_extended_features', 'resnet50_dropout' or "vit".
        image_size: int
            The expected length/width dimension of the image inputs. Only Useful if model_type="vit"
        num_outputs: int
            The number of output classes for the model.
        device: torch.device
            The device to which the model should be loaded.
        num_channels: int
            Set the number of channels to 3 if rgb and 1 if grayscale.
        model_configuration (str, optional):
            Path to the YAML file containing model parameters. Defaults to None.
        additional_features_dim: int, optional:
            Dimension of additional features to be concatenated with image features. Defaults to 0.
        dropout_rate: float, optional:
            The dropout rate to be applied to the dropout layers when running ResNet50_dropout.
    Returns:
        torch.nn.Module: The model with the specified number of output classes.

    Raises:
        ValueError: If an invalid model_type is provided.
        ValueError: If image_size isn't provided for ViT model.
    """

    if model_type == "resnet32":
        layers = [3, 4, 6]
        return rs32.ResNet(num_layers=layers, num_output=num_outputs, num_channels=num_channels).to(device)

    elif model_type == "resnet50":
        layers = [3, 4, 6, 3]
        return rs50.ResNet(num_layers=layers, num_output=num_outputs, num_channels=num_channels).to(device)

    elif model_type == "resnet50_scene":
        return rs50_scene.ResNet(num_layers=[3, 4, 6, 3], num_output=num_outputs).to(device)

    elif model_type == "resnet50_extended_features":
        layers = [3, 4, 6, 3]
        return rs50_extended_features.ResNet(
            num_layers=layers, num_output=num_outputs, num_channels=num_channels, additional_features_dim=additional_features_dim
        ).to(device)
    elif model_type == "vit":
        if image_size == 0:
            raise ValueError("Argument image_size expected")

        vit_parameters = load_model_configuration(model_configuration, model_type)
        vit_parameters.update({"image_size": image_size, "num_classes": num_outputs})

        return vit.ViT(**vit_parameters).to(device)

    elif model_type == "vit_extended":
        if image_size == 0:
            raise ValueError("Argument image_size expected for vit_extended model.")

        from . import vit_extended

        model = vit_extended.ViTExtended(
            image_size=image_size,
            patch_size=16,
            num_classes=num_outputs,
            emb_dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            channels=num_channels,
            additional_features_dim=additional_features_dim,
        )
        return model.to(device)

    else:
        raise ValueError(
            "Invalid model_type. Supported options: 'resnet32', 'resnet50', 'resnet50_scene','resnet50_extended_features', 'resnet50_dropout', 'vit' or 'vit_extended'."
        )


def load_model_configuration(model_cfg_path: Optional[str], model_type: str) -> Dict:
    """
    Loads model configuration parameters from a YAML file. If a specific configuration file
    path is not provided, the function loads default configuration parameters for the specified
    model type from a predefined set of configuration file paths.

    Args:
        model_cfg_path: str
            Path to the custom YAML configuration file.
        model_type: str
            Specifies the type of model for which the configuration is being loaded. Used
            to select the appropriate default configuration file from a preset dictionary.

    Returns:
        loaded_model_cfg: dict
            A dictionary containing the configuration parameters of the model.

    Raises:
        FileNotFoundError: If the specified `model_cfg_path` does not exist or the default file
            for the `model_type` is not defined in the `default_model_cfg_files` dictionary.
    """
    default_model_cfg_files = {
        "vit": "vit_config_default.yml",
    }
    loaded_model_cfg = None
    if model_cfg_path:
        # Load model parameters from the YAML file if provided
        cfg_path = model_cfg_path
        print(f"Model configuration path: \n\t{cfg_path}")
        with open(cfg_path, "r") as file:
            loaded_model_cfg = yaml.safe_load(file)
    else:
        cfg_path = default_model_cfg_files[model_type]
        print(f"Model configuration path: \n\t{cfg_path} (no file inputted, loading default file...)")
        file = pkgutil.get_data(__name__, cfg_path)
        loaded_model_cfg = yaml.safe_load(file)

    print("Model parameters:")
    for key in loaded_model_cfg:
        print(f"\t{key}: {loaded_model_cfg[key]}")

    return loaded_model_cfg


def get_model_parameters(model_type: str) -> tuple[float, float, int]:
    """
    Return parameters to be used for model optimizer.
    Args:
        model_type: str
            The model name to initialize.

    Returns:
        learning_rate: float
        gamma: float
            The learning rate decay factor
        step_size: int
            The intervals in epochs where the learning rate is decayed.f

    """

    if model_type in ["resnet32", "resnet50", "resnet50_scene", "resnet50_dropout"]:
        learning_rate = 0.01
        gamma = 0.1  # learning rate decay factor
        step_size = 50

    elif model_type == "vit":
        learning_rate = 3e-5
        gamma = 0.7  # learning rate decay factor
        step_size = 10

    # default values
    else:
        learning_rate = 0.001
        gamma = 0.1
        step_size = 10

    return learning_rate, gamma, step_size


def load_pretrained_weights(model: Module, model_type: str, pretrained_path: str, device: torch.device) -> None:
    """
    Load pretrained weights for all layers except the output layer.

    Args:
        model: torch.nn.Module
            The model for which to load the pretrained weights.
        model_type: str
            The type of the model ("resnet32", "resnet50", "resnet50_scene", "resnet50_dropout" or "vit").
        pretrained_path: str
            The path to the pretrained weights checkpoint.
    """
    # Load pre-trained weights for all layers except the output layer
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=device)["model_state_dict"]
    except KeyError:
        # If file loaded is the model itself, this should be run instead
        pretrained_dict = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()

    # Exclude output layer weights from the checkpoint
    if model_type in ["resnet32", "resnet50", "resnet50_scene", "resnet50_dropout"]:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "head" not in k}
    elif model_type == "vit":
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "mlp_head" not in k}
    else:
        print("Unrecognized model type.")

    # Update model dictionary with pre-trained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("Loaded model state_dict for other layers")


def get_model_version(pretrained_path: str) -> str:
    """
    This function returns the version of the model.

    Args:
        pretrained_path: str
            The path to the pretrained weights checkpoint.
    Returns:
        pretrained_version: str
    """
    # Load the checkpoint once
    checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

    # Retrieve model_version from the checkpoint if available, else return "N/A"
    pretrained_version = checkpoint.get("model_version", "N/A")

    return pretrained_version


def convert_to_onnx(
    model_or_path: Union[str, torch.nn.Module],
    model_type: str,
    num_outputs: int,
    device: torch.device,
    image_size: int = 64,
    output_path: str = None,
    num_images: int = 5,
    num_channels: int = 3,
) -> str:
    """
    Converts a PyTorch model to ONNX format, accepting either a model object or a path to a .pth file, along with the
    model type.

    Parameters
    ----------
    model_or_path : Union[str, torch.nn.Module]
        The model to be converted or the path to the pretrained model weights (.pth file).
    model_type : str
        The type of the model (e.g., 'vit', 'resnet32', 'resnet50', 'resnet50_scene', 'resnet50_dropout').
    image_size : int
        The height and width of the input images expected by the model.
    num_outputs : int
        The number of output features from the model.
    device : torch.device
        The device to run the model on ('cpu' or 'cuda').
    output_path : str, optional
        The file path where the ONNX model will be saved. If not provided, it will be derived from model_or_path.
    num_images : int
        If resnet50_scene is given as model_type, num_images is required for the number of images
    num_channels: int
        Num_channels is 3 when it is an RGB image and 1 if it's grayscale.

    Returns
    -------
    str
        The path to the saved ONNX model file.

    Raises
    ------
    FileNotFoundError
        If the model weights file does not exist when a path is provided.
    TypeError
        If model_or_path is neither a string nor a torch.nn.Module instance.
    """

    # Load the model from a file or use the provided model object
    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"Weights file not found at {model_or_path}")
        print(model_or_path)
        # Load the state dict or model directly from the .pth file
        state_dict_or_model = torch.load(model_or_path, map_location=device)
        if "model_state_dict" in state_dict_or_model:
            # Load model using get_model function if model_state_dict is present
            model = get_model(
                model_type=model_type, image_size=None, num_outputs=num_outputs, device=device, num_channels=num_channels
            )
            model.load_state_dict(state_dict_or_model["model_state_dict"])
        else:
            # If the loaded object is a model, assign it directly
            model = state_dict_or_model
    elif isinstance(model_or_path, torch.nn.Module):
        model = model_or_path
    else:
        raise TypeError("model_or_path must be a string (path to .pth file) or a torch.nn.Module instance.")

    # Set the model to evaluation mode and move it to the correct device
    model.to(device).eval()

    # Define example input data
    if model_type == "resnet50_scene":
        example_input = torch.randn(1, num_images, num_channels, image_size, image_size, device=device)
    else:
        example_input = torch.randn(1, num_channels, image_size, image_size, device=device)

    # Determine the output path for the ONNX model
    if output_path is None:
        if isinstance(model_or_path, str):
            output_path = model_or_path.replace(".pth", ".onnx")
        else:
            raise ValueError("output_path must be provided when model object is directly passed.")

    # Export the model to ONNX format
    torch.onnx.export(model, example_input, output_path, export_params=True, opset_version=11)
    return output_path


def get_num_channels(data_loader: torch.utils.data.DataLoader) -> int:
    """
    Determines the number of channels in the images from the first batch of the DataLoader.

    Args:
        training_loader (DataLoader): A PyTorch DataLoader object containing the training data.

    Returns:
        int: The number of channels in the images.
    """
    images, _ = next(iter(data_loader))
    return images.size(1)


def get_image_shape(data_loader: torch.utils.data.DataLoader) -> Tuple[int]:
    """
    Determines image shape from the first batch of the DataLoader.

    Returns
    -------
    Tuple[int]
        Image shape (height, width, channels)
    """
    if len(data_loader) == 0:
        raise RuntimeError("get_image_shape: data_loader length is 0")

    images, _, _ = next(iter(data_loader))  # <-- Now expect three outputs
    channels, height, width = images.size(1), images.size(2), images.size(3)

    return height, width, channels


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to retrieve the learning rate.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def normalize(values: List[float], max_values: List[float]) -> List[float]:
    """
    Normalize a list of values to a range of [-1, 1] based on corresponding max values.

    Args:
        values (List[float]): The list of values to normalize.
        max_values (List[float]): The list of maximum possible values for normalization.

    Returns:
        List[float]: The list of normalized values.
    """
    if len(values) != len(max_values):
        raise ValueError("The length of values and max_values must be the same.")

    normalized_values = [(value / max_value - 0.5) * 2 for value, max_value in zip(values, max_values)]
    return normalized_values


def resize_image_to_width(image: Image, new_width: int) -> Image:
    """From an image tensor return the image resized to the target width keeping the aspect ratio.

    Parameters
    ----------
    image_tensor: Image
        image to resize
    new_width: int
        Target width

    Returns
    -------
    Image
        The resized image as a PIL.Image
    """
    width, height = image.size
    aspect_ratio = height / width
    new_height = new_width * aspect_ratio
    new_height = int(new_height)
    resized_image = image.resize((new_width, new_height))
    return resized_image


def resize_aspect_ratio(image_tensor: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """From an image tensor return the image resized to the correct aspect ratio, losing (or adding) pixels on height and not width

    Parameters
    ----------
    image_tensor: torch.Tensor
        Tensor of the image
    width: int
        Target width
    height: int
        Target height

    Returns
    -------
    torch.Tensor
        The resized image as a tensor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    image = resize_image_to_width(to_pil(image_tensor), width)
    _, current_height = image.size
    left = 0
    top = int((current_height - height) / 2)
    right = width
    bottom = int((current_height + height) / 2)
    image = image.crop((left, top, right, bottom))
    image = to_tensor(image)
    padding = -int((current_height - height) / 2)
    return image, padding
