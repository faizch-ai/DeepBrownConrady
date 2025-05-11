import torch
import torch.nn as nn
from typing import Union
import os
from DBC.models.vit import ViT
from DBC.models import resnet50_scene as rs50_scene
from DBC.models import resnet32 as rs32
from DBC.models import resnet50 as rs50
from DBC.models import resnet50_extended_features as rs50_extended_features

import yaml
import torchvision.transforms as transforms


def identity(x):
    return x


def create_processing(export_config: str, pre_or_post: str) -> nn.Sequential:
    """From a export_config yml file read the func and param asked to generate the pre(or post)processing function for nn.Sequential type.

    Parameters
    ----------
    export_config : str
        Path to the yml file
    pre_or_post : str
        String = 'pre' or 'post' to see if we need to look at pre or post processing parameters and function

    Returns
    -------
    nn.Sequential
        Return the sequence of functions in nn.Sequential to be compiled with the model

    Raises
    ------
    Exception
        Argument pre_or_post isn't pre or post
    Exception
        There is different amount of block of parameters than functions, which means the user probably made a mistake
    Exception
        Call Resize function without any image_size.
    Exception
        Call Normalize function without any std.
    Exception
        Call Normalize function without any mean.
    Exception
        Ask a function not in : normalize or resize.
    """
    if pre_or_post != "pre" and pre_or_post != "post":
        raise Exception("pre_or_post argument needs to be equal to 'pre' or 'post' ")
    with open(export_config, "r") as file:
        config = yaml.safe_load(file)
    pre_procs = []
    functions = config[pre_or_post + "processing_func"].split(";")
    params = config[pre_or_post + "processing_func_param"].split(";")

    if len(functions) != len(params):
        raise Exception("Size of parameters and function doesn't match, verify the number of ;")
    for i in range(len(functions)):
        param = eval("{" + params[i] + "}")
        if functions[i] == "resize":
            if not (param["image_size"]):
                raise Exception("Please provide the image_size with resize function")
            pre_procs.append(transforms.Resize(param["image_size"]))
        elif functions[i] == "normalize":
            if "std" not in param:
                raise Exception("Please provide the std with normalize function")
            if "mean" not in param:
                raise Exception("Please provide the mean with normalize function")
            pre_procs.append(transforms.Normalize(mean=param["mean"], std=param["std"]))
        else:
            raise Exception(f"Name '{functions[i]}' not in the list of processing function accepted")

    return nn.Sequential(*pre_procs)


def save_model(export_config: str, save_pt: bool = True, save_onnx: bool = False):
    """From a export_config yml file, save the model in pt or/and onnx format for an easy use in the future.

    Parameters
    ----------
    export_config : str
        Path to the yml file
    save_pt : bool, optional
        Boolean to save (or not) the model also in Pytorch format, by default True
    save_onnx : bool, optional
        Boolean to save (or not) the model also in Onnx format please be careful
        that not all pre/post processing function works with onnx export, Only normalise for now by default False

    Raises
    ------
    Exception
        Error if there isn't model_type param in .yml file
    Exception
        Error if there isn't version param in .yml file
    Exception
        Error if there isn't device param in .yml file
    """
    with open(export_config, "r") as file:
        config = yaml.safe_load(file)
    if "postprocessing_func" in config:
        postprocessing_function = create_processing(export_config, "post")
    else:
        postprocessing_function = identity
    if "preprocessing_func" in config:
        preprocessing_function = create_processing(export_config, "pre")
    else:
        preprocessing_function = identity

    if "model_type" not in config:
        raise Exception("No attribute model_type in yml file.")
    model_type = config["model_type"]
    if "version" not in config:
        raise Exception("No attribute version in yml file.")
    print(preprocessing_function)
    if model_type == "resnet32":
        layers = [3, 4, 6]
        model = torch.jit.script(
            rs32.ResNet(
                num_layers=layers,
                preprocessing_function=preprocessing_function,
                postprocessing_function=postprocessing_function,
                version=config["version"],
                num_output=8,
            )
        )

    elif model_type == "resnet50":
        layers = [3, 4, 6, 3]
        model = torch.jit.script(
            rs50.ResNet(
                num_layers=layers,
                preprocessing_function=preprocessing_function,
                postprocessing_function=postprocessing_function,
                version=config["version"],
                num_output=8,
            )
        )

    elif model_type == "resnet50_scene":
        layers = [3, 4, 6, 3]
        model = torch.jit.script(
            rs50_scene.ResNet(
                num_layers=layers,
                preprocessing_function=preprocessing_function,
                postprocessing_function=postprocessing_function,
                version=config["version"],
                num_output=8,
            )
        )

    elif model_type == "resnet50_extended_features":
        layers = [3, 4, 6, 3]
        model = torch.jit.script(
            rs50_extended_features.ResNet(
                num_layers=layers,
                preprocessing_function=preprocessing_function,
                postprocessing_function=postprocessing_function,
                version=config["version"],
                num_output=8,
            )
        )
    if "weights_path" in config:
        model.load_state_dict(torch.load(config["weights_path"], map_location=config["device"])["model_state_dict"], strict=True)

    if "device" not in config:
        raise Exception("No attribute device in yml file.")
    model.to(config["device"])
    model.eval()
    if save_pt:
        if config["device"] == "cpu":
            torch.jit.save(model, config["folder_dst"] + config["model_type"] + "_v" + config["version"] + "_cpu.pt")
        else:
            torch.jit.save(model, config["folder_dst"] + config["model_type"] + "_v" + config["version"] + "_gpu.pt")
    if save_onnx:
        dummy_input = torch.randn(
            config["batch_size"], config["number_of_channel"], config["image_size_width"], config["image_size_height"]
        )
        # Export the model
        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            config["folder_dst"]
            + config["model_type"]
            + "_v"
            + config["version"]
            + ".onnx",  # where to save the model (can be a file or file-like object)
        )
