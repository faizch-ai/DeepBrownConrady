import torch
import torch.nn as nn
from typing import Tuple, Optional


def identity(x):
    return x


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layers. Defaults to 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        # Define the shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_layers: Tuple[int, int, int, int],
        num_output: int,
        dropout_rate: float = 0.25,
        num_channels: int = 3,
        postprocessing_function: callable = identity,
        preprocessing_function: callable = identity,
        additional_features_dim: int = 0,
        version: str = "",
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.num_output = num_output
        self.additional_features_dim = additional_features_dim

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(64, num_layers[0], stride=1)
        self.layer2 = self.make_layer(128, num_layers[1], stride=2)
        self.layer3 = self.make_layer(256, num_layers[2], stride=2)
        self.layer4 = self.make_layer(512, num_layers[3], stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize the fully connected layer with the assumption that additional features will be concatenated
        num_features = 512 + self.additional_features_dim
        self.fc = nn.Linear(num_features, self.num_output)

        self.postprocessing_function = postprocessing_function
        self.preprocessing_function = preprocessing_function
        self.version = version

    def make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, additional_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.preprocessing_function(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        if additional_features is not None:
            batch_size = x.shape[0]
            # Concatenate additional features
            x = torch.cat((x, additional_features.repeat(batch_size, 1)), dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.postprocessing_function(x)
        return x
