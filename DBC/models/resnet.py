import torch
import torch.nn as nn
from typing import List


def identity(x):
    return x


# Define the basic building block for ResNet
class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layers. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
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
        """
        Forward pass through the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)  # Add the shortcut connection
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture.

    Args:
        num_layers (List[int]): List of integers specifying the number of blocks in each layer.
        num_output (int): Number of output classes.
        dropout_rate (float, optional): Dropout rate for fully connected layers. Defaults to 0.25.
        preprocessing_function (callable,optionnal) preprocessing function to apply to x before forward
        postprocessing_function (callable,optionnal) preprocessing function to apply to x after forward
        version(str, optional): Save version as a parameter of the model.
    """

    def __init__(
        self,
        num_layers: List[int],
        num_output: int,
        dropout_rate: float = 0.25,
        preprocessing_function: callable = identity,
        postprocessing_function: callable = identity,
        version: str = "",
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Create the layers of the ResNet architecture
        self.layers = []
        for idx, num_blocks in enumerate(num_layers):
            stride = 1 if idx == 0 else 2
            out_channels = 64 * 2**idx  # Calculate the number of output channels for this block
            self.layers.append(self.make_layer(out_channels, num_blocks, stride))

        self.layers = nn.Sequential(*self.layers)

        # Head - Fully connected layers
        last_block_out_channels = 64 * 2 ** (len(num_layers) - 1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_block_out_channels, num_output),  # Use the calculated output channels here
        )
        self.preprocessing_function = preprocessing_function
        self.postprocessing_function = postprocessing_function
        self.version = version

    def make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create a layer consisting of multiple BasicBlocks.

        Args:
            out_channels (int): Number of output channels for the layer.
            num_blocks (int): Number of BasicBlocks in the layer.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.

        Returns:
            nn.Sequential: Layer consisting of multiple BasicBlocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet architecture.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.preprocessing_function(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.head(out)
        out = self.postprocessing_function(out)
        return out
