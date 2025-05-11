import torch
import torch.nn as nn


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


# Define the ResNet architecture
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
        num_layers: list[int],
        num_output: int,
        dropout_rate: float = 0.25,
        num_channels: int = 3,
        preprocessing_function: callable = identity,
        postprocessing_function: callable = identity,
        version: str = "",
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 16
        # Resizing Layer
        # self.resize = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False)
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Create the layers of the ResNet architecture
        self.layer1 = self.make_layer(16, num_layers[0], stride=2)
        self.layer2 = self.make_layer(32, num_layers[1], stride=2)
        self.layer3 = self.make_layer(64, num_layers[2], stride=2)

        # Head - Fully connected layers
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(p=dropout_rate), nn.Linear(64, num_output)
        )
        self.preprocessing_function = preprocessing_function
        self.postprocessing_function = postprocessing_function
        self.version = version

    def make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Create a layer consisting of multiple BasicBlocks.

        Args:
            out_channels (int): Number of output channels for the layer.
            num_blocks (int): Number of BasicBlocks in the layer.
            stride (int): Stride for the convolutional layers.

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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.head(out)
        out = self.postprocessing_function(out)
        return out
