import torch
import torch.nn as nn
from typing import Tuple


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
        num_layers (Tuple[int, int, int, int]): Number of blocks in each stage.
        num_output (int): Number of output classes.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.25.
        num_channels (int, optional): channels can be 3 for RGB or 1 for grayscale
    """

    def __init__(
        self,
        num_layers: Tuple[int, int, int, int],
        num_output: int,
        dropout_rate: float = 0.25,
        num_channels: int = 3,
        preprocessing_function: callable = identity,
        postprocessing_function: callable = identity,
        version: str = "",
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Create the layers of the ResNet architecture
        self.layer1 = self.make_layer(64, num_layers[0], stride=1)
        self.layer2 = self.make_layer(128, num_layers[1], stride=2)
        self.layer3 = self.make_layer(256, num_layers[2], stride=2)
        self.layer4 = self.make_layer(512, num_layers[3], stride=2)
        # Head - Fully connected layers
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_output),
        )
        self.postprocessing_function = postprocessing_function
        self.preprocessing_function = preprocessing_function
        self.version = version

    def make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Create a layer in ResNet.

        Args:
            out_channels (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the layer.

        Returns:
            nn.Sequential: Layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagates an input tensor through the neural network model bypassing the head layer.

        Args:
            x: (torch.Tensor):
                The input tensor to the neural network. The dimensions and data type of `x` should be compatible with
                the first layer of the network.

        Returns:
            torch.Tensor:
                The output tensor after it has been processed by the network.

        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def forward_decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes the input tensor through a series of stages to generate multiple outputs, including
        feature extraction, projection matrix computation, and cosine similarity calculation.

        Args:
            x (torch.Tensor): The input tensor. Expected to be a multidimensional tensor, typically
            representing data such as images.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
              1. result: The output of the model's head applied to the features.
              2. flattened_features.t(): The transposed flattened features after adaptive average pooling.
              3. h_projected: Projected components using projection matrices.
              4. h_orthogonal: Orthogonal components calculated using projection matrices.
              5. cos_similarity: Cosine similarity for each output with respect to the flattened features.

        Workflow:
            1. Extract features from the input tensor.
            2. Applies the model's head to the features to obtain the primary result.
            3. Applies Adaptive Average Pooling and flattening to the features.
            4. Normalizes the weight matrix from the model's head.
            5. Computes the projection matrices.
            6. Calculates the projected and orthogonal components of the features.
            7. Computes the cosine similarity for each output.

        Note:
            This function is part of a larger neural network architecture and utilizes PyTorch's nn module.
        """
        features = self.get_features(x)
        batch_size = features.shape[0]
        result = self.head(features)
        num_outputs = result.shape[1]

        # Apply AdaptiveAvgPool2d and Flatten
        adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        pooled_features = adaptive_avg_pool(features)
        flattened_features = flatten(pooled_features)

        # Extract and normalize the weight matrix
        w_matrix = self.head[3].weight.data
        norms = torch.norm(w_matrix, p=2, dim=1, keepdim=True)
        normalized_w_matrix = w_matrix / norms

        # Compute projection matrices
        projection_matrices = torch.bmm(normalized_w_matrix.unsqueeze(2), normalized_w_matrix.unsqueeze(1))

        # Compute projected and orthogonal components
        flattened_features_transposed = flattened_features.t()
        h_projected = torch.bmm(projection_matrices, flattened_features_transposed.unsqueeze(0).expand(num_outputs, -1, -1))
        identity_matrix = torch.eye(projection_matrices.shape[1], device=x.device).unsqueeze(0).expand(num_outputs, -1, -1)
        h_orthogonal = torch.bmm(
            identity_matrix - projection_matrices, flattened_features_transposed.unsqueeze(0).expand(num_outputs, -1, -1)
        )
        # Calculate cosine similarity for each output
        cos_similarity = torch.empty(num_outputs, batch_size, device=x.device)
        for i in range(num_outputs):
            cos_similarity[i] = nn.CosineSimilarity(dim=0, eps=1e-8)(flattened_features_transposed, h_projected[i])
        return result, flattened_features.t(), h_projected, h_orthogonal, cos_similarity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagates an input tensor through the neural network model.

        Args:
            x: (torch.Tensor):
                The input tensor to the neural network. The dimensions and data type of `x` should be compatible with
                the first layer of the network.

        Returns:
            torch.Tensor:
                The output tensor after it has been processed by the network.

        """
        x = self.preprocessing_function(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.head(out)
        out = self.postprocessing_function(out)
        return out
