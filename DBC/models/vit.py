import torch
from packaging import version
from torch import Tensor

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()

from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Callable, Tuple

# Helpers


def pair(t: int) -> Tuple[int, int]:
    """
    Converts an integer to a tuple of two identical integers.

    Parameters
    ----------
    t : int
        Input integer.

    Returns
    -------
    Tuple[int, int]
        A tuple with the same integer repeated twice.
    """
    return t if isinstance(t, tuple) else (t, t)


# Classes


class PreNorm(nn.Module):
    """
    Pre-normalization layer for transformers.

    Parameters
    ----------
    dim : int
        Dimension of the input.
    fn : Callable
        The function applied after normalization.

    Methods
    -------
    forward(x: Tensor, **kwargs) -> Tensor
        Applies layer normalization followed by the given function.
    """

    def __init__(self, dim: int, fn: Callable) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Applies layer normalization followed by the given function.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor after normalization and function application.
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Feedforward network for transformers.

    Parameters
    ----------
    dim : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float, optional
        Dropout rate (default is 0.0).

    Methods
    -------
    forward(x: Tensor) -> Tensor
        Applies the feedforward network to the input tensor.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes the input tensor through the feedforward network.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Processed output tensor.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head attention layer for transformers.

    Parameters
    ----------
    dim : int
        Input dimension.
    heads : int, optional
        Number of attention heads (default is 8).
    dim_head : int, optional
        Dimension of each attention head (default is 64).
    dropout : float, optional
        Dropout rate (default is 0.0).

    Methods
    -------
    forward(x: Tensor) -> Tensor
        Computes attention and outputs the processed tensor.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes the input tensor using multi-head attention.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor after attention.
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer block for the Vision Transformer model.

    Parameters
    ----------
    dim : int
        Input dimension.
    depth : int
        Number of layers.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension of each attention head.
    mlp_dim : int
        Dimension of the feedforward layer.
    dropout : float, optional
        Dropout rate (default is 0.0).

    Methods
    -------
    forward(x: Tensor) -> Tensor
        Processes the input tensor through multiple transformer layers.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes the input tensor through the Transformer block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) implementation.

    Parameters
    ----------
    image_size : int
        Input image size (assumed to be square).
    patch_size : int
        Size of image patches.
    num_classes : int
        Number of output classes.
    emb_dim : int
        Dimension of embeddings.
    depth : int
        Number of transformer layers.
    heads : int
        Number of attention heads.
    mlp_dim : int
        Dimension of the MLP layer.
    pool : str, optional
        Pooling method ('cls' or 'mean', default is 'cls').
    channels : int, optional
        Number of input channels (default is 3).
    dim_head : int, optional
        Dimension of each attention head (default is 64).
    dropout : float, optional
        Dropout rate (default is 0.1).
    emb_dropout : float, optional
        Dropout rate for the embedding layer (default is 0.1).

    Methods
    -------
    forward(img: Tensor) -> Tensor
        Processes the input image and outputs class predictions.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        emb_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1
    ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls or mean."

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(emb_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes))

    def forward(self, img: Tensor) -> Tensor:
        """
        Processes the input image through the Vision Transformer.

        Parameters
        ----------
        img : Tensor
            Input image tensor.

        Returns
        -------
        Tensor
            Output tensor with class predictions.
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
