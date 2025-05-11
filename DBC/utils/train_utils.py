from matplotlib import pyplot as plt
from torch import Tensor


def visualize_images(images: Tensor, samples: int = 10, cols: int = 5) -> None:
    """
    Visualize some samples from the batch of augmented images.

    Args:
        images (torch.Tensor): The batch of augmented images.
        samples (int, optional): Number of images to show in a grid. Defaults to 10.
        cols (int, optional): Number of columns in the grid. Defaults to 5.

    Returns:
        None: Shows a grid of images from the batch.
    """
    if images.shape[0] < samples:
        samples = images.shape[0]
        rows = 2
    else:
        rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        ax.ravel()[i].imshow(img)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
