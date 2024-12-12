from PIL import Image
import matplotlib.pyplot as plt


def plot_mask_on_image(
    image_path: str,
    mask: dict,
    title: str,
    image_size: tuple[int, int] = (1000, 1000),
    alpha: float = 0.5,
):
    """
    Plot a segmentation mask on top of an image.

    Args:
        image_path (str): Path to the original image file.
        mask (np.ndarray): 1D numpy array (flattened) representing the segmentation mask.
        title (str): Title of the plot.
        image_size (tuple): Dimensions of the image (width, height).
        alpha (float): Transparency level of the mask overlay (0 to 1).
    """
    mask_2d = mask.reshape(image_size)
    image = Image.open(image_path).resize(image_size)

    plt.imshow(image, cmap="gray")
    plt.imshow(mask_2d, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title(title)