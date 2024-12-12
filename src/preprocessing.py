import pandas as pd
import numpy as np
import json
from PIL import Image, ImageDraw
from tqdm import tqdm


def keep_roofs(
    image_path: str, 
    output_path: str,
    mask: np.ndarray, 
    color: list[int] = [255, 0, 255]
) -> None:
    """
    Overlay the mask on the image and keep only the pixels that are colored with the specified color.

    Args:
        image_path (str): The path to the image.
        output_path (str): The path to save the modified image.
        mask (np.ndarray): The mask to overlay on the image.
        color (list[int]): The color to keep, must be a 3-element list with the RGB values. Default is magenta, i.e. [255, 0, 255].
    
    Returns:
        PIL.Image: The image with only the pixels colored with the specified color.
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    image_array[mask == 0] = color
    modified_image = Image.fromarray(image_array)
    modified_image.save(output_path) 


def load_json_labels(file_path : str = "../data/labels/labels.json") -> pd.DataFrame:
    """
    Load the solar panels labels in JSON format..

    Args:
        file_path (str): The file containing JSON labels.
    Returns:
        pd.DataFrame: The labels for the dataset in a data frame.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    extracted_data = [
        {
            "data_title": entry.get("data_title"),
            "polygon": obj.get("polygon"),
            "value": obj.get("value"),
        } 
        for entry in data
        for unit in entry.get("data_units", {}).values()
        for obj in unit.get("labels", {}).get("objects", [{"polygon": None, "value": None}])
    ]
    return pd.DataFrame(extracted_data)


def generate_label_masks(
    df: pd.DataFrame,
    image_size: tuple[int, int] = (1000, 1000),
    target_value: str = "solar_panel",
) -> dict:
    """
    Create segmentation masks for images with labeled polygons.

    Args:
        df (pd.DataFrame): data frame containing `data_title`, `polygon`, and `value` features.
        image_size (tuple): dimensions of the image (width, height).
        target_value (str): value to filter polygons for masking (e.g., "solar_panel").

    Returns:
        dict: dictionary with `data_title` as keys and 1D numpy arrays (segmentation masks) as values.
    """
    image_names = df["data_title"].unique()
    masks = {name: np.zeros(image_size[0] * image_size[1]) for name in image_names}
    for _, row in tqdm(df.iterrows(), desc="Creating masks", total=len(df)):
        if row["value"] == target_value and isinstance(row["polygon"], dict):
            mask = Image.new("L", image_size)
            draw = ImageDraw.Draw(mask)
            polygon_points = [
                (vertex["x"] * image_size[0], vertex["y"] * image_size[1])
                for vertex in row["polygon"].values()
            ]
            draw.polygon(polygon_points, outline=1, fill=1)
            mask_array = np.array(mask).flatten()
            masks[row["data_title"]] += mask_array
            
    return {k: np.clip(v, 0, 1) for k, v in masks.items()}