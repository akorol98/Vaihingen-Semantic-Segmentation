import numpy as np
import pandas as pd
import cv2

import os
from tqdm import tqdm


def rgb_to_binary_mask(img: np.array, color_map: list = None) -> np.array:
    """
    Converts a BGR image mask to binary mask

    High (H) and width (W) of the mask will be the same as for input image.
    Binary mask can have more then one output channel.
    The number of output channels (n_classes)
    will be defined by number of colors in the color_map.

    Parameters:
        img: BGR image mask of shape [H, W, 3]
        color_map: list of tuples representing color mappings

    Returns:
        out: binary mask of shape [n_classes, H, W]
    """

    # define color map
    if color_map is None:
        color_map = [
            (255, 255, 255),  # white
            (0, 255, 0),  # green
            (255, 0, 0),  # blue
            (255, 255, 0),  # turquoise
            (0, 255, 255),  # yellow
        ]

    # prepare empty list to collect binary mask for each color
    binary_masks = []

    # create binary  mask for each color and append it to the list
    for color in color_map:
        binary_mask = cv2.inRange(img, color, color)
        binary_mask[binary_mask > 0] = 1
        binary_masks.append(binary_mask)

    # concatenate the list and return
    return np.array(binary_masks)


def split_on_tiles(img: np.array, width: int, height: int) -> np.array:
    """
    Split an image on tiles with given width and height

    If the tile size is not an integer fraction of image
    size the remnant will be skipped.
    For exaple, if image size 1500x1000
    and tile size 200x200, we will get 7x5 tiles and 1400x1000 of
    effectively cropped area the rest will be skipped.

    Parameters:
        img: an image of shape [h, w, n_channels] to be splitted
        width: width of the tile
        height: height of the tile

    Returns:
        tiles: an array of tiles of shape [n_tiles, h, w, n_channels]
    """

    # prepare list to collect cropped tiles
    tiles = []
    # iterate
    for h in range(0, img.shape[0], height):
        for w in range(0, img.shape[1], width):
            tile = img[h:h + height, w:w + width, :]
            if tile.shape[0] + tile.shape[1] == width + height:
                tiles.append(img[h:h + height, w:w + width, :])

    return np.array(tiles)


def get_img_label(imgs: np.array) -> np.array:
    """
        Gives an image level for binary mask

        Parameters:
            imgs: a binary masks of shape [batch, n_classes, h, w]

        Returns:
            label: a label for given mask
    """

    label = (imgs.sum(axis=-1).sum(axis=-1) > 0).astype('int')

    return label


def makedirs(dirs: str):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


if __name__ == '__main__':

    makedirs('data/preprocessed/imgs')
    makedirs('data/preprocessed/masks')

    meta_data = []

    for file in tqdm(os.listdir('data/ISPRS_semantic_labeling_Vaihingen/top')):
        img = cv2.imread(f'data/ISPRS_semantic_labeling_Vaihingen/top/{file}')
        label = cv2.imread(f'data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/{file}')

        binary_mask = rgb_to_binary_mask(label)
        binary_mask = np.moveaxis(binary_mask, 0, -1)

        binary_mask_tiles = split_on_tiles(binary_mask, 200, 200)
        binary_mask_tiles = np.moveaxis(binary_mask_tiles, -1, 1)

        img_tiles = split_on_tiles(img, 200, 200)
        img_tiles = np.moveaxis(img_tiles, -1, 1)

        labels = get_img_label(binary_mask_tiles)

        for i in range(0, len(labels)):
            np.save(f'data/preprocessed/imgs/{file[:-4]}_tile{i}.npy', img_tiles[i])
            np.save(f'data/preprocessed/masks/{file[:-4]}_tile{i}.npy', binary_mask_tiles[i])

            temp_df = pd.DataFrame({
                'img': file[:-4],
                'tile': f'_tile{i}',
                'label': [labels[i]]
            })
            meta_data.append(temp_df)

    meta_data = pd.concat(meta_data).reset_index(drop=True)

    files = os.listdir('data/ISPRS_semantic_labeling_Vaihingen/top')
    files = list(map(lambda x: x[:-4], files))

    meta_data['split'] = 'train'
    meta_data.loc[meta_data.img.isin(files[3:-7]), 'split'] = 'weak_train'
    meta_data.loc[meta_data.img.isin(files[-7:]), 'split'] = 'validation'

    meta_data.to_csv('data/preprocessed/metadata.csv', index=False)
