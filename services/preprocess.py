import os

from config.config import (
    TRAIN_IMGS_PATH,
    TRAIN_MASKS_PATH,
    TEST_IMGS_PATH,
    TEST_MASKS_PATH
)


def imgs_list(path: str) -> list:
    """
    The function allows to read all images from the selected folder
    and save it as a list.

    Parameters:
    - path - selected path to the folder
    """
    return [_ for _ in os.listdir(path) if _[-4:] in ['.jpg', '.png', 'jpeg']]


train_images: list = imgs_list(TRAIN_IMGS_PATH)
train_masks: list = imgs_list(TRAIN_MASKS_PATH)
test_images: list = imgs_list(TEST_IMGS_PATH)
test_masks: list = imgs_list(TEST_MASKS_PATH)
