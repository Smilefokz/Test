import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transparent_background import Remover

from services.metrics import iou_numpy
from config.config import (
    TEST_IMGS_PATH,
    TEST_MASKS_PATH,
    TEST_OUT_IMGS_PATH,
    TEST_OUT_MASKS_PATH,
    TRAIN_IMGS_PATH,
    TRAIN_MASKS_PATH,
    TRAIN_OUT_IMGS_PATH,
    TRAIN_OUT_MASKS_PATH
)


def remove_bg(remover: Remover, images: list, masks: list,
              test_mode: bool = True, graph_mode: bool = False):
    """
    The function allows to remove the background from an image and calculate
    the IoU metric based on the real and predicted masks of this image.

    Parameters:
    - remover - remove the background from an image
    - images - list with images names
    - masks - list with masks names
    - test_mode - switches the function to work with new data
    (True - by default)
    - graph_mode - switches the function to visual comparison of masks
    (False - by default)
    """
    # Paths for test and train images and masks:
    if test_mode:
        in_imgs_path: str = TEST_IMGS_PATH
        in_masks_path: str = TEST_MASKS_PATH
        out_imgs_path: str = TEST_OUT_IMGS_PATH
        out_masks_path: str = TEST_OUT_MASKS_PATH
    else:
        in_imgs_path: str = TRAIN_IMGS_PATH
        in_masks_path: str = TRAIN_MASKS_PATH
        out_imgs_path: str = TRAIN_OUT_IMGS_PATH
        out_masks_path: str = TRAIN_OUT_MASKS_PATH

    # list for storing IoU metric values:
    iou_list: list = []

    for n, (i, m) in enumerate(zip(images, masks)):
        img: Image = Image.open(in_imgs_path + i).convert('RGB')
        mask: Image = Image.open(in_masks_path + m).convert('RGB')

        # if image and mask sizes don't coincide,
        # converts mask size to the image size:
        if img.size != mask.size:
            mask = mask.resize(img.size)

        # remove background and save new image in .png format:
        out_pic: Image = remover.process(img, type='rgba', threshold=0.5)
        out_pic.save(out_imgs_path + str(n) + '.png')

        # get and save predicted mask in .png format:
        out_mask: Image = remover.process(img, type='map', threshold=0.5)
        out_mask.save(out_masks_path + str(n) + '.png')

        # calculation and preservation of IoU metric:
        iou: float = iou_numpy(np.array(out_mask), np.array(mask))
        iou_list.append(iou)

        # if graph_mode turn on:
        if graph_mode:
            # displays real and predicted masks:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.tight_layout(pad=0, w_pad=0, h_pad=0)
            ax.axis('off')
            plt.title(f'IoU = {round(iou, 4)}')
            plt.xticks([]), plt.yticks([])

            fig.add_subplot(1, 2, 1)
            plt.imshow(mask)
            plt.title('Real mask')
            plt.xticks([]), plt.yticks([])

            fig.add_subplot(1, 2, 2)
            plt.imshow(out_mask)
            plt.title('Predicted mask')
            plt.xticks([]), plt.yticks([])
            plt.show()
    return iou_list
