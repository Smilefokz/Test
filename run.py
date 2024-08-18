from transparent_background import Remover

from services.remover import remove_bg
from services.preprocess import test_images, test_masks

remover: Remover = Remover(mode='base')

if __name__ == '__main__':

    iou_list: list = remove_bg(remover, test_images, test_masks)

    if len(iou_list) != 0:
        file = open('results.txt', 'w+')

        for (iou, i) in zip(iou_list, test_images):
            text = 'IOU metric between real and predicted masks for image '
            file.write(text + f'{i} is above {round(iou, 5)}\n')
            print(text + f'{i} is above {round(iou, 5)}')

        mean_iou = sum(iou_list)/len(iou_list)
        file.write(f'Mean IOU metric for all images is above {round(mean_iou, 5)}')
        print(f'Mean IOU metric for all images is above {round(mean_iou, 5)}')

        file.close()
    else:
        print('Please add images in "test/pictures/in" and masks in "test/masks/in".')
