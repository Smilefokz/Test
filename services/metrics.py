import numpy as np


def iou_numpy(outputs: np.array, labels: np.array) -> float:
    """
    The function allows to calculate the IoU metric using Numpy.

    Parameters:
    - outputs - predicted masks
    - labels - real masks
    """
    intersection: np.array = (outputs & labels).sum((1, 2))
    union: np.array = (outputs | labels).sum((1, 2))

    iou: np.array = (intersection + 1e-6) / (union + 1e-6)

    thresholded: np.array = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded.mean()
