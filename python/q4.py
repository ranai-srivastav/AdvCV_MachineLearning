import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

from matplotlib import pyplot as plt


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions
    ##########################
    ##### your code here #####
    image = image.astype(np.float32)
    gray = skimage.color.rgb2gray(image)

    blurred = skimage.filters.gaussian(gray, 1)

    otsu_threshold = skimage.filters.threshold_otsu(gray)

    thresholded = blurred > otsu_threshold
    thresholded = np.bitwise_not(thresholded).astype(np.float32)

    selem = skimage.morphology.disk(3)
    plt.imshow(thresholded, cmap="gray")
    # closed = skimage.morphology.binary_closing(thresholded, selem)
    # plt.imshow(closed, cmap="gray")
    opened = skimage.morphology.binary_dilation(thresholded, selem).astype(np.float32)
    plt.imshow(opened, cmap="gray")
    plt.show()

    clustered = skimage.measure.label(opened)
    regions = skimage.measure.regionprops(clustered)

    bboxes = [region.bbox for region in regions]
    ##########################

    return bboxes, opened
