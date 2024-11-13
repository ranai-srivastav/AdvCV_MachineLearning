import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


for img in os.listdir("../images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox

        if (maxr-minr) <= 15 or (maxc - minc) <= 15:
            bboxes.remove(bbox)
            continue

        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
            )
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    bboxes = np.array(bboxes)
    centers = np.array([(bboxes[:, 0] + bboxes[:, 2]) // 2.0, bboxes[:, 1] + bboxes[:, 3] // 2.0, np.arange(bboxes.shape[0])]).T

    sorted_rows = np.array(sorted(centers, key=lambda x: x[0]))
    row_wise = []
    prev_pixel = sorted_rows[0, :]
    row = [prev_pixel]
    threshold = 32
    for idx in range(1, sorted_rows.shape[0]):
        r, c, _ = sorted_rows[idx, :]
        pr, pc, _ = prev_pixel

        if (r - pr) <= threshold:
            row.append(sorted_rows[idx, :])
        else:
            row_wise.append(row)
            row = [ sorted_rows[idx, :] ]
        prev_pixel = sorted_rows[idx, :]
    row_wise.append(row)
    sorted_rc = [sorted(row, key=lambda x: x[1]) for row in row_wise]
    ##########################

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    bw = np.abs(bw - 1)
    im_h, im_w = bw.shape[:2]
    row_wise_letter_imgs = []
    for row in sorted_rc:
        row_letter_imgs = np.zeros( [len(row), 32*32] )
        for sample_idx, pixel in enumerate(row):
            center_r, center_c, bbox_idx = pixel
            bbox_tl_r, bbox_tl_c, bbox_br_r, bbox_br_c = bboxes[int(bbox_idx)]

            bbox_h = bbox_br_r - bbox_tl_r
            bbox_w = bbox_br_c - bbox_tl_c

            pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0
            if bbox_h > bbox_w:
                larger_dim = bbox_h
                pad_w = bbox_h - bbox_w
                pad_l = pad_w // 2
                pad_r = pad_w - pad_l
            else:
                larger_dim = bbox_w
                pad_h = bbox_w - bbox_h
                pad_t = pad_h // 2
                pad_b = pad_h - pad_t

            square = np.full([larger_dim, larger_dim], 1, np.float32)
            # plt.imshow(square, cmap = "Greys")
            # plt.show()

            square[pad_t:pad_t + bbox_h, pad_l:pad_l + bbox_w] = bw[bbox_tl_r:bbox_br_r, bbox_tl_c:bbox_br_c]
            # plt.imshow(square, cmap = "Greys")
            # plt.show()

            prepad = skimage.transform.resize(square, [28, 28])
            # plt.imshow(prepad, cmap = "Greys")
            # plt.show()

            prepad = prepad.T
            # plt.imshow(prepad, cmap = "Greys")
            # plt.show()

            padded = np.full([32, 32], 1, np.float32)
            # plt.imshow(padded, cmap = "Greys")
            # plt.show()

            padded[2:30, 2:30] = prepad
            # plt.imshow(padded, cmap = "Greys")
            # plt.show()

            padded = padded**2/np.max(padded)

            # plt.imshow(ready_img, cmap="gray")
            # plt.show()
            row_letter_imgs[sample_idx, :] = padded.flatten()
        row_wise_letter_imgs.append(row_letter_imgs)


    ##########################

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("q3_weights.pickle", "rb"))
    row_wise_letter_preds = []
    ##########################
    ##### your code here #####
    for row_images in row_wise_letter_imgs:
        h1 = forward(row_images, params, "layer1")
        probs = forward(h1, params, "output", softmax)
        pred_classes = np.argmax(probs, axis=1)
        pred_char = letters[pred_classes]

        print("".join(pred_char))
        row_wise_letter_preds.append(pred_char)
    ##########################
