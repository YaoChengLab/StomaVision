import random
import numpy as np
import cv2
from skimage import measure


def binary_mask_to_polygon(binary_mask, tolerance=0):
    r"""Converts a binary mask to COCO polygon representation

    Args
    ----
        - binary_mask: a 2D binary numpy array where '1's represent the object
        - tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """

    def close_contour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5, fully_connected="high")
    for contour in contours:
        contour = close_contour(contour)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        #         segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons


def fit_polygons_to_rotated_bboxes(polygons):
    r"""
    convert polygons to rotated bboxes using cv2.minAreaRect().

    Args:
     - polygons (list): is a list of polygon points [x1, y1, x2, y2,...]
    """
    rbboxes = []
    for p in polygons:
        pts_x = p[::2]
        pts_y = p[1::2]
        pts = [[x, y] for x, y in zip(pts_x, pts_y)]
        pts = np.array(pts, np.float32)
        rect = cv2.minAreaRect(pts)  #  ((cx, cy), (w, h), a)
        rbboxes.append(rect)
    return rbboxes


def calc_polygon_area(polygons):
    r"""Calculate polygon area with shoelace formula

    Args:
     - polygons (list): is a list of polygon points [x1, y1, x2, y2,...]
    """
    areas = []
    for p in polygons:
        pts_x = p[::2]
        pts_y = p[1::2]
        pts = [[x, y] for x, y in zip(pts_x, pts_y)]
        pts = np.array(pts, np.float32)
        # shift coordinates
        x_ = pts_x - np.mean(pts_x)
        y_ = pts_y - np.mean(pts_y)
        # calculate area
        correction = x_[-1] * y_[0] - y_[-1] * x_[0]
        main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
        area = 0.5 * np.abs(main_area + correction)
        areas.append(area)

    return areas


def random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
