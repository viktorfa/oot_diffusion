import numpy as np
import cv2
from PIL import Image


label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}


def get_palette(num_cls: int):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


def find_midpoint_y(mask: np.ndarray):
    # Find all the y coordinates of neck pixels
    y_coords, _ = np.where(mask)
    if len(y_coords) > 0:
        # Calculate the midpoint
        midpoint = np.median(y_coords).astype(int)
        return midpoint
    else:
        return None


def delete_irregular(logits_result):
    parsing_result = np.argmax(logits_result, axis=2)
    upper_cloth = np.where(parsing_result == 4, 255, 0)
    contours, hierarchy = cv2.findContours(
        upper_cloth.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    area = []
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        area.append(abs(a))
    if len(area) != 0:
        top = area.index(max(area))
        M = cv2.moments(contours[top])
        cY = int(M["m01"] / M["m00"])

    dresses = np.where(parsing_result == 7, 255, 0)
    contours_dress, hierarchy_dress = cv2.findContours(
        dresses.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    area_dress = []
    for j in range(len(contours_dress)):
        a_d = cv2.contourArea(contours_dress[j], True)
        area_dress.append(abs(a_d))
    if len(area_dress) != 0:
        top_dress = area_dress.index(max(area_dress))
        M_dress = cv2.moments(contours_dress[top_dress])
        cY_dress = int(M_dress["m01"] / M_dress["m00"])
    wear_type = "dresses"
    if len(area) != 0:
        if len(area_dress) != 0 and cY_dress > cY:
            irregular_list = np.array([4, 5, 6])
            logits_result[:, :, irregular_list] = -1
        else:
            irregular_list = np.array([5, 6, 7, 8, 9, 10, 12, 13])
            logits_result[:cY, :, irregular_list] = -1
            wear_type = "cloth_pant"
        parsing_result = np.argmax(logits_result, axis=2)
    # pad border
    parsing_result = np.pad(
        parsing_result, pad_width=1, mode="constant", constant_values=0
    )
    return parsing_result, wear_type


def hole_fill(img: Image.Image):
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask(mask: np.ndarray):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
        # keep large area in skin case
        for j in range(len(area)):
            if j != i and area[i] > 2000:
                cv2.drawContours(refine_mask, contours, j, color=255, thickness=-1)
    return refine_mask


def remove_outliers(
    image: np.ndarray,
):
    image = image.astype(np.uint8)

    # Define a kernel size for the morphological operation
    kernel_size = 3  # You can adjust this to be larger if the noise dots are bigger
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform morphological opening
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def refine_hole(parsing_result_filled, parsing_result, arm_mask):
    filled_hole = (
        cv2.bitwise_and(
            np.where(parsing_result_filled == 4, 255, 0),
            np.where(parsing_result != 4, 255, 0),
        )
        - arm_mask * 255
    )
    contours, hierarchy = cv2.findContours(
        filled_hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    refine_hole_mask = np.zeros_like(parsing_result).astype(np.uint8)
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        # keep hole > 2000 pixels
        if abs(a) > 2000:
            cv2.drawContours(refine_hole_mask, contours, i, color=255, thickness=-1)
    return refine_hole_mask + arm_mask
