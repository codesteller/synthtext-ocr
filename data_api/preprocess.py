from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GAUSSIAN_HEATMAP = None


def _init_gaussian_heatmap(sigma=10, spread=4):
    """
    `spread` multiplies the area of the heatmap matrix, `sigma` defines width of bulge around the center.
    For a larger area, increase spread to 3 or 5. For a larger center, increase corresponding sigma to 11 and 12.
    Recommended values: (sigma=10, spread=4),(sigma=11, spread=5), (sigma=12, spread=7),
    :param sigma:
    :param spread:
    :return:
    """
    global GAUSSIAN_HEATMAP
    if GAUSSIAN_HEATMAP is None:
        extent = int(spread * sigma) + 1
        center = spread * sigma / 2
        GAUSSIAN_HEATMAP = np.zeros([extent, extent], dtype=np.float32)

        for i in range(extent):
            for j in range(extent):
                GAUSSIAN_HEATMAP[i, j] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i - center) ** 2 + (j - center) ** 2) / (sigma ** 2))

        GAUSSIAN_HEATMAP = (GAUSSIAN_HEATMAP / np.max(GAUSSIAN_HEATMAP) * 255).astype(np.uint8)
    return GAUSSIAN_HEATMAP


GAUSSIAN_HEATMAP = _init_gaussian_heatmap()


# Initialized the module


def make_heatmap(im_path: Path, char_bbox, text):
    char_bbox = char_bbox.transpose((2, 1, 0))
    with Image.open(im_path) as img:
        w, h = img.size

    region_map = np.zeros([h, w], dtype=np.uint8)
    affinity_map = np.zeros([h, w], dtype=np.uint8)

    # create character region map
    _make_region_map(region_map, char_bbox)

    # create affinity region map
    affinity_bbox = _derive_affinity_boxes(char_bbox, text)

    if affinity_bbox is not None:
        _make_region_map(affinity_map, affinity_bbox)

    return region_map, affinity_map, affinity_bbox


def _make_region_map(region_map, char_bbox):
    for char_i in range(char_bbox.shape[0]):
        bbox = char_bbox[char_i].copy()

        # given char bounding box, check if its inside the image dimensions or not
        if np.any(bbox < 0) or np.any(bbox[:, 0] > region_map.shape[1]) or np.any(bbox[:, 1] > region_map.shape[0]):
            continue

        # get the top left pt of the char
        top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
        if top_left[1] > region_map.shape[0] or top_left[0] > region_map.shape[1]:
            # This means there is some bug in the character bbox
            # Will have to look into more depth to understand this
            continue
        bbox -= top_left[None, :]
        transformed = _four_point_transform(GAUSSIAN_HEATMAP.copy(), bbox.astype(np.float32))

        start_row = max(top_left[1], 0) - top_left[1]
        start_col = max(top_left[0], 0) - top_left[0]
        end_row = min(top_left[1] + transformed.shape[0], region_map.shape[0])
        end_col = min(top_left[0] + transformed.shape[1], region_map.shape[1])

        region_map[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[
                                                                                start_row:end_row - top_left[1],
                                                                                start_col:end_col - top_left[0]]

    return region_map


def _derive_affinity_boxes(char_bbox, text):
    all_words = []
    for j in text:
        all_words += [k for k in ' '.join(j.split('\n')).split() if k != '']

    affinity_boxes = []
    total_chars = 0
    for word in all_words:
        for _ in range(len(word) - 1):
            char_box_1 = char_bbox[total_chars].copy()
            char_box_2 = char_bbox[total_chars + 1].copy()

            center_1, center_2 = np.mean(char_box_1, axis=0), np.mean(char_box_2, axis=0)
            tl = np.mean([char_box_1[0], char_box_1[1], center_1], axis=0)
            bl = np.mean([char_box_1[2], char_box_1[3], center_1], axis=0)
            tr = np.mean([char_box_2[0], char_box_2[1], center_2], axis=0)
            br = np.mean([char_box_2[2], char_box_2[3], center_2], axis=0)

            affinity = np.array([tl, tr, br, bl])
            affinity_boxes.append(affinity)
            total_chars += 1
        total_chars += 1
    return np.stack(affinity_boxes, axis=0) if len(affinity_boxes) > 0 else None


def _four_point_transform(image, pts):
    max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

    dst = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(image, transform_matrix, (max_x, max_y))

    return warped


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def sigmoid(x):
        return 1. / (1. + np.exp(-x))


    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(GAUSSIAN_HEATMAP)
    plt.subplot(2, 3, 2)
    plt.imshow(GAUSSIAN_HEATMAP / 255.)
    plt.subplot(2, 3, 3)
    plt.imshow(sigmoid(GAUSSIAN_HEATMAP / 255.))
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.resize(GAUSSIAN_HEATMAP, dsize=None, fx=0.5, fy=0.5))
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.resize(GAUSSIAN_HEATMAP, dsize=None, fx=0.5, fy=0.5) / 255.)
    plt.show()
