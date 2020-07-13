import numpy as np
from trainer.config import config


# python functions are implemented in this file
def normalize_coordinate_to_grid_cell(bbox, x_grid_cell_index, y_grid_cell_index):
    '''

    :param bbox: [x_mid, y_mid, w, h]
    :param x_grid_cell_index: grid cell index in x direction (0 ~ config.num_grid_x-1)
    :param y_grid_cell_index: grid cell index in y direction (0 ~ config.num_grid_y-1)
    :return:
    '''

    x_offset = config.grid_width * x_grid_cell_index
    y_offset = config.grid_height * y_grid_cell_index

    x_mid, y_mid, w, h = bbox

    x_mid_norm = (x_mid - x_offset) / config.grid_width
    y_mid_norm = (y_mid - y_offset) / config.grid_height

    # Normalizing the box W & H may not be necessary since the image is already standardized in size,
    # depends on the activiation of choice in model
    w_norm = w / config.grid_width
    h_norm = h / config.grid_height


    return [x_mid_norm, y_mid_norm, w_norm, h_norm]


def bbox_2_grid_cell(bboxs):
    '''

    :param bboxs:
    :return:
    '''

    grid_label = np.zeros((config.num_grid_y, config.num_grid_x, config.box_vector_size))

    for box in bboxs:
        y_min, x_min, y_max, x_max = box
        y_mid = y_min + (y_max - y_min) / 2
        x_mid = x_min + (x_max - x_min) / 2

        y_grid_cell_index = int(y_mid / config.grid_height)
        x_grid_cell_index = int(x_mid / config.grid_width)


        # convert to format: [X_MID, Y_MID, W, H] and than normalize coordinate for each cell
        box_normalized = normalize_coordinate_to_grid_cell([x_mid, y_mid, x_max - x_min, y_max - y_min],
                                                           x_grid_cell_index, y_grid_cell_index)

        box_vector = [1] + box_normalized
        grid_label[y_grid_cell_index, x_grid_cell_index, :] = box_vector

    return grid_label


def grid_cell_2_bbox(grid_cell, threshold=0):
    """

    :param grid_cell Tensor (Batch_Size, config.num_grid_y, config.num_grid_x, config.box_vector_size):
    :return: bbox_list a list of python bbox list after non-maximum suppression
    """
    batch_size = grid_cell.shape[0]
    bbox_grid = np.zeros((batch_size, config.num_grid_y, config.num_grid_x, 4))
    for b in range(len(grid_cell)):
        for r in range(config.num_grid_y):
            for c in range(config.num_grid_x):
                if grid_cell[b, r, c][0] > threshold:
                    bbox_grid[b, :] = _bbox_coordinate_cell_2_image(grid_cell[b, r, c][1:], r, c)

    bbox_grid = np.reshape(bbox_grid, [config.batch_size, config.num_grid_y * config.num_grid_x, 4])
    return bbox_grid



def IoU(box_1, box_2):
    """

    :param box_1: np.array, (ymin, xmin, ymax, xmax)
    :param box_2: np.array, (ymin, xmin, ymax, xmax)
    :return: the Intersection of Union between box_1 and box_2
    """
    area1 = _cal_rect_area(*box_1)
    area2 = _cal_rect_area(*box_2)
    area_intxn = _cal_intxn_area(*box_1, *box_2)

    iou = (area_intxn / (area1 + area2 - area_intxn)) * 1.0

    return iou


def non_maximum_suppression(bbox_list, probs=None, IoU_threshold=0.3):
    """Perform non maximum suppression on the prediceted bounding boxes

    Args:
        bbox_list: `np.array`
        probs: `float`
        IoU_threshold: `float`
    """
    # if there are no boxes, return an empty list
    if len(bbox_list) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    ymins = bbox_list[:, 0]
    xmins = bbox_list[:, 1]
    ymaxs = bbox_list[:, 2]
    xmaxs = bbox_list[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    areas = _cal_rect_area(ymins, xmins, ymaxs, xmaxs)

    # sort by probs or ymax
    idxs = probs if probs else ymaxs
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        yy1 = np.maximum(ymins[i], ymins[idxs[:last]])
        xx1 = np.maximum(xmins[i], xmins[idxs[:last]])
        yy2 = np.minimum(ymaxs[i], ymaxs[idxs[:last]])
        xx2 = np.minimum(xmaxs[i], xmaxs[idxs[:last]])

        # compute the width and height of the bounding box
        h = np.maximum(0, yy2 - yy1)
        w = np.maximum(0, xx2 - xx1)

        # compute the ratio of overlap
        overlaps = (h * w) / areas[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlaps > IoU_threshold)[0])))

    # return only the bounding boxes that were picked
    return bbox_list[pick]


def FROC():
    """Todo
    Compute the FROC (Case Level) given prediction results
    :return:
    """
    froc = -1
    return froc


def _bbox_coordinate_cell_2_image(bbox_local, y_grid_cell_index, x_grid_cell_index):
    """
    
    Args:
        bbox_local:
        y_grid_cell_index:
        x_grid_cell_index:

    Returns:

    """
    x_mid_norm, y_mid_norm, w_norm, h_norm = bbox_local
    x_offset = config.grid_width * x_grid_cell_index
    y_offset = config.grid_height * y_grid_cell_index
    x_mid = x_mid_norm * config.grid_width + x_offset
    y_mid = y_mid_norm * config.grid_height + y_offset

    w = w_norm * config.grid_width
    h = h_norm * config.grid_height


    xmin = (2 * x_mid - w) / 2
    ymin = (2 * y_mid - h) / 2
    xmax = (2 * x_mid + w) / 2
    ymax = (2 * y_mid + h) / 2

    return [xmin, ymin, xmax, ymax]


def _cal_rect_area(ymin, xmin, ymax, xmax):
    """area of rect equals height * width

    Args:
        xmin: `np.float` or `np.array`,
        ymin: `np.float` or `np.array`,
        xmax: `np.float` or `np.array`,
        ymax: `np.float` or `np.array`,

    Returns:
        `np.float` or `np.array`
    """
    return (ymax - ymin) * (xmax - xmin)


def _cal_intxn_area(ymin_1, xmin_1, ymax_1, xmax_1, ymin_2, xmin_2, ymax_2, xmax_2):
    """find max_min_x, max_min_y, min_max_x, min_max_y"""
    area_intxn = 0

    intxn_ymin = max(ymin_1, ymin_2)
    intxn_xmin = max(xmin_1, xmin_2)
    intxn_ymax = min(ymax_1, ymax_2)
    intxn_xmax = min(xmax_1, xmax_2)

    if not (intxn_xmin >= intxn_xmax or intxn_ymin >= intxn_ymax):
        area_intxn = _cal_rect_area(intxn_ymin, intxn_xmin, intxn_ymax, intxn_xmax)

    return area_intxn
