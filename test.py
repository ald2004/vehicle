from itertools import combinations

import numpy as np


def bb_intersection_over_union(boxA, boxB):
    boxA = (boxA[0], boxA[1], boxA[0] + boxA[2], boxA[3] + boxA[1])
    boxB = (boxB[0], boxB[1], boxB[0] + boxB[2], boxB[3] + boxB[1])
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def kill_duplicate_by_score(prediction, xou_thres=.7, score_thres=.3):
    prediction[:] = [x for x in prediction if float(x[1]) > score_thres]
    boxcombins = combinations(prediction, 2)
    print([x for x in boxcombins])
    for boxcomb in boxcombins:
        try:
            xou = bb_intersection_over_union(boxcomb[0][2], boxcomb[1][2])
            if xou > float(xou_thres):
                prediction.remove(boxcomb[1] if boxcomb[0][1] > boxcomb[1][1] else boxcomb[0])
        except:
            continue

    return prediction


def remove_repetition(boxes):
    """
    remove repetited boxes
    :param boxes: [N, 4]
    :return: keep:
    """
    _, x1_keep = np.unique(boxes[:, 0], return_index=True)
    _, x2_keep = np.unique(boxes[:, 2], return_index=True)
    _, y1_keep = np.unique(boxes[:, 1], return_index=True)
    _, y2_keep = np.unique(boxes[:, 3], return_index=True)

    x_keep = np.union1d(x1_keep, x2_keep)
    y_keep = np.union1d(y1_keep, y2_keep)
    return np.union1d(x_keep, y_keep)


# tdata = [('car', 0.5513402223587036, (332.39215087890625, 359.1761779785156, 566.8775634765625, 503.3076477050781))
#     , ('truck', 0.30799180269241333, (332.36285400390625, 344.0808410644531, 561.5469360351562, 496.47698974609375))]
#
# aa=np.array([(332.39215087890625, 359.1761779785156, 566.8775634765625, 503.3076477050781),
#             (332.36285400390625, 344.0808410644531, 561.5469360351562, 496.47698974609375)])
# sdata = kill_duplicate_by_score(tdata)
# print(remove_repetition(aa))


def convert_xminymin_xcenterycenter(h, w, xmin, ymin, xmax, ymax):
    # < x_center > < y_center > < width > < height > - float values relative to width and height of image, it can  be  equal from (0.0 to 1.0]
    dw = 1. / (float(w))
    dh = 1. / (float(h))
    x = (xmin + xmax) / 2.0

    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    #     return x, y, w, h
    return f'{x} {y} {w} {h}'


def convert_back_xcenterycenter(ph, pw, x, y, w, h):
    dw, dh = float(pw) * float(w), float(ph) * float(h)
    xmin = round((x * pw) - (dw / 2), 6)
    xmax = round((x * pw) + (dw / 2), 6)
    ymin = round((y * ph) - (dh / 2), 6)
    ymax = round((y * ph) + (dh / 2), 6)
    return f"{xmin} {ymin} {xmax} {ymax}"


def unitest():
    tdata = np.array([213, 34, 255, 50])
    tbdata = np.array([0.684211, 0.184211, 0.122807, 0.070175])
    h, w = 228, 342

    print(convert_xminymin_xcenterycenter(h, w, *tdata))
    # 0.684211 0.184211 0.122807 0.070175
    print(convert_back_xcenterycenter(h, w, *tbdata))


unitest()

# 1 0.684211 0.184211 0.122807 0.070175
# 0 0.682749 0.438596 0.230994 0.192982
# 0 0.494152 0.135965 0.081871 0.035088
# 1 0.222222 0.175439 0.192982 0.131579
# 0 0.336257 0.16886 0.052632 0.057018

# 00000000,pickup_truck,213,34,255,50
# 00000000,car,194,78,273,122
# 00000000,car,155,27,183,35
# 00000000,articulated_truck,43,25,109,55
# 00000000,car,106,32,124,45

