import random
import torch
import numpy as np
import os

def vp_normalize(in_p, pad_x, pad_y, width, height):
    if len(in_p) == 2:
        x0, y0 = in_p
        x0 = x0 + pad_x
        y0 = y0 + pad_y
        sx0 = round(x0 / width, 3)
        sy0 = round(y0 / height,3)
        return [sx0, sy0, -1, -1]
    elif len(in_p) == 4:
        x0, y0, w, h = in_p
        x0 = x0 + pad_x
        y0 = y0 + pad_y
        sx0 = round(x0 / width, 3)
        sy0 = round(y0 / height, 3)
        sx1 = round((x0 + w) / width, 3)
        sy1 = round((y0 + h) / height, 3)
        return [sx0, sy0, sx1, sy1]


def Transform_Visual_Prompts(vp, width, height):
    if height > width:
        pad_x0 = int((height - width) / 2)
        pad_y0 = 0
        width = height
    else:
        pad_x0 = 0
        pad_y0 = int((width - height) / 2)
        height = width

    vp_length = len(vp)
    if len(vp[0]) == 2:
        label = "point"
    elif len(vp[0]) == 4:
        label = "box"
    else:
        assert False, "vp length error"

    for i, item in enumerate(vp):
        norm_item = vp_normalize(item,pad_x0,pad_y0,width,height)
        vp[i] = norm_item
    if vp_length > 10:
        vp = vp[:10]
    else:
        if label == "point":
            while vp_length < 10:
                vp.append([0, 0, -1, -1])
                vp_length += 1
        else:
            while vp_length < 10:
                vp.append([0, 0, 0, 0])
                vp_length += 1

    sparse_vp_input = torch.tensor(vp)

    return sparse_vp_input


def noise_augmentation(bbox, delta_x_pos=5, delta_y_pos=5, delta_size=0.1):
    """
    """
    x, y, w, h = bbox

    noise_x = np.random.randint(-delta_x_pos, delta_x_pos + 1)
    noise_y = np.random.randint(-delta_y_pos, delta_y_pos + 1)

    noise_w = w * (1 + np.random.uniform(-delta_size, delta_size))
    noise_h = h * (1 + np.random.uniform(-delta_size, delta_size))

    x += noise_x
    y += noise_y
    w = max(0, noise_w)
    h = max(0, noise_h)

    return [x, y, w, h]