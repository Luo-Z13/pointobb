import copy

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ..builder import build_head
import copy
from torch.nn import functional as F
from ..builder import HEADS, build_loss
import math
from typing import Tuple, Union
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision import transforms


def resize_image(inputs, resize_ratio=0.5):
    down_inputs = F.interpolate(inputs, 
                                scale_factor=resize_ratio, 
                                mode='nearest')
    
    return down_inputs

def resize_proposal(img_metas, generate_proposals, gt_true_bboxes, gt_bboxes_ignore, ratio = 0.5):
    
    img_meta_out = copy.deepcopy(img_metas)
    generate_proposals_out = []
    gt_true_bboxes_out = []
    gt_bboxes_ignore_out = []
    for i in range(len(img_metas)):
        h, w, c = img_metas[i]['img_shape']
        img_meta_out[i]['img_shape'] = (math.ceil(h * ratio), math.ceil(w * ratio), c)
        img_meta_out[i]['pad_shape'] = (math.ceil(h * ratio), math.ceil(w * ratio), c)
        tmp_proposal = generate_proposals[i] * ratio
        generate_proposals_out.append(tmp_proposal)
        tmp_gt_true_bbox = gt_true_bboxes[i] * ratio
        gt_true_bboxes_out.append(tmp_gt_true_bbox)
        gt_bboxes_ignore_out.append(gt_bboxes_ignore[i]*ratio)
    return generate_proposals_out, gt_true_bboxes_out, img_meta_out, gt_bboxes_ignore_out

def resize_single_proposal(generate_proposals, ratio = 0.5):
    generate_proposals_out = []
    for i in range(len(generate_proposals)):
        tmp_proposal = generate_proposals[i] * ratio
        generate_proposals_out.append(tmp_proposal)

    return generate_proposals_out

def flip_tensor(tensor,
            img_shape: Tuple[int, int],
            direction: str = 'horizontal') -> None:
    """Flip boxes horizontally or vertically in-place.

    Args:
        img_shape (Tuple[int, int]): A tuple of image height and width.
        direction (str): Flip direction, options are "horizontal",
            "vertical" and "diagonal". Defaults to "horizontal"
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = tensor
    if direction == 'horizontal':
        flipped[..., 0] = img_shape[1] - flipped[..., 0]
        flipped[..., 4] = -flipped[..., 4]
    elif direction == 'vertical':
        flipped[..., 1] = img_shape[0] - flipped[..., 1]
        flipped[..., 4] = -flipped[..., 4]
    else:
        flipped[..., 0] = img_shape[1] - flipped[..., 0]
        flipped[..., 1] = img_shape[0] - flipped[..., 1]
    return flipped

def hboxlist2cxcywha(bbox_list):
    batch_bbox = []

    for i in range(len(bbox_list)):
        gt_box = bbox_list[i]
        # xyxy2cxcywha
        cx = (gt_box[:,0] + gt_box[:,2]) /2
        cy = (gt_box[:,1] + gt_box[:,3]) /2
        w = gt_box[:,2] - gt_box[:,0]
        h = gt_box[:,3] - gt_box[:,1]
        theta = torch.zeros_like(w, dtype=w.dtype)
        gt_box_new = torch.stack([cx, cy, w, h, theta], dim=-1)
        batch_bbox.append(gt_box_new)

    return batch_bbox


def merge_batch_list(batch_gt_bboxes, batch_proposals):
    merged_list = []
    flag = []

    for gt_bboxes, proposals in zip(batch_gt_bboxes, batch_proposals):
        merged_list.append(torch.cat([gt_bboxes, proposals], dim=0))
        flag.append([gt_bboxes.size(0), proposals.size(0)])

    return merged_list, flag

def split_batch_list(merged_list, flags):
    out_list1 = []
    out_list2 = []
    for merged_tensor, flag in zip(merged_list, flags):
        out_list1.append(merged_tensor[:flag[0]])
        out_list2.append(merged_tensor[flag[0]:])

    return out_list1, out_list2


# # 添加拼接角度维度并改为[cy,cy,w,h,a]格式
# for i, gen_proposals in enumerate(generate_proposals_0):
#     gen_proposals_xyxy = gen_proposals.reshape(
#         len(gt_bboxes[i]), -1, gen_proposals.size(-1))
#     gt_angle_expand = gt_bboxes[i][:,-1].unsqueeze(1).unsqueeze(1).expand_as(gen_proposals_xyxy[:,:,[-1]])
#     x1, y1, x2, y2 = gen_proposals_xyxy.split((1, 1, 1, 1), dim=-1)
#     gen_proposals_cxcywh = torch.cat([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)
#     generate_proposals_0[i] = torch.cat((gen_proposals_cxcywh, gt_angle_expand), 
#                                         dim=-1).view(generate_proposals_0[i].size(0), -1)


def regularize_boxes(boxes,
                     pattern: str = None,
                     width_longer: bool = True,
                     start_angle: float = -90) -> Tensor:
    """Regularize rotated boxes.

    Due to the angle periodicity, one rotated box can be represented in
    many different (x, y, w, h, t). To make each rotated box unique,
    ``regularize_boxes`` will take the remainder of the angle divided by
    180 degrees.

    However, after taking the remainder of the angle, there are still two
    representations for one rotate box. For example, (0, 0, 4, 5, 0.5) and
    (0, 0, 5, 4, 0.5 + pi/2) are the same areas in the image. To solve the
    problem, the code will swap edges w.r.t ``width_longer``:

    - width_longer=True: Make sure the width is longer than the height. If
        not, swap the width and height. The angle ranges in [start_angle,
        start_angle + 180). For the above example, the rotated box will be
        represented as (0, 0, 5, 4, 0.5 + pi/2).
    - width_longer=False: Make sure the angle is lower than
        start_angle+pi/2. If not, swap the width and height. The angle
        ranges in [start_angle, start_angle + 90). For the above example,
        the rotated box will be represented as (0, 0, 4, 5, 0.5).

    For convenience, three commonly used patterns are preset in
    ``regualrize_boxes``:

    - 'oc': OpenCV Definition. Has the same box representation as
        ``cv2.minAreaRect`` the angle ranges in [-90, 0). Equal to set
        width_longer=False and start_angle=-90.
    - 'le90': Long Edge Definition (90). the angle ranges in [-90, 90).
        The width is always longer than the height. Equal to set
        width_longer=True and start_angle=-90.
    - 'le135': Long Edge Definition (135). the angle ranges in [-45, 135).
        The width is always longer than the height. Equal to set
        width_longer=True and start_angle=-45.

    Args:
        pattern (str, Optional): Regularization pattern. Can only be 'oc',
            'le90', or 'le135'. Defaults to None.
        width_longer (bool): Whether to make sure width is larger than
            height. Defaults to True.
        start_angle (float): The starting angle of the box angle
            represented in degrees. Defaults to -90.

    Returns:
        Tensor: Regularized box tensor.
    """

    if pattern is not None:
        if pattern == 'oc':
            width_longer, start_angle = False, -90
        elif pattern == 'le90':
            width_longer, start_angle = True, -90
        elif pattern == 'le135':
            width_longer, start_angle = True, -45
        else:
            raise ValueError("pattern only can be 'oc', 'le90', and"
                                f"'le135', but get {pattern}.")
    start_angle = start_angle / 180 * np.pi

    x, y, w, h, t = boxes.unbind(dim=-1)
    if width_longer:
        # swap edge and angle if h >= w
        w_ = torch.where(w > h, w, h)
        h_ = torch.where(w > h, h, w)
        t = torch.where(w > h, t, t + np.pi / 2)
        t = ((t - start_angle) % np.pi) + start_angle
    else:
        # swap edge and angle if angle > pi/2
        t = ((t - start_angle) % np.pi)
        w_ = torch.where(t < np.pi / 2, w, h)
        h_ = torch.where(t < np.pi / 2, h, w)
        t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
    obb = torch.stack([x, y, w_, h_, t], dim=-1)
    return obb

import torch.distributed as dist
def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

import torch

from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_rotated'])

def box_iou_rotated(bboxes1: torch.Tensor,
                    bboxes2: torch.Tensor,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    .. note::
        The operator assumes:

        1) The positive direction along x axis is left -> right.

        2) The positive direction along y axis is top -> down.

        3) The w border is in parallel with x axis when angle = 0.

        However, there are 2 opposite definitions of the positive angular
        direction, clockwise (CW) and counter-clockwise (CCW). MMCV supports
        both definitions and uses CW by default.

        Please set ``clockwise=False`` if you are using the CCW definition.

        The coordinate system when ``clockwise`` is ``True`` (default)

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha+0.5h\\sin\\alpha
                \\\\
                y_{center}-0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}


        The coordinate system when ``clockwise`` is ``False``

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (-pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha-0.5h\\sin\\alpha
                \\\\
                y_{center}+0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}

    Args:
        boxes1 (torch.Tensor): rotated bboxes 1. It has shape (N, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        boxes2 (torch.Tensor): rotated bboxes 2. It has shape (M, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.
            `New in version 1.4.3.`

    Returns:
        torch.Tensor: Return the ious betweens boxes. If ``aligned`` is
        ``False``, the shape of ious is (N, M) else (N,).
    """
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros(rows * cols)
    if not clockwise:
        flip_mat = bboxes1.new_ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    ext_module.box_iou_rotated(
        bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious

def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))

def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def obb2poly_np(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_np_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_np_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_np_le90(rbboxes)
    else:
        raise NotImplementedError
    return results

def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le135(rrects):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys

def obb2xyxy(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    if version == 'oc':
        results = obb2xyxy_oc(rbboxes)
    elif version == 'le135':
        results = obb2xyxy_le135(rbboxes)
    elif version == 'le90':
        results = obb2xyxy_le90(rbboxes)
    else:
        raise NotImplementedError
    return results

def obb2xyxy_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    # pi/2 >= a > 0, so cos(a)>0, sin(a)>0
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)

def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()

def obb2xyxy_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    N = rotatex_boxes.shape[0]
    if N == 0:
        return rotatex_boxes.new_zeros((rotatex_boxes.size(0), 4))
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)

def obb2xyxy_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    # N = obboxes.shape[0]
    # if N == 0:
    #     return obboxes.new_zeros((obboxes.size(0), 4))
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center - bias, center + bias], dim=-1)