
import torch
import math
# a = torch.tensor([16, 16, 0]).repeat(5, 1)
# print('a',a)
# print(a.size())
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
        (np.array(combine[force_flag]).reshape(8)))


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

import torch
from mmcv.ops import box_iou_rotated

# 假设 gt_bboxes 是一个列表
gt_bboxes = [torch.tensor([[572.9122, 119.6902,  42.9535,  13.4099,  -1.1384],
        [337.0000, 306.0000, 254.8686, 195.6579,  -1.1546],
        [632.5000, 416.0000, 236.4344, 172.8114,  -1.1564]]), 
        torch.tensor([[515.0000, 389.9999, 250.0000, 230.0000,  -1.5708]])]

gt_bboxess = torch.cat(gt_bboxes)

pseudo_bboxes = torch.tensor([[564.2621, 111.7765, 146.7731, 202.4169,   0.0000],
        [343.1044, 300.1440, 362.3149, 445.1118,   0.0000],
        [617.1939, 404.1903, 220.6112, 497.0878,   0.0000],
        [491.7470, 386.4711,  68.2218, 249.5302,   0.0000]])

iou1 = box_iou_rotated(pseudo_bboxes, gt_bboxess, aligned=True)
print(f'iou1:{iou1}')


import cv2
import numpy as np

# Helper function to convert (cx, cy, w, h, theta) to four corner points
def get_rotated_rect(cx, cy, w, h, angle_rad):
    rect = cv2.boxPoints(((cx, cy), (w, h), np.degrees(angle_rad)))
    return rect.astype(np.int32)

# 创建空白图像
height, width = 800, 800
image = np.zeros((height, width, 3), dtype=np.uint8)

# 绘制gt_bboxes中的旋转框（蓝色）
for gt_bbox in gt_bboxess:
    center, w, h, theta, score = np.split(gt_bbox.reshape(1,-1), (2, 3, 4, 5), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4,np.zeros((1,1))], axis=-1)
    polys = get_best_begin_point(polys)
    polys = np.array(polys.reshape(-1,2), dtype=np.int32)
#     rect_points = get_rotated_rect(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], gt_bbox[4])
    cv2.polylines(image, [polys.reshape(-1,2)], isClosed=True, color=(255, 0, 0), thickness=2)

# 绘制pseudo_bboxes中的旋转框（红色）
for pseudo_bbox in pseudo_bboxes:
    center, w, h, theta, score = np.split(gt_bbox.reshape(1,-1), (2, 3, 4, 5), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4,np.zeros((1,1))], axis=-1)
    polys = get_best_begin_point(polys)
    polys = np.array(polys.reshape(-1,2), dtype=np.int32)
#     rect_points = get_rotated_rect(pseudo_bbox[0], pseudo_bbox[1], pseudo_bbox[2], pseudo_bbox[3], pseudo_bbox[4])
    cv2.polylines(image, [polys.reshape(-1,2)], isClosed=True, color=(0, 0, 255), thickness=2)

# 显示图像
cv2.imshow('Rotated Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# # 假设 grids_xy_lvl 的形状是 [40000, 2]
# grids_xy_lvl = torch.randn(40000, 2)

# # 假设 gt_point 的形状是 [4, 2]
# gt_point = torch.randn(3, 2)

# # 使用广播计算距离
# points_gt_dist = torch.norm(grids_xy_lvl.unsqueeze(0) - gt_point.unsqueeze(1), dim=2)

# # 计算最小距离及其索引
# min_dist, min_dist_index = torch.min(points_gt_dist, dim=1)

# # 打印结果
# print("points_gt_dist shape:", points_gt_dist.shape)
# print("min_dist:", min_dist)
# print("min_dist_index:", min_dist_index)
# print("min_dist_index shape:", min_dist_index.shape)

        # [[2,2],[6,2],[10,2],...,[794,798],[798,798]]
        # [[4,4],[12,4],[20,4],...[788,796],[796,796]]
        # [[8,8],[24,8],[40,8],...[776,792],[792,792]]
        # [[16,16],[48,16],[80,16],...[752,784],[784,784]]

# compacted_bid_targets = torch.empty_like(bid)
# # 遍历索引并按照指定规则进行累积
# for i in range(len(bid)):
#     mask = (idx == i)
#     if mask.sum() == 0:  # 如果没有匹配的元素，将结果置零
#         compacted_bid_targets[i] = 0
#     else:  # 根据归约规则对匹配的元素进行归约
#         if 'mean' == 'mean':
#             compacted_bid_targets[i] = pos_bid_targets[mask].mean(dim=0)
# b_flp = (compacted_bid_targets % 1 > 0.5).sum() > 0  # tensor[],只要大于0则含有filp
# # Generate a mask to eliminate bboxes without correspondence
# _, bidx, bcnt = torch.unique(
#     compacted_bid_targets.long(),  # long舍弃小数取整
#     return_inverse=True,  # 返回bidx张量,表示每个元素在原始张量中第一次出现的索引,可用于重构原始张量
#     return_counts=True)   # 返回bcnt张量,表示每个唯一元素在原始张量中出现的次数
# bmsk = bcnt[bidx] == 2  # 取long之后可能出现bcnt==1的值,可能旋转填充后某个点只在一个视图上分配标签导致?
# # angle targets
# compacted_angle_targets = torch.empty_like(bid)
# for i in range(len(bid)):
#     mask = (idx == i)
#     if mask.sum() == 0:
#         compacted_angle_targets[i] = 0
#     else:  # 'mean'
#         compacted_angle_targets[i] = pos_angle_targets[:, 0][mask].mean(dim=0)
# compacted_angle_targets = compacted_angle_targets[bmsk].view(-1, 2)


# # 初始化累积结果的张量
# compacted_bid_targets = torch.zeros(len(bid))
# compacted_angle_targets = torch.zeros(len(bid), 2)

# # 遍历索引并按照指定规则进行累积
# for i in range(len(bid)):
#     mask = (idx == i)
#     if mask.sum() > 0:
#         compacted_bid_targets[i] = pos_bid_targets[mask].mean(dim=0)
#         compacted_angle_targets[i] = pos_angle_targets[:, 0][mask].mean(dim=0)

# # 计算flip标志
# b_flp = (compacted_bid_targets % 1 > 0.5).sum() > 0

# # 生成用于消除没有对应关系的边界框的掩码
# _, bidx, bcnt = torch.unique(compacted_bid_targets.long(), return_inverse=True, return_counts=True)
# bmsk = bcnt[bidx] == 2

# # 仅保留符合条件的角度目标
# compacted_angle_targets = compacted_angle_targets[bmsk].view(-1, 2)
