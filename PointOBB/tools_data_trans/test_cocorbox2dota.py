
import os
import cv2
import numpy as np
import torch
import math
from PIL import Image
import json
import argparse


def regularize_boxes(boxes,
                     pattern: str = None,
                     width_longer: bool = True,
                     start_angle: float = -90):
   
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
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
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
    return np.array(combine[force_flag]).reshape(8).tolist()

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
    polys = get_best_begin_point(polys.reshape(1,-1))
    return polys

def main(json_name, txt_root):
    if not os.path.exists(txt_root):
        os.mkdir(txt_root)

    with open(json_name, 'r') as json_file:
        data = json.load(json_file)

    for image_info in data["images"]:
        file_name = image_info["file_name"]
        image_id = image_info["id"]
        name = file_name.replace(".jpg", ".txt")  # DIOR

        with open(os.path.join(txt_root, name), 'w') as txt_file:
            for annotation in data["annotations"]:
                if annotation["image_id"] == image_id:
                    bbox = annotation["bbox"]
                    weight = 0
                    if "ann_weight" in annotation:
                        weight = annotation["ann_weight"]

                    if len(bbox) > 4:
                        cx, cy, w, h, theta = bbox

                        # filter
                        if w < 2 or h < 2:
                            continue

                        obb_ori = torch.tensor((cx, cy, w, h, theta))
                        obb = regularize_boxes(obb_ori, pattern='le90')
                        poly = obb2poly_np_le90(obb).reshape(-1)
                        x0, y0, x1, y1, x2, y2, x3, y3 = poly
                        category_id = annotation["category_id"]
                        category_name = next(item["name"] for item in data["categories"] if item["id"] == category_id)

                        if weight < 0.05:  # difficulty = 1
                            txt_file.write(f"{x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3} {category_name} 1\n")
                        else:             # difficulty = 0
                            txt_file.write(f"{x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3} {category_name} 0\n")
                    else:
                        print(f'bbox {bbox} is not an oriented bounding box!')

    print('done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON file to create TXT files.')
    parser.add_argument('--json_name', type=str, required=True, help='Path to the JSON file.')
    parser.add_argument('--txt_root', type=str, required=True, help='Root directory for saving TXT files.')

    args = parser.parse_args()
    main(args.json_name, args.txt_root)
