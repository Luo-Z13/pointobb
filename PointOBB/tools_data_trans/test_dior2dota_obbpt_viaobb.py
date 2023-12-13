import os
import cv2
import json
import numpy as np
from PIL import Image
import math
import xml.etree.ElementTree as ET

obb_anno_root = 'DIOR/Annotations/Oriented Bounding Boxes/'
image_root = 'DIOR/JPEGImages-trainval/'

labelTxt_out_path = 'DIOR/labelTxt_obb_pt_trainval'
# labelTxt_out_path = 'DIOR/labelTxt_obb_pt_trainval-noise20'
# labelTxt_out_path = 'DIOR/labelTxt_obb_pt_trainval-noise0'
label_name_list = os.listdir(obb_anno_root)

# random range
size_range = 0.1  # 0/0.1/0.2

count = 0
for label_name in label_name_list:
    print(f'begin {label_name}')
    label_img_name = label_name.split('.')[0]+'.jpg'

    in_file = open(os.path.join(obb_anno_root, label_name), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    find = 0
    labeTxt_name = label_name.split('.')[0] + '.txt'
    labelTxt_out_name = os.path.join(labelTxt_out_path, labeTxt_name)
    with open(labelTxt_out_name, 'w') as txt_file:

        for obj in root.iter('object'):
            name = obj.find('name')
            class_name = name.text
            difficult = int(obj.find('difficult').text)
            robndbox = obj.find('robndbox')
            if robndbox is not None:
                x1 = float(robndbox.find('x_left_top').text)
                y1 = float(robndbox.find('y_left_top').text)
                x2 = float(robndbox.find('x_right_top').text)
                y2 = float(robndbox.find('y_right_top').text)
                x3 = float(robndbox.find('x_right_bottom').text)
                y3 = float(robndbox.find('y_right_bottom').text)
                x4 = float(robndbox.find('x_left_bottom').text)
                y4 = float(robndbox.find('y_left_bottom').text)

                x_min = np.min((x1,x2,x3,x4))
                y_min = np.min((y1,y2,y3,y4))
                x_max = np.max((x1,x2,x3,x4))
                y_max = np.max((y1,y2,y3,y4))

                # center pt
                cx = (x1 + x2 + x3 + x4) / 4
                cy = (y1 + y2 + y3 + y4) / 4
                w = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                h = math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)

                # angle = float(obj.find('angle').text)/180
                bboxpolys = np.array([x1, y1, x2, y2, x3, y3, x4, y4],dtype=np.float32).reshape((4, 2))
                rbbox = cv2.minAreaRect(bboxpolys)
                x_r, y_r, w_r, h_r, a_r = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
                while not 0 < a_r <= 90:
                    print(f'a_r:{a_r}')
                    if a_r == -90:
                        a_r += 180
                    else:
                        if a_r <= 0:
                            a_r += 90
                            w_r, h_r = h_r, w_r
                        if a_r > 90:
                            a_r -= 90
                            w_r, h_r = h_r, w_r

                a_r = a_r / 180 * np.pi
                assert 0 < a_r <= np.pi / 2
                sigma = (w_r * size_range, h_r * size_range) 

                # random point
                theta = np.random.uniform(0, 2 * np.pi)
                r = np.sqrt(np.random.uniform(0, 1))
                x_in_ellipse = r * sigma[0] * np.cos(theta)
                y_in_ellipse = r * sigma[1] * np.sin(theta)
                x, y = cx + x_in_ellipse, cy + y_in_ellipse

                # rot 
                pt_x = (x - cx) * math.cos(a_r) - (y - cy) * math.sin(a_r) + cx
                pt_y = (x - cx) * math.sin(a_r) + (y - cy) * math.cos(a_r) + cy

                # print(f"随机生成的点坐标 (x, y):({pt_x}, {pt_y})")

                # write
                class_name = class_name.lower()  # DIOR
                # HBB + pt label
                # txt_file.write(f"{x_min} {y_min} {x_max} {y_min} {x_max} {y_max} {x_min} {y_max} {pt_x} {pt_y} {class_name} {difficult}\n")
                # OBB + pt label
                txt_file.write(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {pt_x} {pt_y} {class_name} {difficult}\n")
                # OBB label
                # txt_file.write(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {class_name} {difficult}\n")

    txt_file.close()
    print(f'finish{label_name}')
print('done!')
