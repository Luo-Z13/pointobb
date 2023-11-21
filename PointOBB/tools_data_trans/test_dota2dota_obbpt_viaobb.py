import os
import cv2
import json
import numpy as np
from PIL import Image
import math


base_root = 'DOTAv10/data/split_ss_dota_1024_200/trainval/'
imageparent = os.path.join(base_root, 'images')
obb_anno_root = os.path.join(base_root, 'annfiles')

labelTxt_out_path = 'DOTAv10/data/split_ss_dota_1024_200/trainval/labelTxt_obb_pt_trainval_viaobb_v1.0'
label_name_list = os.listdir(obb_anno_root)

if not os.path.exists(labelTxt_out_path):
    os.makedirs(labelTxt_out_path)

size_range = 0.1  # 0/0.1/0.2

count = 0
for label_name in label_name_list:
    label_img_name = label_name.split('.')[0]+'.png'
    txtpath = os.path.join(obb_anno_root, label_name)

    with open(txtpath, 'r') as f_in:
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        boxes = []
        labelTxt_out_name = os.path.join(labelTxt_out_path, label_name)
        with open(labelTxt_out_name, 'w') as txt_file:

            for i, splitline in enumerate(splitlines):

                x1 = float(splitline[0])
                y1 = float(splitline[1])
                x2 = float(splitline[2])
                y2 = float(splitline[3])
                x3 = float(splitline[4])
                y3 = float(splitline[5])
                x4 = float(splitline[6])
                y4 = float(splitline[7])
                class_name = splitline[8]
                difficult = splitline[-1]

                x_min = np.min((x1,x2,x3,x4))
                y_min = np.min((y1,y2,y3,y4))
                x_max = np.max((x1,x2,x3,x4))
                y_max = np.max((y1,y2,y3,y4))

                # center pt
                cx = (x1 + x2 + x3 + x4) / 4
                cy = (y1 + y2 + y3 + y4) / 4
                w = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                h = math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)
    
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
                theta = np.random.uniform(0, 2 * np.pi)
                r = np.sqrt(np.random.uniform(0, 1))
                x_in_ellipse = r * sigma[0] * np.cos(theta)
                y_in_ellipse = r * sigma[1] * np.sin(theta)
                x, y = cx + x_in_ellipse, cy + y_in_ellipse
                # rot
                pt_x = (x - cx) * math.cos(a_r) - (y - cy) * math.sin(a_r) + cx
                pt_y = (x - cx) * math.sin(a_r) + (y - cy) * math.cos(a_r) + cy

                # print(f"random pt (x, y):({pt_x}, {pt_y})")
                    
                # # HBB + pt label
                # txt_file.write(f"{x_min} {y_min} {x_max} {y_min} {x_max} {y_max} {x_min} {y_max} {pt_x} {pt_y} {class_name} {difficult}\n")
                # OBB + pt label
                txt_file.write(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {pt_x} {pt_y} {class_name} {difficult}\n")

    txt_file.close()
    print(f'finish{label_name}')
print('done!')
