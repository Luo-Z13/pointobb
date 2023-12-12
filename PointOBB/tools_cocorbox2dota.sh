#!/bin/bash
conda activate openmmlab

python docs/test_cocorbox2dota.py \
       --json_name xxx/work_dir/test_pointobb_r50_fpn_2x_dior/pseudo_obb_result_ann_1.json\
       --txt_root ../Dataset/DIOR/Annotations/pseudo_obb_labelTxt_dior_pointobb/

