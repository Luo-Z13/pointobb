
#!/bin/bash
conda activate openmmlab

export CUDA_VISIBLE_DEVICES=0,1

# DIOR
WORK_DIR='xxx/work_dir/pointobb_r50_fpn_2x_dior/'
tools/dist_train.sh --config configs2/pointobb/pointobb_r50_fpn_2x_dior.py 2 \
                    --work-dir ${WORK_DIR}\
                    --cfg-options evaluation.save_result_file=${WORK_DIR}'pseudo_obb_result.json'

# DOTA
# WORK_DIR="xxx/work_dir/pointobb_r50_fpn_2x_dota10/"
# tools/dist_train.sh configs2/pointobb/pointobb_r50_fpn_2x_dota10.py 2 \
                    # --work-dir ${WORK_DIR}\
                    # --cfg-options evaluation.save_result_file=${WORK_DIR}'pseudo_obb_result.json'
