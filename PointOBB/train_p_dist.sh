
#!/bin/bash
conda activate openmmlab

export CUDA_VISIBLE_DEVICES=0,1

tools/dist_train.sh configs2/pointobb/pointobb_r50_fpn_2x_dior.py 2 \
                    --work-dir xxx/work_dir/pointobb_r50_fpn_2x_dior\
                    --cfg-options evaluation.save_result_file='xxx/work_dir/pointobb_r50_fpn_2x_dota10_dist/pseudo_obb_result.json'

# tools/dist_train.sh configs2/pointobb/pointobb_r50_fpn_2x_dota10.py 2 \
#                     --work-dir xxx/work_dir/pointobb_r50_fpn_2x_dota10_dist\
#                     --cfg-options evaluation.save_result_file='xxx/work_dir/pointobb_r50_fpn_2x_dota10_dist/pseudo_obb_result.json'
