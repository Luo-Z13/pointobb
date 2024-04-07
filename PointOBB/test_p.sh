#!/bin/bash
conda activate mmdet20_2

export CUDA_VISIBLE_DEVICES=1

### Inference
## way1
python tools/train.py \
    --config configs2/pointobb/pointobb_r50_fpn_2x_dior.py \
    --work-dir xxx/work_dir/test_pointobb_r50_fpn_2x_dior/ \
    --cfg-options evaluation.save_result_file='xxx/work_dir/test_pointobb_r50_fpn_2x_dior/pseudo_obb_result.json', \
    evaluation.do_first_eval=True, \
    runner.max_epochs=0, \
    load_from='xxx/work_dir/epoch_12.pth'

## way2 
# Note: You need to uncomment the Inference section in configs2\pointobb\pointobb_r50_fpn_2x_dior.py and run the following command: 
# python tools/train.py\
#        --config configs2/pointobb/pointobb_r50_fpn_2x_dior.py \
#        --work-dir xxx/work_dir/test_pointobb_r50_fpn_2x_dior/ \
#        --cfg-options evaluation.save_result_file='xxx/work_dir/test_pointobb_r50_fpn_2x_dior/pseudo_obb_result.json'


### Transform Fmt
python exp/tools/result2ann_obb.py \
       ../Dataset/DIOR/Annotations/trainval_rbox_pt_P2Bfmt.json \
       xxx/work_dir/test_pointobb_r50_fpn_2x_dior/pseudo_obb_result.json \
       xxx/work_dir/test_pointobb_r50_fpn_2x_dior/pseudo_obb_result_ann_1.json
