Our GPUs: 2 * TeslaV100 (16GB)

# Prerequisites
install environment following
```shell script
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install pytorch
# conda install -c pytorch pytorch torchvision -y
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch
# install mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# install mmdetection
pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
chmod +x tools/dist_train.sh
```

```shell script
conda install scikit-image  # or pip install scikit-image
```


# Prepare Dataset DIOR/DOTA

1. **Download Dataset**

2. **Generate 'obb+pt' Format:**

   - Follow the following scripts to convert the dataset format:
     - `tools_data_trans/test_dior2dota_obbpt_viaobb.py` (for DIOR-R)
     - `tools_data_trans/test_dota2dota_obbpt_viaobb.py` (for DOTA)

3. **Generate COCO Format:**

   - Follow the following script to convert the dataset format:
     - `tools_data_trans/test_dota2coco_P2B_obb-pt.py`


# Train/Inference

1. **Train**

To train the model, follow these steps:

```bash
cd PointOBB
## train with single GPU, note adjust learning rate or batch size accordingly
# DIOR
python tools/train.py --config configs2/pointobb/pointobb_r50_fpn_2x_dior.py --work-dir xxx/work_dir/pointobb_r50_fpn_2x_dior --cfg-options evaluation.save_result_file='xxx/work_dir/pointobb_r50_fpn_2x_dior_dist/pseudo_obb_result.json'

# DOTA
# python tools/train.py --config configs2/pointobb/pointobb_r50_fpn_2x_dota10.py --work-dir xxx/work_dir/pointobb_r50_fpn_2x_dota --cfg-options evaluation.save_result_file='xxx/work_dir/pointobb_r50_fpn_2x_dota_dist/pseudo_obb_result.json'

## train with multiple GPUs
sh train_p_dist.sh
```

2. **Inference** 
  
To inference (generate pseudo obb label), follow these steps:
```bash
# obtain COCO format pseudo label for the training set 
# (在训练集上推理,从单点生成旋转框的伪标签)
sh test_p.sh
# convert COCO format to DOTA format 
# (将伪标签从COCO格式转换为DOTA格式)
sh tools_cocorbox2dota.sh
# train standard oriented object detectors 
# (使用伪标签训练mmrotate里的标准旋转检测器)
# Please use algorithms in mmrotate (https://github.com/open-mmlab/mmrotate)
```






