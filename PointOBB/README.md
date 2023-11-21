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
     - `tools_data_trans/test_dior2dota_obbpt_viaobb.py`
     - `tools_data_trans/test_dota2dota_obbpt_viaobb.py`

3. **Generate COCO Format:**

   - Follow the following script to convert the dataset format:
     - `tools_data_trans/test_dota2coco_P2B_obb-pt.py`


# Train/Inference

1. **Train**

To train the model, follow these steps:

```bash
cd PointOBB
sh train_p_dist.sh
```

2. **Inference** (to generate pseudo obb label)
To inference (generate pseudo obb label), follow these steps:
```bash
cd PointOBB
# obtain COCO fmt pseudo label
sh test_p.sh
# generate DOTA fmt label
tools_cocorbox2dota.sh
```






