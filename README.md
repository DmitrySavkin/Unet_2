## Installation using anaconda and pip

```bash
conda create -n coco_unet_tensorflow python=3.8
conda activate coco_unet_tensorflow
conda install tensorflow cudatoolkit=11
pip install -r requirements.txt # will skip tensorflow
```