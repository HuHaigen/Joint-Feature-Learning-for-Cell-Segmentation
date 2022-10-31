# Joint Feature Learning for Cell Segmentation Based on Multi-scale Convolutional U-Net

<!-- > **The Paper Links**: [](), [Arxiv](). -->
> **Authors:** [Zhichao Jin](), [Haigen Hu](), [Qianwei Zhou](), [Qiu Quan](), [Xiaoxin Li](), [Qi Chen]()

## Abstract
A major challenge in the analysis of tissue imaging data is cell segmentation, the task of identifying precisely the boundary of each cell in a microscopic image. The cell seg- mentation task is still challenging due to the variable shapes, large size differences, uneven grayscale, and dense distribution of biological cells in microscopic images. In this paper, we propose a joint feature learning method that integrates the density and boundary branch into a multi-scale convolutional U-Net (MC- Unet). To enhance the supervision of cell density and boundary detection, the density and boundary loss is constructed to guide the joint learning of multiple features, where the density loss branch can address the challenges posed by high density, while the boundary loss branch can address the problems of unclear cell boundaries and partial cell occlusion. A series of experiments on different cell datasets show that two auxiliary branches improve the learning of features on cell density and cell boundaries and that the proposed method is effective on different segmentation models.

## Usage

You can build a runtime environment and prepare your dataset by following these steps：

- **Configuring your environment:**
  + Creating a virtual environment in terminal: `conda create -n {env_name} python=3.8`, and then run `conda activate env_name`.
  + Installing the pytorch environment: `conda install pytorch torchvision cudatoolkit=11.3 -c pytorch` 

Create a new folder `datasets`，then the directory structure is as follows:

```shell
./datasets/{dataset_name}
├── train
│   ├── corner      # cell boundary
│   ├── density     # cell density 
│   ├── image
│   └── label
└── val
    ├── corner
    ├── density
    ├── image
    └── label
```
You can modify the parameter settings in `/configs/{cfg_name}.yml`
```yaml
model: MCUNET
in_channels: 3
out_channels: 2
batch_size: 1
learning_rate: 0.001
epochs: 400
iters: 100000
patience: 20
...
```
Finally, run `python train.py configs/{cfg_name}.yml`, and then the training logs will be saved in `./work_dirs` folder by default.


<!-- ## 5. Citation
Please cite our paper if you find the work useful:  -->
