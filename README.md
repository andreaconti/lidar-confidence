# [Unsupervised confidence estimation on LiDAR depth maps and applications]()

LiDAR (Light Detection And Ranging) sensors are rapidly spreading in autonomous driving applications thanks to their long-range effectiveness and their dropping costs, leading  to  an  increasing  number  of  applications  exploiting  sparse  depth  data  comingfrom LiDAR devices.  However, when coupled with RGB cameras and projected over images, LiDAR depth maps expose several outliers due to noise and, more frequently, dis-occlusions between the sensor itself and the RGB cameras, resulting in large errors on the inputs to all the applications that process such depth maps.

This repository collects the code and the final models used to study in our [paper]() the issue of estimating the confidence of the LiDAR depthmaps by leveraging a deep learning framework. The dataset used is a subset of the KITTI dataset.

|        Image        |      Lidar Depth           |         Lidar Depth Filtered       |
| :-----------------: | :------------------------: | :--------------------------------: |
| ![image](https://github.com/andreaconti/lidar-confidence/blob/master/resources/teaser/image.png) | ![lidar](https://github.com/andreaconti/lidar-confidence/blob/master/resources/teaser/lidar.png) | ![lidar confidence](https://github.com/andreaconti/lidar-confidence/blob/master/resources/teaser/lidar_filtered.png) |

## Reproduce the experiments

To build a working environment use [conda](https://docs.conda.io/en/latest/) and
[dvc](https://dvc.org), they will take care of downloading the dependencies and
data:

```bash
$ # download the project
$ git clone https://github.com/andreaconti/lidar-confidence
$ cd lidar-confidence

$ # setup of the virtualenv, note that conda installs dvc
$ conda env create -f environment.yml
$ conda activate lidar-confidence
$ pip install -e .

$ # download the data
$ dvc pull  # or dvc fetch & dvc checkout

$ # now you have both data and code and you can run the experiments
$ dvc repro -R experiments
```

When you reproduce the training procedure the experiments will try to connect to
[wandb](https://wandb.ai) to log results, you can control and even disable wandb
entirely using environment variables listed
[here](https://docs.wandb.ai/library/environment-variables).

At the end of the experiments all the metrics will be logged in .json format in
the folder [results](https://github.com/andreaconti/lidar-confidence/tree/main/results)
anyway.

## Download a pretrained model

You can use [dvc](https://dvc.org) itself to download the pretrained models
directly from this project, see [here](https://dvc.org/doc/start/data-and-model-access).
Note that all the pre-trained models can be found
in [results](https://github.com/andreaconti/lidar-confidence/tree/main/results), for
instance you can download the pre-trained unsupervised model using:

```bash
$ dvc get https://github.com/andreaconti/lidar-confidence results/lce/model.pth
```

All models are in [TorchScript](https://pytorch.org/docs/stable/jit.html) format
thus you can simply download the file and load it as below, without the need of
the model source code:

```python
In [1]: !dvc get https://github.com/andreaconti/lidar-confidence results/lce/model.pth
In [2]: import torch
In [3]: model = torch.jit.load("model.pth")
```
