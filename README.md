# [Unsupervised confidence for LiDAR depth maps and applications](https://arxiv.org/pdf/2210.03118.pdf)

<p>
<div align="center">
    <a href="https://andreaconti.github.io">Andrea Conti</a>
    &middot;
    <a href="https://mattpoggi.github.io">Matteo Poggi</a>
    &middot;
    <a href="https://filippoaleotti.github.io/website">Filippo Aleotti</a>
    &middot;
    <a href="http://vision.deis.unibo.it/~smatt/Site/Home.html">Stefano Mattoccia</a>
</div>
<div align="center">
    <a href="https://arxiv.org/pdf/2210.03118.pdf">[Arxiv]</a>
    <a href="https://github.com/andreaconti/lidar-confidence/blob/master/torchhub-example.ipynb">[Demo]</a>
</div>
</p>

LiDAR (Light Detection And Ranging) sensors are rapidly spreading in autonomous driving applications thanks to their long-range effectiveness and their dropping costs, leading  to  an  increasing  number  of  applications  exploiting  sparse  depth  data  comingfrom LiDAR devices.  However, when coupled with RGB cameras and projected over images, LiDAR depth maps expose several outliers due to noise and, more frequently, dis-occlusions between the sensor itself and the RGB cameras, resulting in large errors on the inputs to all the applications that process such depth maps.

This repository collects the code and the final models used to study in our [paper](https://arxiv.org/pdf/2210.03118.pdf) the issue of estimating the confidence of the LiDAR depthmaps by leveraging a deep learning framework. The dataset used is a subset of the KITTI dataset.

|                                              Image                                               |                                           Lidar Depth                                            |                                                 Lidar Depth Filtered                                                 |
| :----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| ![image](https://github.com/andreaconti/lidar-confidence/blob/master/resources/teaser/image.png) | ![lidar](https://github.com/andreaconti/lidar-confidence/blob/master/resources/teaser/lidar.png) | ![lidar confidence](https://github.com/andreaconti/lidar-confidence/blob/master/resources/teaser/lidar_filtered.png) |

## Citation

```
@inproceedings{aconti2022lidarconf,
  title={Unsupervised confidence for LiDAR depth maps and applications},
  author={Conti, Andrea and Poggi, Matteo and Aleotti, Filippo and Mattoccia, Stefano},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems},
  note={IROS},
  year={2022}
}
```

## Reproduce the experiments

To build a working environment use [conda](https://docs.conda.io/en/latest/) and
[dvc](https://dvc.org), they will take care of downloading the dependencies and
data. If conda takes too much time too build the environment please try to use
[mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html).

```bash
$ # download the project
$ git clone https://github.com/andreaconti/lidar-confidence
$ cd lidar-confidence

$ # setup of the virtualenv, note that conda installs dvc
$ conda env create -f environment.yml
$ conda activate lidar-confidence
$ pip install -e .

$ # download the data
$ # NOTE: the first time you call dvc pull it will ask you to login on
$ #       google drive, asking to open your browser, it will require to
$ #       redirect to localhost:8080 to complete the procedure. So if
$ #       you are working on a remote machine please apply ssh port
$ #       forwarding like ssh -L 8080:localhost:8080 <username>@<machine>
$ dvc pull  # or dvc fetch & dvc checkout

$ # now you have both data and code and you can run the experiments
$ dvc repro -R experiments
```

When you reproduce the training procedure the experiments will try to connect to
[wandb](https://wandb.ai) to log results, you can control and even disable wandb
entirely using environment variables listed
[here](https://docs.wandb.ai/library/environment-variables).

At the end of the experiments all the metrics will be logged in .json format in
the folder [results](https://github.com/andreaconti/lidar-confidence/tree/master/results)
anyway.

## Download a pretrained model

You can use [dvc](https://dvc.org) itself to download the pretrained models
directly from this project, see [here](https://dvc.org/doc/start/data-and-model-access).
Note that all the pre-trained models can be found
in [results](https://github.com/andreaconti/lidar-confidence/tree/master/results), for
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

## TorchHub Interface

Finally, we provide a simple TorchHub interface to load the pretrained model and the 142 split dataset. A minimal example to reproduce results can be found [here](https://github.com/andreaconti/lidar-confidence/blob/master/torchhub-example.ipynb)
