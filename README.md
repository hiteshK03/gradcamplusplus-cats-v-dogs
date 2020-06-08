# GradCam & GradCam++ Keras
## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Credits](#credits)

## Overview
This is the Keras implementation of GradCam & GradCam++ for explainability of Deep Learning model applied to the kaggle playground competition [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks(https://arxiv.org/abs/1710.11063) by Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian

![cat55](https://user-images.githubusercontent.com/45922320/84067097-8b202a00-a9e4-11ea-8d10-1dfbe533a4e0.png)
![dog196](https://user-images.githubusercontent.com/45922320/84067151-9ffcbd80-a9e4-11ea-97d5-4ab27806be8d.png)
![cat152](https://user-images.githubusercontent.com/45922320/84067132-983d1900-a9e4-11ea-9cb2-8b18776382bc.png)
![dog333](https://user-images.githubusercontent.com/45922320/84067173-a68b3500-a9e4-11ea-9c31-4668e076186a.png)

## Installation
Some of the important dependencies required are:
* python 3.5
* tensorflow
* Keras
* numpy
* pandas
* matplotlib
* CUDA (for using GPU)

## Dataset

The original dataset can be downloaded from the [kaggle competitions page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data).

After downloading it can be extracted at the current directory (same as that of the zip file downloaded above) by running:
```
python3 create_data.py
```
The structure of the working directory is shown below:
```bash
.
├── cats-v-dogs
│   ├── testing
│   │   └── test [12500 entries]
│   ├── train
│   │   ├── cats [11250 entries]
│   │   └── dogs [11250 entries]
│   └── validation
│       ├── cats [1250 entries]
│       └── dogs [1250 entries]
├── create_data.py
├── dogs-vs-cats-redux-kernels-edition.zip
└── sample_submission.csv
```
(or) For using with Google Colab the you can simply use the zipped working directory from [here](https://drive.google.com/file/d/1--Ejrtj8WyFQg_2Al_rPuaXAnrQgX-6w/view?usp=sharing)

For using this data, refer [jupyter notebook](dogs_vs_cats_gradcam.ipynb)

## Usage

For directly viewing the GradCam visualisations with the already pretrained model (can be downloaded from this [link](https://drive.google.com/file/d/1QfJpZOeaHuutEEdC6cyAQdxQGl65_PUh/view?usp=sharing)),
simply run ```python3 apply_gradcam.py -i <input-image-path> -m <pre-trained-model>```

To train and test the model, simply run ```python3 train.py```.
Once trained, you can view the gradcam visualisations with your trained model by running ```python3 apply_gradcam.py -i <input-image-path>```

(or) Simply run the [jupyter notebook](dogs_vs_cats_gradcam.ipynb) file in google colab.

## Credits
Grad-Cam codes for keras were adopted from [https://github.com/totti0223/gradcamplusplus](https://github.com/totti0223/gradcamplusplus)
