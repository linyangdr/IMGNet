# IMGNet: Image Mask-Guided Cross-modal Network for Radiology Report Generation

## Environment Setup

- `torch==1.8.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Dataset Setup

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`. You can apply the dataset [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) with your license of [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

+++

Before the training starts, it is necessary to **generate the masks** corresponding to the images.

Run `preprocess_mask\preprocess_mask.py` to train a model.

The generated masks are relatively large, so you need to wait for a while.

## Training

Run `train_rl.py` to train a model.

## Test

Run `bash test.py` to train a model.
