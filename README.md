# FacebookPro_AICore
A python recreation of the system behind Facebook marketplace, which uses AI to recommend the most relevant listings based on a personalised search query.

This project requires [Pandas](https://github.com/pandas-dev/pandas), [Pillow](https://github.com/python-pillow/Pillow), [Sklearn](https://github.com/scikit-learn/scikit-learn) and [Pytorch](https://github.com/pytorch/pytorch.github.io).

## Milestone 1 ~ 3: Preliminary processing of the dataset

I downloaded the text and image dataset in .csv and .jpg format respectively. Then I performed some preliminary cleaning of the dataset.

To clean the tabular dataset I converted the prices of the products to np.float64 objects by removing the pound signs and the commas. To clean the image dataset, I made use of the Pillow library to resize all images to 256x256 size and standardized them to RGB mode.

## Milestone 4: Create a vision model

Firstly I created a dataset that feeds entries to the model. I created a class which inherits from torch.utils.data.Dataset...

In progress...

## Notes

1. If you have a Nvidia GPU on your device, I highly recommend installing [CUDA](https://developer.nvidia.com/cuda-downloads) to get your GPU working on the training of the vision model and speed the learning process up. See [Pytorch website](https://pytorch.org/get-started/locally/) for a full guide. Once CUDA is installed try out `train_GPU.py`.