# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

from PIL import Image, ImageFile
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import pandas as pd
import os
import matplotlib.pyplot as plt


class Image_Dataset(torch.utils.data.Dataset):
    '''
    The Image_Dataset class inherits from the torch.utils.data.Dataset class.
    It loads all image in a given directory and assign labels to them according to their categories, then does custom
    transformation on the images. Finally it converts the images into torch tensors ready for training our models.

    Args:
    ----------
    URL: str
    The URL of the website to be scraped.
    driver: webdriver
    The browser used to load the webpage.
    data_dir: str
    The relative path of the directory in which the data will be stored.
    headless: bool
    When True, the script will run headlessly (to save GPU & CPU when scraping)
    '''
    def __init__(self,
                 product_file = 'raw_data/Products_cleaned.csv',
                 image_file = 'raw_data/Images.csv',
                 image_dir = 'raw_data/cleaned_images',
                 transform = None,
                 use_cuda = False):
        super().__init__()
        self.image_dir = image_dir
        self.cuda = use_cuda
        # Read data
        self.product_table = pd.read_csv(product_file, lineterminator = "\n")
        self.product_table['category'] = self.product_table['category'].str.split("/").str[0].str[:-1]  # Ignore sub-categories - intended?
        product_labels = self.product_table['category'].tolist() # labels for each product
        
        # Encoder & Decoder
        # self.category_count = len(set(self.product_labels))
        self.encoder = {category_name: label_number for (label_number, category_name) in enumerate(set(product_labels))}
        self.decoder = {label_number: category_name for (label_number, category_name) in enumerate(set(product_labels))}

        # Read images and match them to corresponding categories
        self.image_table = pd.read_csv(image_file, usecols = ['id', 'product_id'])
        self.match_images(save = True)

        # Augmentation of images
        if transform is None:
            self.transform = T.Compose([
                T.RandomPerspective(distortion_scale=0.2, p=0.4),
                T.RandomHorizontalFlip(p=0.7),
                T.PILToTensor() #if use ToTensor, dtype = float
            ])

    def __len__(self):
        return len(self.image_table)
    
    def __getitem__(self, idx):
        '''
        Returns the image as a torch.Tensor object, and its category (encoded to an integer).
        '''
        # print(idx)
        image_path = os.path.join(self.image_dir, f"{self.image_table.iloc[idx, 0]}.jpg")
        image = Image.open(image_path)
        image = self.transform(image)

        label = self.image_table['category'][idx] # Category this image is in
        label = self.encoder[label] # Encode to int

        if self.cuda:
            image = image.cuda() # Get our GPU working on training the model
        return image, label

    def match_images(self, save = False):
        '''
        Match each image to the corresponding products, get their category and save in a new 'category' column in self.image_table
        If save is True, will also save a csv file in raw_data folder
        '''
        self.image_table['category'] = [self.product_table.loc[self.product_table['id'] == x, 'category'].iloc[0] for x in self.image_table['product_id']]
        if save:
            self.image_table.to_csv('raw_data/Images_cleaned.csv', columns = ['id', 'product_id', 'category'])


if __name__ == '__main__':
    dataset = Image_Dataset(use_cuda = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)

    # For debugging purpose only: Print a sample image
    images, labels = next(iter(dataloader))
    index = 1
    print(images[index])
    print(labels[index])
    print(images[index].is_cuda)
    print(images[index].size())
    # Show it with matplotlib
    plt.imshow(images[index].cpu().permute(1,2,0))
    plt.title("Sample Image")
    plt.show()

    