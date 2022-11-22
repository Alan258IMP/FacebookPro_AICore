# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

from PIL import Image, ImageFile
import torch
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd
import os


class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 product_file = 'raw_data/Products_cleaned.csv',
                 image_file = 'raw_data/Images.csv',
                 image_dir = 'raw_data/cleaned_images',
                 transform = None):
        super().__init__()
        self.image_dir = image_dir
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
            self.transform = transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.3, p=0.7),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.PILToTensor() #if use ToTensor, dtype = float
            ])

    def __len__(self):
        return len(self.image_table)
    
    def __getitem__(self, index: int):
        '''
        Returns the image as a torch tensor and its category in the form
        '''
        # print(index)
        image_path = os.path.join(self.image_dir, f"{self.image_table.iloc[index, 0]}.jpg")
        image = Image.open(image_path)
        image = self.transform(image)
        # convert_tensor = transforms.PILToTensor()
        # image = convert_tensor(image)

        label = self.image_table['category'][index] # Category this image is in
        label = self.encoder[label] # Encode to int
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
    dataset = Image_Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)
    
    # For debugging purpose only: print 3 pictures in a batch
    images, labels = next(iter(dataloader))
    for i in range(3):
        print(images[i])
        print(labels[i])
        print(images[i].size())

    