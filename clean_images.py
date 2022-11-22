# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

from PIL import Image
from PIL import UnidentifiedImageError
import os
import shutil

def resize_image(final_size, im):
    '''
    Resize a given image to a given size and in RGB channels.
    This function is adapted from https://aicore-files.s3.amazonaws.com/MLOps/clean_images.py

    Parameters
    ----------
    final_size
        The size of the image after resizing.
    im
        The image to be resized.
    '''
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def make_save_directory(dir: str):
    '''
    Create a new directory to save the files generated.
    If the target directory already exists, ask the user if they want to overwrite all the files in the
    existing directory.
    '''
    if os.path.exists(dir):
        print('Overwrite the target directory? This will delete all files in the existing directory')
        ask = input('Press y to continue or any other keys to exit the program')
        if ask == 'y':
            shutil.rmtree(dir)
            os.mkdir(dir)
        else:
            raise Exception('The target directory already exists - please check the directory manually before proceeding.')
    else:
        os.mkdir(dir)


if __name__ == '__main__':
    final_size = 256 # Need to tweak this number to balance between quality of training & speed
    path = "raw_data/images/"
    image_dirs = os.listdir(path)
    save_dir = 'raw_data/cleaned_images'
    make_save_directory(save_dir)
    
    for n, item in enumerate(image_dirs, 1):
        try:
            im = Image.open('raw_data/images/' + item)
            new_im = resize_image(final_size, im)
            new_im.save(save_dir + f'/{item}') # Names will be unchanged
        except UnidentifiedImageError:
            print(f'Warning: {item} is not in a recognized format by PIL. Is this an image?')