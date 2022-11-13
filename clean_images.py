# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

from PIL import Image
import os

def resize_image(final_size, im):
    '''
    Resize a given image.
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



if __name__ == '__main__':
    path = "images/"
    dirs = os.listdir(path)
    final_size = 512
    for n, item in enumerate(dirs[:5], 1):
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{n}_resized.jpg')

