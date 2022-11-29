# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

import PIL
from clean_images import resize_image
from torchvision import transforms as T

def image_process(im):
    '''
    Take in an image and apply transformations needed to be fed into the model.
    Returns a torch tensor of shape (1, 3, 128, 128).
    
    Parameters
    ----------
    im: PIL.Image.Image or str
        The image to be processed in PIL image format or the path of the image.
    '''
    if isinstance(im, str):
        im = PIL.Image.open(im)
    elif isinstance(im, PIL.Image.Image):
        pass
    else:
        raise TypeError(f"Input type should be either PIL.Image.Image or str, but got {type(im)}")
    
    transform = T.Compose([
                T.RandomPerspective(distortion_scale=0.2, p=0.4),
                T.RandomHorizontalFlip(p=0.7),
                T.ToTensor(),
                ])

    new_im = resize_image(128, im)
    new_im = transform(im)
    # Add a dimension (batch size = 1) for single images before feeding to model
    new_im = new_im[None, :, :, :] 
    return new_im

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Read a sample image
    sampleimage_dir = 'raw_data/cleaned_images/00369118-eaaf-49ee-af6f-66b557f07140.jpg'
    image = image_process(sampleimage_dir)
    print(image.shape)
    plt.imshow(image.permute(0,2,3,1)[0,:,:,:])
    plt.title("Sample Image")
    plt.show()


