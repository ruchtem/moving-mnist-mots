import math
import os
import sys
import json

import numpy as np
import pycocotools.mask as cocotools
from sacred import Experiment
from PIL import Image, ImageEnhance, ImageMath



###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
# taken from https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe
###########################################################################################

ex = Experiment("moving-mnist")
ex.add_config("config.yaml")

# helper functions
def arr_from_img(im, mean=0, std=1):
    '''

    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract

    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    '''
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(X, index, mean=0, std=1, norm=True):
    '''

    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    if norm:
        ret = (((X[index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    else:
        ret = X[index].reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


# loads mnist from web on demand
def load_dataset(training=True):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    if training:
        return load_mnist_images('train-images-idx3-ubyte.gz')
    return load_mnist_images('t10k-images-idx3-ubyte.gz')


def adjust_size(img, size, increase, config):
    # Check if we hit the size boundaries
    if increase and size >= config["size_band"][1]:
        increase = False
    elif not increase and size <= config["size_band"][0]:
        increase = True

    # Now change the size
    size = size * config["original_size"]
    delta = size * config["size_change"]
    new_size = int(np.round(size + delta if increase else size - delta, decimals=0))
    img = img.resize((new_size, new_size), Image.BICUBIC)
    return img, new_size / config["original_size"], increase


def generate_moving_mnist(config):
    '''

    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height

    '''
    mnist = load_dataset(config["training"])
    width, height = config["shape"]
    original_size = config["original_size"]
    num_frames = config["num_frames"]
    num_sequences = config["num_sequences"]
    digits_per_image = config["digits_per_image"]
    brightness_band = config["brightness_band"]
    size_band = config["size_band"]


    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_sequences, 1, width, height), dtype=np.uint8)
    
    # Store coco annotations
    annotations = []

    id = 0  # id for each annotation

    for img_idx in range(num_sequences):
        # Randomly generate direction, speed, velocity and brightness for all images
        direcs = np.pi * (np.random.rand(digits_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=digits_per_image) + 2
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
        brights = np.random.uniform(low=brightness_band[0], high=brightness_band[1], size=digits_per_image)
        sizes = np.random.uniform(low=size_band[0], high=size_band[1], size=digits_per_image)
        brightness_increases = np.random.choice([True, False], size=digits_per_image)   # True for increase brightness
        size_increases = np.random.choice([True, False], size=digits_per_image)

        # Get a list containing two PIL images randomly sampled from the database
        mnist_images = []
        for i, r in enumerate(np.random.randint(0, mnist.shape[0], digits_per_image)):
            img = Image.fromarray(get_image_from_array(mnist, r, mean=0))
            img = img.resize((original_size, original_size), Image.BICUBIC)
            mnist_images.append(img)
        
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(digits_per_image)])

        # Generate new frames for the entire num_frames
        for frame_idx in range(num_frames):
            image_id = img_idx * num_frames + frame_idx     # Id for coco annotations

            background = Image.new("L", (width, height))
            masks = [Image.new("1", (width, height)) for _ in range(digits_per_image)]

            # In canv (i.e Image object) place the image at the respective positions
            for i in range(digits_per_image):
                # Adjust size
                img, new_size, increases = adjust_size(mnist_images[i], sizes[i], size_increases[i], config)
                sizes[i] = new_size
                size_increases[i] = increases

                # Adjust image brightness first
                img_bright = ImageEnhance.Brightness(img).enhance(brights[i])

                # As mask we take the original image, so we have foreground and background
                background.paste(img_bright, tuple(positions[i].astype(int)), mask=img)

                # Save the ground truth mask
                mask = img.point(lambda p: p > config["seg_threshold"] and 255)
                masks[i].paste(mask, tuple(positions[i].astype(int)))
                
                # Correct overlay by foreground mask
                for j in range(i):
                    masks[j] = ImageMath.eval("a & ~b", a = masks[j], b=masks[i])
            
            # Now do the annotations
            for i in range(digits_per_image):
                id += 1
                mask = cocotools.encode(np.asarray(masks[i]).astype(np.uint8).T)
                mask["counts"] = mask["counts"].decode("utf-8")     # json is serializing it later
                area = int(cocotools.area(mask))                    # same here
                annotation = {"image_id": image_id,
                              "id": id,
                              "iscrowd": 0,
                              "segmentation": mask,
                              "area": area}
                annotations.append(annotation)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Adjust brightness
            for i, brightness in enumerate(brights):
                # Switch brightness change to reduce if it hit the maximum
                if brightness_increases[i] and brightness >= brightness_band[1]:
                    brightness_increases[i] = False
                # Same for minimum
                elif not brightness_increases[i] and brightness <= brightness_band[0]:
                    brightness_increases[i] = True
                
                # Now actually change the brightness
                delta = config["brightness_change"] * brightness
                brights[i] = brightness + delta if brightness_increases[i] else brightness - delta

            # Add the canvas to the dataset array
            dataset[image_id] = np.asarray(background).astype(np.uint8).T

    return dataset, annotations

@ex.automain
def main(config):
    np.random.seed(config["seed"])
    dat, annotations = generate_moving_mnist(config)
    dest = config["dest"]
    if config["filetype"] == 'npz':
        np.savez(dest, dat)
    elif config["filetype"] == 'jpg':
        for i in range(dat.shape[0]):
            Image.fromarray(get_image_from_array(dat, i, mean=0, norm=False)).save(os.path.join(dest, '{}.jpg'.format(i)))

    with open("annotations.json", "w") as f:
        json.dump(annotations, f)
