import math
import os
import sys
import json
import datetime
import shutil
import itertools
import argparse
import yaml
import copy

import numpy as np
import pycocotools.mask as cocotools
from sacred import Experiment
from PIL import Image, ImageEnhance, ImageMath
from itertools import groupby
from pathlib import Path


# Unique ids for each image and annotation. Required by coco.
GLOBAL_image_id = 0
GLOBAL_label_id = 0


def mask_to_poly(mask):
    """It seems coco is not using its own RLE format but the polygon (see instances_val2017.json)
    See also https://github.com/cocodataset/cocoapi/issues/131
    """
    from skimage import measure
    contours = measure.find_contours(mask, 0.5)

    segs = []
    for c in contours:
        c = np.flip(c, axis=1)
        segmentation = c.ravel().tolist()
        segs.append(segmentation)
    return segs


def get_categories_for_coco():
    """ Coco file contains a dedicated section for available categories
    """
    return [
        {"supercategory": "digit", "id": 0, "name": "0"},
        {"supercategory": "digit", "id": 1, "name": "1"},
        {"supercategory": "digit", "id": 2, "name": "2"},
        {"supercategory": "digit", "id": 3, "name": "3"},
        {"supercategory": "digit", "id": 4, "name": "4"},
        {"supercategory": "digit", "id": 5, "name": "5"},
        {"supercategory": "digit", "id": 6, "name": "6"},
        {"supercategory": "digit", "id": 7, "name": "7"},
        {"supercategory": "digit", "id": 8, "name": "8"},
        {"supercategory": "digit", "id": 9, "name": "9"}]


def clean_dir(path):
    if os.path.exists(path):
        for f in Path(path).iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                os.remove(f)


class Digit:
    """ 
    Handles loading digit image, moving the digit for the next 
    frame, mask generation, and creation of labels.
    """

    """Load the mnist dataset just once and then store here."""
    mnist_images = None
    mnist_labels = None

    def __init__(self, cfg, track_id):
        """
        Load random mnist image and store. Initialize size, brightness, velocity etc.
        """
        self.track_id = track_id
        self.train = cfg["training"]
        self.label_type = cfg['labels']['labeltype']
        self.mask_type = cfg['labels']['masktype'] if self.label_type == "coco" else None
        self.seg_threshold = cfg['seg_threshold']
        self.size_band = cfg['size_band']
        self.brightness_band = cfg['brightness_band']
        self.size_change = cfg['size_change']
        self.brightness_change = cfg['brightness_change']
        self.leave_prob = cfg['leave_probability'] if isinstance(cfg['digits_per_image'], list) else 0

        self._is_leaving = False
        self.image, self.category = Digit.load_random_image(self.train)

        self.current_size = self.image.size
        
        self.world_width, self.world_height = cfg["shape"]

        assert self.world_width - self.image.width > 20, "Shape not considerbly larger than digit image."
        # Make sure initial position is not too much at the border to not get messed
        # with increased size and moving direction
        init_border = int((self.world_width - self.image.width) * .2)
        self.position = np.array([np.random.rand() * (self.world_width - self.image.width - init_border),
                                  np.random.rand() * (self.world_height - self.image.height - init_border)])

        # Randomly generate direction, speed, velocity, brightness, and size for all images
        self.direction = np.pi * (np.random.rand() * 2 - 1)
        self.speed = np.random.randint(3, 5)
        self.velocity = np.array([self.speed * math.cos(self.direction), self.speed * math.sin(self.direction)])
        self.brightness = np.random.uniform(low=self.brightness_band[0], high=self.brightness_band[1])
        self.size = np.random.uniform(low=self.size_band[0], high=self.size_band[1])    # as factor of original size
        self.bright_increasing = np.random.choice([True, False])   # True for increase brightness
        self.size_increasing = np.random.choice([True, False])

        self.x_lim = (-2, self.world_width + 2)
        self.y_lim = (-2, self.world_width + 2)
        

    def move(self):
        """
        Updates position, size, and brightness. Only position change 
        is applied directly. This is because changes affecting the image
        change its quality when done frequently (like resizing). Therefore
        only chnge the resize factor and apply it every time on the original
        image. Same for brightness.
        """
        # Adjust size first, might already change the direction
        self._adjust_size()

        x_next, y_next = self.position + self.velocity

        if not self._is_leaving:
            # If we hit the border change the direction
            if x_next < self.x_lim[0] or x_next + self.current_size[0] > self.x_lim[1]:
                if np.random.rand() < self.leave_prob:
                    self._is_leaving = True
                else:
                    # We dont leave
                    self.velocity[0] = self.velocity[0] * -1
            elif y_next < self.y_lim[0] or y_next + self.current_size[1] > self.y_lim[1]:
                if np.random.rand() < self.leave_prob:
                    self._is_leaving = True
                else:
                    self.velocity[1] = self.velocity[1] * -1

        self.position = self.position + self.velocity

        self._adjust_brightness()


    def draw_and_mask(self, frame):
        """
        Applies all necessary changes to the digit image and then pastes it
        into the frame.

        Mask generation is handled here, because the mask must match the current size 
        of the digit.
        """
        # Apply resize
        self.current_size = tuple(int(np.round(x * self.size, decimals=0)) for x in self.image.size)
        img_resize = self.image.resize(self.current_size, Image.BICUBIC)

        # Apply brightness
        img_bright = ImageEnhance.Brightness(img_resize).enhance(self.brightness)

        # To override background and other digits, set mask = resized image
        frame.image.paste(img_bright, tuple(self.position.astype(int)), mask=img_resize)

        # Save the ground truth mask
        digit_mask = img_resize.point(lambda p: p > self.seg_threshold and 255)      # and 255 is needed for some reason
        world_mask = Image.new("1", (self.world_width, self.world_height))
        world_mask.paste(digit_mask, tuple(self.position.astype(int)))

        return world_mask


    def create_label(self, mask, image_id, frame_index=None):
        """ 
        Creates a label record for this digit for the current frame.
        """
        mask_compressed = cocotools.encode(np.asfortranarray(mask).astype(np.uint8))
        assert cocotools.area(mask_compressed) > 0, "Trying to write an empty mask"
        if self.label_type == "coco":
            global GLOBAL_label_id
            label_id = GLOBAL_label_id
            GLOBAL_label_id += 1

            if self.mask_type == "rle":
                mask = mask_compressed
                mask["counts"] = mask["counts"].decode("utf-8")             # json expects decoded data
            elif self.mask_type == "polygon":
                mask = mask_to_poly(np.asarray(mask))
            else:
                raise ValueError("Unknown mask format: '%s'. Choose either 'rle' or 'polygon'." % self.mask_type)
            
            return {"image_id": image_id,
                    "id": label_id,
                    "iscrowd": 0,
                    "segmentation": mask,
                    "area": int(cocotools.area(mask_compressed)),           # json is expecting native datatypes
                    "bbox": list(cocotools.toBbox(mask_compressed)),
                    "category_id": int(self.category)}

        else:
            # MOTS label type
            assert frame_index is not None, "Frame id required for mots label"
            return [frame_index, 
                    self.track_id, 
                    self.category, 
                    mask_compressed["size"][0], 
                    mask_compressed["size"][1], 
                    mask_compressed["counts"].decode("utf-8")]


    def _adjust_size(self):
        """ 
        Adjust the size of the digit depending whether its inreasing or not.
        Also checks if the maximum size is reached and changes to reduction/increase.
        """
        # Check if we hit the size boundaries
        if self.size_increasing and self.size >= self.size_band[1]:
            self.size_increasing = False
        elif not self.size_increasing and self.size <= self.size_band[0]:
            self.size_increasing = True

        # Now change the size
        delta = self.size * self.size_change
        self.size = self.size + delta if self.size_increasing else self.size - delta
        

    def _adjust_brightness(self):
        """
        Applying brightness is done in draw() because
        we need to keep the original image to not decrease quality.
        """
        # Check if we hit the maximum
        if self.bright_increasing and self.brightness >= self.brightness_band[1]:
            self.bright_increasing = False
        # Same for minimum
        elif not self.bright_increasing and self.brightness <= self.brightness_band[0]:
            self.bright_increasing = True
        else:
            pass    # We are inside the specified interval
        
        # Now actually change the brightness
        delta = self.brightness_change * self.brightness
        self.brightness = self.brightness + delta if self.bright_increasing else self.brightness - delta


    @staticmethod
    def load_random_image(train=True):
        """
        Picks a random image out of the mnist dataset.
        """
        if Digit.mnist_images is None or Digit.mnist_labels is None:
            Digit.mnist_images, Digit.mnist_labels = Digit.load_mnist(train)
        
        random_index =  np.random.randint(0, len(Digit.mnist_labels))
        # Images are transposed
        image = Image.fromarray(Digit.denormalize(np.squeeze(Digit.mnist_images[random_index]).T))
        category = Digit.mnist_labels[random_index]
        return image, category


    @staticmethod
    def denormalize(image, mean=0, std=1.):
        """
        mnist images are stored already normalized between [-1, 1].
        Denormalizing maps that back to raw pixel values between [0, 255].
        """
        normalized_image = (image + mean) * 255. * std
        return normalized_image.clip(0, 255).astype(np.uint8)


    @staticmethod
    def load_mnist(training=True):
        """
        Loads mnist from file or if not existing from web on demand
        """
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

        def load_mnist_lables(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
            return data

        if training:
            images = load_mnist_images('train-images-idx3-ubyte.gz')
            labels = load_mnist_lables("train-labels-idx1-ubyte.gz")
        else:
            images = load_mnist_images('t10k-images-idx3-ubyte.gz')
            labels = load_mnist_lables("t10k-labels-idx1-ubyte.gz")
        
        assert images.shape[0] == len(labels), "Size of loaded mnist images does not equal categories"
        return images, labels


class Frame:
    """
    Represents one frame in a sequence.
    """

    def __init__(self, cfg, digits, index):
        """
        Besides initialization also moves and draws the digits onto the
        frame image.

        Note that `digits` are modified by this class!

        """
        self.index = index
        self.train = cfg['training']
        self.shape = cfg['shape']   # width, height
        self.label_type = cfg['labels']['labeltype']
        self.digits_per_image = cfg['digits_per_image']
        self.enter_prob = cfg['enter_probability']

        self.image = Image.new("L", self.shape)

        global GLOBAL_image_id
        self.image_id = GLOBAL_image_id
        GLOBAL_image_id += 1
        
        for digit in digits:
            digit.move()

        self.masks = []
        for digit in digits:
            # First is the most background
            mask = digit.draw_and_mask(self)

            # Correct overlay by foreground mask
            for j, _ in enumerate(self.masks):
                self.masks[j] = ImageMath.eval("a & ~b", a = self.masks[j], b=mask)
            
            self.masks.append(mask)
        
        # Handle leaving digits/empty masks
        def mask_valid(mask):
            mask = cocotools.encode(np.asfortranarray(mask).astype(np.uint8))
            return cocotools.area(mask) > 0

        # Modify the original list as it is a reference to the one in the sequence
        keep_idx = [i for i, mask in enumerate(self.masks) if mask_valid(mask)]
        for i in reversed(range(len(digits))):
            if i not in keep_idx:
                del digits[i]
                del self.masks[i]

        # Copy current state of digits as the original is a reference
        # which could be modified elsewhere
        self._digits = copy.deepcopy(digits)


    def get_labels(self):
        """
        Returns all label records for this frame.
        """
        return [digit.create_label(mask, self.image_id, self.index) for digit, mask in zip(self._digits, self.masks)]


    def get_image_labels(self):
        """
        Only valid for coco label type. Returns an image record for coco.
        """
        assert self.label_type == "coco", "Image labels are only required for coco"
        return {"license": 1,
            "file_name": "%s.jpg" % self.image_id,
            "height": self.image.size[1],
            "width": self.image.size[0],
            "date_captured": datetime.datetime.now().strftime('%d-%m-%Y '),
            "id": self.image_id}


    def save_image(self, path):
        """
        Saves the frame image under the given path in the expected format.
        """
        img_name = "%s.jpg" % self.image_id if self.label_type == "coco" else "%06d.jpg" % self.index
        self.image.save(os.path.join(path, img_name))


class Sequence():

    def __init__(self, cfg, index):
        self.index = index
        self.lenght = cfg['num_frames']
        self.label_type = cfg['labels']['labeltype']
        self.out_dir = cfg['dest']
        self.name = cfg['name']

        if isinstance(cfg['digits_per_image'], list):
            # Digits can enter and leave in this sequence.
            # Class frame is responsible to make sure we stay inside this boundary
            num_digits = np.random.randint(low=cfg['digits_per_image'][0], high=cfg['digits_per_image'][1])
        elif isinstance(cfg['digits_per_image'], int):
            # We have the same digits per sequence
            num_digits = cfg['digits_per_image']
        else:
            raise AssertionError("Expecting digits_per_image to be int or list, not %s." % type(cfg['digits_per_image']))

        digits = [Digit(cfg, track_id=i) for i in range(num_digits)]
        last_track_id = num_digits
        
        self.frames = []

        for j in range(self.lenght):
            self.frames.append(Frame(cfg, digits, index=j))

            # Handle initialization of new digit
            if isinstance(cfg['digits_per_image'], list):
                # Initialize a digit with a certain probability
                if len(digits) < cfg['digits_per_image'][1] and np.random.rand() < cfg['enter_probability']:
                    digits.append(Digit(cfg, track_id=last_track_id + 1))
                    last_track_id += 1
                
                # Make sure we have always at least the minimum number of digits
                if len(digits) < cfg['digits_per_image'][0]:
                    for i in range(cfg['digits_per_image'][0] - len(digits)):
                        digits.append(Digit(cfg, track_id=last_track_id + 1))
                        last_track_id += 1


    def get_labels(self):
        labels = [frame.get_labels() for frame in self.frames]
        return [item for sublist in labels for item in sublist]     # flatten the list


    def get_image_labels(self):
        return [frame.get_image_labels() for frame in self.frames]


    def write_images(self):
        if self.label_type == "coco":
            path = os.path.join(self.out_dir, self.name)
        else:
            path = os.path.join(self.out_dir, self.name, "images", "%04d" % self.index)
        
        if not os.path.exists(path):
            os.makedirs(path)

        for frame in self.frames:
            frame.save_image(path=path)
    

    def write_labels(self):
        assert self.label_type == "mots", "Writing label files for each sequence only for MOTS"
        path = os.path.join(self.out_dir, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        labels = self.get_labels()
        with open(os.path.join(path, "%04d.txt" % self.index), "w") as f:
            for line in labels:
                f.write(" ".join(str(x) for x in line) + "\n")


ex = Experiment("moving-mnist")

@ex.automain
def main(config):
    np.random.seed(config["seed"])
    name = config['name']
    clean_dir(os.path.join(config['dest'], name))

    # Assumtion throughout the code that labeltye is mots if it isn't coco
    assert config['labels']['labeltype'] in ["coco", "mots"], \
            "Unknown labeltype: %s" % config['labels']['labeltype']

    # Create data
    sequences = []
    for i in range(config['num_sequences']):
        sequences.append(Sequence(config, index=i))

    # Write images
    for s in sequences:
        s.write_images()

    # Write labels
    if config['labels']['labeltype'] == "coco":
        image_labels = []
        labels = []
        for s in sequences:
            image_labels += s.get_image_labels()
            labels += s.get_labels()

        with open(os.path.join(config['dest'], "%s.json" % name), "w") as f:
            json.dump({"images": image_labels, 
                    "annotations": labels,
                    "categories": get_categories_for_coco()}, f)
    
    else:
        for s in sequences:
            s.write_labels()
    
    # Write copy of config
    with open(os.path.join(config['dest'], name, "%s_config.yaml" % name), 'w') as outfile:
        yaml.dump(copy.deepcopy(config), outfile, default_flow_style=True)
