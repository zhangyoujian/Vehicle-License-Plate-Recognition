import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import samples.licenseplate.licenseplate as licenseplate


ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import licenseplate config
sys.path.append(os.path.join(ROOT_DIR, "execute/samples/licenseplate/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
LICENSEPLATE_MODEL_PATH = os.path.join(ROOT_DIR, "execute/samples/licenseplate/mask_rcnn_licenseplate.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(LICENSEPLATE_MODEL_PATH):
    utils.download_trained_weights(LICENSEPLATE_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/licenseplate/Mydataset")



class InferenceConfig(licenseplate.LicensePlateConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(LICENSEPLATE_MODEL_PATH, by_name=True)

class_names = ['BG', 'licenseplate','balloon','person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

file_names = '浙A5299J.jpg'
image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names))


# showFile = '5951960966_d4e1cda5d0_z.jpg'
# image = skimage.io.imread(os.path.join(IMAGE_DIR, showFile))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
matplotlib.use('TkAgg')
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])










def main():
    print("Hello World")


if __name__=='__main__':
    main()