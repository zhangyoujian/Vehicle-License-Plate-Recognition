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
import execute.samples.licenseplate.licenseplate as licenseplate
import random

class InferenceConfig(licenseplate.LicensePlateConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def random_colors(N):
    colors = []
    for i in range(N):
        colors.append((random.randint(0,255), random.randint(0,255),random.randint(0,255)))

    return colors



class MaskRCNN():
    def __init__(self):
        ROOT_DIR = os.path.abspath("./")

        # Import licenseplate config
        sys.path.append(os.path.join(ROOT_DIR, "execute/samples/licenseplate/"))  # To find local version

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        LICENSEPLATE_MODEL_PATH = os.path.join(ROOT_DIR, "execute/samples/licenseplate/mask_rcnn_licenseplate.h5")

        config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(LICENSEPLATE_MODEL_PATH, by_name=True)
        self.model = model

        self.class_names = ['BG', 'licenseplate', 'balloon', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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


    def ReadImage(self,FileName):
        image = skimage.io.imread(FileName)
        return image


    def MaskRCNNDetect(self,image):
        results = self.model.detect([image], verbose=1)
        # Visualize results
        r = results[0]
        matplotlib.use('TkAgg')
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             self.class_names, r['scores'])

        return r['rois'], r['masks']



