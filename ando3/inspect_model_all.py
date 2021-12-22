import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

CURRENT_DIR = os.getcwd()
DEFAULT_DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:1"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import filament

args = sys.argv
#print("args:", args)
#sys.exit()
MODEL_DIR  =  os.path.join(CURRENT_DIR, "logs")
MODEL_PATH =  os.path.join(CURRENT_DIR, args[1])

config = filament.FilamentConfig()

#SAVE_DIR = "/home/maskrcnn/filament/result/predictions/"

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def main():
    config = InferenceConfig()
    config.display()
    # 
    dataset  = filament.FilamentDataset()
    dataset.load_coco(DEFAULT_DATASET_DIR,"val")
    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    weights_path = MODEL_PATH
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    for i in range(len(dataset.image_ids)):
        image_id = dataset.image_ids[i]
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                            dataset.image_reference(image_id)))
        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title=info["id"])
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        plt.savefig("/home/maskrcnn/filament/result/predictions/{}/prediction_{}.png".format(args[1], info["id"]))

main()
