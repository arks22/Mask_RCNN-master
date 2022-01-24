"""
predict画像出力用のスクリプト
/logs/学習実行日時/ ディレクトリ中のh5ファイルを引数で指定

$ python3 inspect_model.py <model_path> <command> <img_id(optional)>
model_path = path/to/.h5file OR last
command = "one" OR "all"
"""

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
COMMAND    = args[2]

MODEL_DIR  = os.path.join(CURRENT_DIR, "logs")
config = filament.FilamentConfig()


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


def generate_pic(dataset, config, image_id):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,title=info["id"])

    dirname = os.path.basename(CURRENT_DIR)
    plt.savefig("{}/result/predictions/{}_prediction_{}.png".format(CURRENT_DIR,dirname,info["id"]))


if __name__ == '__main__':
    config = InferenceConfig()
    config.display()
    dataset  = filament.FilamentDataset()
    dataset.load_coco(DEFAULT_DATASET_DIR,"val")
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    if args[1] == "last":
        model_path = model.find_last()
    else:
        model_path = os.path.join(CURRENT_DIR, args[1])

    print("model filen name: {}".format(model_path))
    weights_path = model_path
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    if COMMAND == "one":
        image_date = args[3]
        for index, item in enumerate(dataset.image_info):
            if item["id"] == image_date:
                image_id = index
                break
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))
        generate_pic(dataset, config, image_id)

    elif COMMAND == "all":
        for i in range(len(dataset.image_ids)):
            image_id = dataset.image_ids[i]
            generate_pic(dataset, config, image_id)
