import os
import sys
from glob import glob
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

import collections as cl
import json

args = sys.argv

if len(args) == 2:
    JSON_FILE = args[1]
elif len(args) == 1:
    loss_log_dir = os.path.join(os.getcwd(), "loss_log/")
    target = os.path.join(loss_log_dir, '*.json')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    JSON_FILE = sorted(files, key=lambda files: files[1])[-1][0]


def ema_loss(data):
    data_len = len(data["train_loss"])

    ema_val_loss = []
    ema_train_loss = []

    ema_val_loss.append(data["val_loss"][0])
    ema_train_loss.append(data["train_loss"][0])
    ema_weight = 0.7

    for i in range(data_len):
        if i > 0:
            ema_train_loss.append((1-ema_weight) * data['train_loss'][i] + ema_weight * ema_train_loss[i-1])
            ema_val_loss.append((1-ema_weight) * data['val_loss'][i]   + ema_weight * ema_val_loss[i-1])

    fig,ax = plt.subplots()

    ax.set_title('loss and val_loss ')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xlim(0,data_len - 1)

    """
    ax.axvspan(0, 27.5, color="g", alpha=0.1)
    ax.axvspan(27.5, 50.5, color="b", alpha=0.1)
    ax.axvspan(50.5,data_len, color="r", alpha=0.1)
    """

    ax.plot(data["train_loss"], alpha=1, c='r', label='loss')
    ax.plot(data["val_loss"], c='b', alpha=0.2, label='val_loss')
    ax.plot(ema_val_loss, c='b', label='val_loss EMA')

    ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
    ax.grid()

    fig.patch.set_alpha(0)
    fig.savefig("result/loss/ema_loss.png",format="png",dpi=300)


def losses(data):
    data_len = len(data["train_loss"])

    plt.rcParams["font.size"] = 8
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False)

    axes[0,0].plot(data["train_loss"], lw=0.75, label='loss')
    axes[0,0].plot(data["val_loss"]  , lw=0.75, label='val_loss')

    axes[0,1].plot(data["rpn_bbox_loss"], lw=0.75,     label='rpn_bbox_loss')
    axes[0,1].plot(data["val_rpn_bbox_loss"], lw=0.75, label='val_rpn_bbox_loss')

    axes[0,2].plot(data["rpn_class_loss"],lw=0.75,      label='rpn_class_loss')
    axes[0,2].plot(data["val_rpn_class_loss"], lw=0.75, label='val_rpn_class_loss')

    axes[1,0].plot(data["mrcnn_mask_loss"], lw=0.75,     label='mrcnn_mask_loss')
    axes[1,0].plot(data["val_mrcnn_mask_loss"], lw=0.75, label='val_mrcnn_mask_loss')

    axes[1,1].plot(data["mrcnn_bbox_loss"], lw=0.75,     label='mrcnn_bbox_loss')
    axes[1,1].plot(data["val_mrcnn_bbox_loss"], lw=0.75, label='val_mrcnn_bbox_loss')

    axes[1,2].plot(data["mrcnn_class_loss"],  lw=0.75,    label='mrcnn_class_loss')
    axes[1,2].plot(data["val_mrcnn_class_loss"], lw=0.75, label='val_mrcnn_class_loss')


    axes[0,0].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=5)
    axes[0,1].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=5)
    axes[0,2].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=5)
    axes[1,0].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=5)
    axes[1,1].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=5)
    axes[1,2].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=5)

    axes[0,0].grid()
    axes[0,1].grid()
    axes[0,2].grid()
    axes[1,0].grid()
    axes[1,1].grid()
    axes[1,2].grid()

    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig("result/loss/losses.png",format="png",dpi=300)


def main():
    json_file = open(JSON_FILE, 'r')
    data = json.load(json_file)

    losses(data)
    ema_loss(data)


if __name__=='__main__':
    main()
