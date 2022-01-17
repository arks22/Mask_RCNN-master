import os
import sys

import numpy as np
import math
import seaborn
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt

import collections as cl
import json

args = sys.argv
file = args[1]

def moving_avg_loss(data):
    fig = plt.figure(num=1, clear=True)

    moving_epoch = 4
    moving_val_loss = np.zeros(1000)
    moving_train_loss = np.zeros(1000)

    for i in range(moving_epoch):
        moving_val_loss[i] = None
        moving_train_loss[i] = None

    for i in range(len(data["train_loss"])):
        if i >= moving_epoch:
            sum_train = 0
            sum_val   = 0
            for j in range(moving_epoch):
                sum_train += data["train_loss"][i - j]
                sum_val   += data["val_loss"][i - j]

            moving_train_loss[i] = sum_train / moving_epoch
            moving_val_loss[i] =   sum_val   / moving_epoch

    plt.title('loss and val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0,len(data["train_loss"]) - 1)

    plt.plot(data["train_loss"], alpha=0.2, c='r', label='loss')
    plt.plot(moving_train_loss, c='r', label='loss (4 epoch moving avg)')
    plt.plot(data["val_loss"], c='b', alpha=0.2, label='val_loss')
    plt.plot(moving_val_loss, c='b', label='val_loss (4 epoch moving avg)')

    plt.legend()
    plt.grid()

    fig.savefig("loss_log/moving_loss.png",format="png",dpi=300)


def loss(data):
    fig,ax = plt.subplots()

    ax.set_title('EarlyStopping(patience=20) loss and val_loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xlim(0,len(data["train_loss"]))

    ax.plot(data["train_loss"], color='r', label='loss')
    ax.plot(data["val_loss"], color='b', label='val_loss')

    ax.legend()
    ax.grid()

    fig.savefig("loss_log/earlystopping20.png",format="png",dpi=300)


def main():
    json_file = open(args[1], 'r')
    data = json.load(json_file)

    loss(data)
    #moving_avg_loss(data)


if __name__=='__main__':
    main()
