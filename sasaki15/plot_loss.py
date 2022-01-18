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

def ema_loss(data):
    fig,ax = plt.subplots()

    moving_epoch = 5
    ema_val_loss = np.zeros(1000)
    ema_train_loss = np.zeros(1000)

    for i in range(moving_epoch):
        ema_val_loss[i] = None
        ema_train_loss[i] = None

    for i in range(len(data["train_loss"])):
        if i >= moving_epoch:
            sum_train = 0
            sum_val   = 0
            k = 0
            for j in range(moving_epoch):
                sum_train += 0.5 ** (j+1) * data["train_loss"][i - j]
                sum_val   += 0.5 ** (j+1) * data["val_loss"][i - j]
                k += 0.5**(j+1)

            ema_train_loss[i] = sum_train / k
            ema_val_loss[i]   = sum_val   / k

    ax.set_title('EMA loss and val_loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xlim(0,len(data["train_loss"]) - 1)

    ax.axvspan(0, 77, color="g", alpha=0.1)
    ax.axvspan(77, 121, color="b", alpha=0.1)
    ax.axvspan(121,len(data["train_loss"]), color="r", alpha=0.1)

    ax.plot(data["train_loss"], alpha=0.2, c='r', label='loss')
    ax.plot(ema_train_loss, c='r', label='loss (EMA)')
    ax.plot(data["val_loss"], c='b', alpha=0.2, label='val_loss')
    ax.plot(ema_val_loss, c='b', label='val_loss (EMA)')

    ax.legend()
    ax.grid()

    fig.savefig("loss_log/ema_loss.png",format="png",dpi=300)


def loss(data):
    fig,ax = plt.subplots()

    ax.set_title('EarlyStopping(patience=10) loss and val_loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xlim(0,len(data["train_loss"]))

    ax.axvspan(0, 15.5, color="g", alpha=0.1)
    ax.axvspan(15.5, 36.5, color="b", alpha=0.1)
    ax.axvspan(36.5,len(data["train_loss"]), color="r", alpha=0.1)

    ax.plot(data["train_loss"], color='r', label='loss')
    ax.plot(data["val_loss"], color='b', label='val_loss')

    ax.legend()
    ax.grid()

    fig.savefig("loss_log/earlystopping10.png",format="png",dpi=300)


def main():
    json_file = open(args[1], 'r')
    data = json.load(json_file)

    loss(data)
    #ema_loss(data)


if __name__=='__main__':
    main()
