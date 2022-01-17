import os
import sys

import numpy
import math
import seaborn
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt

import collections as cl
import json

args = sys.argv
file = args[1]

def loss(data):
     fig = plt.figure(num=1, clear=True)

     plt.title('loss and val_loss')
     plt.xlabel('epoch')
     plt.ylabel('loss')
     plt.xlim(0,len(data["train_loss"]))

     plt.plot(data["train_loss"], label='loss')
     plt.plot(data["val_loss"], color='y', label='val_loss')

     plt.legend()

     fig.savefig("loss_log/loss.png",format="png",dpi=300)


def main():
    json_file = open(args[1], 'r')
    data = json.load(json_file)

    loss(data)


if __name__=='__main__':
    main()
