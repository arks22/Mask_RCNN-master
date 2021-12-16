# 太陽フィラメントを検出するMask R-CNN
:file_folder: **ando1** : 安藤さん実験1のスクリプト、データセット
:file_folder: **ando2** : 安藤さん実験2のスクリプト、データセット
:file_folder: **ando3** : 安藤さん実験3のスクリプト、データセット
:file_folder: **mrcnn** : Mask R-CNNモデル
:file_folder:
:file_folder:

実験ごとにデータセット、スクリプトを分割

## Directory structure example
:file_folder: ando1
 |- :file_folder: **dataset** データセット
 |- :file_folder: **logs** 学習により変更された重み、バイアスが保存されているh5ファイルの保存先
 |- :file_folder: **result** Predict画像の保存先
 |- :snake: **filament.py** 学習(train)及びテスト(evaluate)の実行スクリプト
 |- :snake: **inspect.py**  予測結果画像を出力するスクリプト
 |- :ledger: README.md 説明


# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](assets/street.png)

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset


The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). If you work on 3D vision, you might find our recently released [Matterport3D](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/) dataset useful as well.
This dataset was created from 3D-reconstructed spaces captured by our customers who agreed to make them publicly available for academic use. You can see more examples [here](https://matterport.com/gallery/).

