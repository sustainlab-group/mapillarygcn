# live_mapillary
Repo for Predicting Livelihood Indicators from Community-Generated Street-Level Imagery in India (NeurIPS 2020).  

# Predicting Livelihood Indicators from Community-Generated Street-Level Imagery in India

This repository is the official implementation of [Predicting Livelihood Indicators from Community-Generated Street-Level Imagery in India](link to preprint coming soon). 

![Methods Graphic](https://drive.google.com/uc?export=view&id=16yNKOv9N830IJAz9hnQa92RIp83MXsbm)

## Requirements and Data

To install requirements:

```setup
pip install -r requirements.txt
```

First, download [data.csv](https://drive.google.com/file/d/1HgzZA55fQwUmSpmMJHXJwKsoUHZ4Zzaw/view?usp=sharing) (1.46 GB). 
There is a row for every image, where:
- `key`: refers to the unique ID of an image
- `unique_cluster`: is the unique ID of the cluster to which the image belongs (i.e. images in the same cluster will have the same `unique_cluster` value)
- `ilon`, `ilat`: represent the longitude, latitude of the image, respectively 
- `features`: represent the object indexes detected in this image (more info on how these are generated below)
- `features_name`: represent the class names of the objects detected 
- `confidence`: confidence values of the detections 
-  `pov`, `pov_label`: represent the poverty indicator value and the binary class label (0 for less than median, 1 for greater than or equal to median), respectively 
- `pop`, `pop_label`: represent the population density indicator value and the binary class label, respectively
- `bmi`, `bmi_label`: represent the women's body-mass-index (BMI) indicator value and the binary class label, respectively

**Images**
Each Mapillary image is identifiable using its `key` and `unique_cluster`. 
To download the high-resolution images, run the script: 
```download
python download_images.py
```
If the script finishes running properly, the `data.csv` should now have a new column called `img_path` that provides the locations of the images (saved to directory `data/img_highres` by default).

**Object Counts**
We fed the high-res images directly into the [Seamless Scene Segmentation model](https://github.com/mapillary/seamseg) to get their segmentations and object detections. The class indexes, class names, and confidences of the detections per image are conveniently available in the csv as `features`, `features_name`, and `confidence`. 

**Image Features**
To get image features, we resized the images to 224x224 and then trained CNNs on the classification task (ResNet34 with pretrained ImageNet weights; trained for 10 epochs with batch size 256 and lr 1e-3). To download the image features for use in the GCN, download [pretrained_image_features.csv](https://drive.google.com/file/d/1tYcegp9zYwFkV5Xgtgfq1-ytOGMTDt-Z/view?usp=sharing) (zip file 2.44 GB).

**Clusters**
The clusters we used for training and validation are specified by `train_clusters_ia.txt` and `val_clusters_ia.txt`, respectively.

## Training

**Image-wise Learning**
Once you have resized the images and specified their location in a column called `img_path_224x224` in `data.csv`, you can train a ResNet34 model to perform classification or regression. Here are the commands:

```train
python imagewise_classify.py --model=resnet34 --label=pov_label --lr=1e-3 --batch_size=256 --pretrained
python imagewise_regress.py --model=resnet34 --label=pov --lr=1e-4 --batch_size=256 --pretrained
```
Note: We use `imagewise_classify.py` to generate the pretrained image features.

**Cluster-wise Learning**
To train the cluster-wise learning model from the paper, run this command:

**Cluster-wise GCN Learning**
To train the cluster-wise GCN learning model from the paper, seee the example commands below. You can specify flags for how you want to represent the nodes (V) or edges (A).

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate the image-wise learning model:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

To evaluate the cluster-wise learning model:

To evaluate the cluster-wise GCN learning model:

To run the baselines (which required no training):

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download the models that were pretrained on classifying each indicator below. Each were trained on the images of the clusters in the training set using a ResNet34 model with learning rate 1e-3 for 10 epochs.

- [Poverty Model](https://drive.google.com/mymodel.pth) 
- [Population Model](https://drive.google.com/mymodel.pth) 
- [Women's BMI Model](https://drive.google.com/mymodel.pth) 


## Results

Our model achieves the following performance on :

### Livelihood Indicator Classification

| Model name              | Pov Accuracy    | Pop Accuracy   | BMI Accuracy   |
| ----------------------- |---------------- | -------------- | -------------- |
| Baseline (Random)       |     85%         |      95%       |       x%       |
| Baseline (Avg Neighbors)|     85%         |      95%       |       x%       |
| Image-wise Learning     |     85%         |      95%       |       x%       |
| Cluster-wise Learning   |     85%         |      95%       |       x%       |
| GCN (V: Obj Counts)     |     85%         |      95%       |       x%       |
| GCN (V: Img Feats)      |     85%         |      95%       |       x%       |
| GCN (V: Both)           |     85%         |      95%       |       x%       |
| GCN (V: Both, A: Random)|     85%         |      95%       |       x%       |

### Livelihood Indicator Regression

| Model name              | Pov r^2        | Pop r^2         | BMI r^2        |
| ----------------------- |---------------- | -------------- | -------------- |
| Baseline (Avg Neighbors)|     85%         |      95%       |       x%       |
| Image-wise Learning     |     85%         |      95%       |       x%       |
| Cluster-wise Learning   |     85%         |      95%       |       x%       |
| GCN (V: Obj Counts)     |     85%         |      95%       |       x%       |
| GCN (V: Img Feats)      |     85%         |      95%       |       x%       |
| GCN (V: Both)           |     85%         |      95%       |       x%       |
| GCN (V: Both, A: Random)|     85%         |      95%       |       x%       |


## Interpretability

We procduced our feature importance with x and tree visualizations with y.


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
