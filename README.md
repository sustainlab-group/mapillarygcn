# mapillarygcn
Repo for Predicting Livelihood Indicators from Community-Generated Street-Level Imagery (AAAI 2021).  

# Predicting Livelihood Indicators from Community-Generated Street-Level Imagery

This repository is the official implementation of [Predicting Livelihood Indicators from Community-Generated Street-Level Imagery](https://arxiv.org/abs/2006.08661). 

![Methods Graphic](https://drive.google.com/uc?export=view&id=16yNKOv9N830IJAz9hnQa92RIp83MXsbm)

## Requirements and Data

To install requirements:

```setup
pip install -r requirements.txt
```

First, download [data.csv](https://drive.google.com/file/d/1HgzZA55fQwUmSpmMJHXJwKsoUHZ4Zzaw/view?usp=sharing) (1.46 GB). 
There is a row for every image, where:
- `key`: refers to the unique ID of an image
- `country`: 'ia' for India, 'ke' for Kenya
- `unique_cluster`: is the unique ID of the cluster to which the image belongs (i.e. images in the same cluster will have the same `unique_cluster` value)
- `ilon`, `ilat`: represent the longitude, latitude of the image, respectively 
- `lon`, `lat`: represent the longitude, latitude of the cluster
- `features`: represent the object indexes detected in this image (more info on how these are generated below)
- `features_name`: represent the class names of the objects detected 
- `confidence`: confidence values of the detections 
-  `pov`, `pov_label`: represent the poverty indicator value and the binary class label (0 for less than median, 1 for greater than or equal to median), respectively 
- `pop`, `pop_label`: represent the population density indicator value and the binary class label, respectively
- `bmi`, `bmi_label`: represent the women's body-mass-index (BMI) indicator value and the binary class label, respectively (Note: As detailed in paper, BMI data is only available for India, so these values are 0 for all Kenya rows.)

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
To get image features, we resized the images to 224x224 and then trained CNNs on the classification task (ResNet34 with pretrained ImageNet weights; trained for 10 epochs with batch size 256 and lr 1e-3). To download the image features for use in the GCN, download [pretrained_image_features.zip](https://drive.google.com/file/d/1tYcegp9zYwFkV5Xgtgfq1-ytOGMTDt-Z/view?usp=sharing) (zip file 2.44 GB).

**Clusters**
For India, the clusters we used for training and validation are specified by `train_clusters_ia.txt` and `val_clusters_ia.txt`, respectively.
For Kenya, `train_clusters_ke.txt` and `val_clusters_ke.txt`.

## Training

**Image-wise Learning**
Once you have resized the images and specified their location in a column called `img_path_224x224` in `data.csv`, you can train a ResNet34 model to perform classification or regression. Here are the commands:

```train
cd models
python train_imagewise_classifier.py --model=resnet34 --label=pov_label --lr=1e-3 --batch_size=256 --pretrained
python imagewise_regressor.py --model=resnet34 --label=pov --lr=1e-4 --batch_size=256 --pretrained
```
Note: We use `train_imagewise_classifier.py` to generate the pretrained image features.

**Cluster-wise Learning**
To train the cluster-wise learning model from the paper, run this command:
```train
cd models
python clusterwise_classifier.py --label=pov_label --log_file=pov_label_results.log
python clusterwise_regressor.py --label=pov --log_file=pov_results.log
```

## Evaluation

To evaluate the image-wise learning model:

```eval
cd models
python eval_imagewise_classifier.py --model_weights=models/pov_classify --label=pov_label
python imagewise_regressor.py --save_name=models/reg_pov --label=pov --eval_mode 
```

To evaluate the cluster-wise learning model:
```eval
cd models
python clusterwise_classifier.py --label=pov_label --log_file=pov_label_results.log \
    --train_saved_feats=features/trainpov_label_feats.npy --train_saved_labels=features/trainpov_label_labels.npy \
    --val_saved_feats=features/valpov_label_feats.npy --val_saved_labels=features/valpov_label_labels.npy

python clusterwise_regressor.py --label=pov --log_file=pov_results.log \
    --train_saved_feats=features/trainpov_feats.npy --train_saved_labels=features/trainpov_labels.npy \
    --val_saved_feats=features/valpov_feats.npy --val_saved_labels=features/valpov_labels.npy
```

To run the baselines (which required no training):
```baseline
cd models
python baseline_nearestneighbor.py --baseline=random --label=pov_label
```

## Pre-trained Models

You can download the model weights for classifying and regressing on each indicator in each country. Each were trained on the images of the clusters in the training set using a ResNet34 model with learning rate 1e-3 for 10 epochs.

India
- Poverty: [Classification Model](https://drive.google.com/file/d/11ftmp0hHsnZHpRDkqAEdaMWC-WhDn-LM/view?usp=sharing), [Regression Model](https://drive.google.com/file/d/1c9Lyxhp3QZZsdd2GlcSDFNFv82TCLH0f/view?usp=sharing) 
- Population Density: [Classification Model](https://drive.google.com/file/d/1uDP1SC_mO2Sl7rSEUYchcoKTaSHQrBTz/view?usp=sharing), [Regression Model](https://drive.google.com/file/d/1lGH5GvxvDtsyHVO5vZaR8iESHzczqPC8/view?usp=sharing) 
- Women's BMI: [Classification Model](https://drive.google.com/file/d/1XR5wpy-OV3LbAdh74LXnqvGhJVcR-ev9/view?usp=sharing), [Regression Model](https://drive.google.com/file/d/1hlQrSA40uGdPoj7ddMbszNdy4CVaX1VB/view?usp=sharing) 

Kenya
- Poverty: [Classification Model](https://drive.google.com/file/d/11zTqcdM2ockIx3SDmyeT2Jli8ELTI5R7/view?usp=sharing), [Regression Model](https://drive.google.com/file/d/14ry6vMsNsyYX7nfGRRWqX2niPu_oYPuV/view?usp=sharing) 
- Population Density: [Classification Model](https://drive.google.com/file/d/1xF1-FwpVGpb5z9dGXxxTefHRAfdt9nky/view?usp=sharing), [Regression Model](https://drive.google.com/file/d/1LBxGswHnJOt-1slWbGtC54tOhaxNzswJ/view?usp=sharing) 

## GCN Train + Evaluation

After downloading the pre-trained model weights, you can train the cluster-wise GCN learning model from the paper by looking at the example commands below. You can specify flags for how you want to represent the nodes (V) or edges (A). You can set V to 'feats' to only represent images by their pretrained features, 'obj' for their object counts, and 'both' for the combination of the two. You can also set A to 'inv' for edges that represent the normalized inverse distance between images in a cluster or 'none' for random edges.

```train
cd models
python run_gcn.py --target pov_label --img_csv ./data.csv --train_val_dir . --pretrained_image_file pov_classify --A_type inv --V_type both --lr 1e-6 --batch_size 256 --num_iter 3000
python run_gcn.py --target pov --img_csv ./data.csv --train_val_dir . --pretrained_image_file pov_regress --A_type inv --V_type both --lr 1e-6 --batch_size 256 --num_iter 3000
```

## Pooled Models

For the pooled Random Forest and GCN models, we randomly sample 50% or 100% of the training clusters from each country and then evaluate on the validation clusters.

## Results

Our model achieves the following performance on :

### Livelihood Indicator Classification

India
| Model name               | Pov Accuracy    | Pop Accuracy   | BMI Accuracy   |
| -----------------------  |---------------- | -------------- | -------------- |
| Baseline (Random)        |     50.70%      |      50.11%    |       51.47%   |
| Baseline (Avg Neighbors) |     63.57%      |      69.06%    |       66.17%   |
| Image-wise Learning      |     74.34%      |      93.50%    |       85.28%   |
| Cluster-wise Learning    |     75.77%      |      91.71%    |       83.63%   |
| GCN (V: Obj Counts)      |     72.05%      |      86.63%    |       80.13%   |
| GCN (V: Img Feats)       |     81.06%      |    **94.71%**  |       89.42%   |
| GCN (V: Both)            |     80.91%      |      94.42%    |   **89.56%**   |
| Random Forest 50% Pooled |     74.20%      |      86.49%    |   N/A  |
| GCN 50% Pooled           |     80.99%      |      94.07%    |   N/A  |
| Random Forest 100% Pooled|     74.98%      |      89.78%    |   N/A  |
| GCN 100% Pooled          |     **81.34%**  |      94.50%    |   N/A  |

Kenya
| Model name               | Pov Accuracy    | Pop Accuracy   | 
| -----------------------  |---------------- | -------------- | 
| Image-wise Learning      |     73.71%      |      87.79%    | 
| Cluster-wise Learning    |     **77.34%**  |      89.59%    | 
| GCN (V: Obj Counts)      |     71.36%      |      88.26%    | 
| GCN (V: Img Feats)       |     76.06%      |      89.67%    | 
| GCN (V: Both)            |     75.59%      |      89.67%    |
| Random Forest 50% Pooled |     74.18%      |      81.69%    |
| GCN 50% Pooled           |     76.53%      |      89.20%    |
| Random Forest 100% Pooled|     74.18%      |      86.85%    |
| GCN 100% Pooled          |     77.055      |      90.14%    |

### Livelihood Indicator Regression

India
| Model name               | Pov r^2         | Pop r^2        | BMI r^2        |
| -----------------------  |---------------- | -------------- | -------------- |
| Baseline (Avg Neighbors) |     0.16        |      0.66      |       0.25     |
| Image-wise Learning      |     0.51        |      0.85      |       0.52     |
| Cluster-wise Learning    |     0.52        |      0.81      |       0.54     |
| GCN (V: Obj Counts)      |     0.39        |      0.86      |       0.38     |
| GCN (V: Img Feats)       |     **0.54**    |      0.82      |   **0.57**     |
| GCN (V: Both)            |     0.53        |     **0.89**   |       0.56     |
| Random Forest 50% Pooled |     0.45        |      0.67      |
| GCN 50% Pooled           |     0.52        |      0.81      |
| Random Forest 100% Pooled|     0.52        |      0.76      |
| GCN 100% Pooled          |     0.51        |      0.88      |

Kenya
| Model name               | Pov r^2         | Pop r^2         |
| -----------------------  |---------------- | -------------- |
| Image-wise Learning      |     0.39        |      **0.90**  |
| Cluster-wise Learning    |     **0.50**    |      0.81      |
| GCN (V: Obj Counts)      |     0.35        |      0.80      |
| GCN (V: Img Feats)       |     0.42        |      0.84      |
| GCN (V: Both)            |     0.40        |      0.85      |
| Random Forest 50% Pooled |     0.41        |      0.51      |
| GCN 50% Pooled           |     0.38        |      0.79      |
| Random Forest 100% Pooled|     0.47        |      0.58      |
| GCN 100% Pooled          |     0.36        |      0.72      |


## Interpretability

We produced our feature importance charts with [random-forest-importances](https://github.com/parrt/random-forest-importances) and tree visualizations with [dtreeviz](https://github.com/parrt/dtreeviz).


## Contributing
MIT License
