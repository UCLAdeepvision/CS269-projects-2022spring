---
layout: post
comments: true
title: Explore wider usage of CLIP 
author: Yuyue Wang, Yufeng Li 
date: 2022-04-22
---


> Explore wider usage of CLIP, a large scale self-supervised models. We believe CLIP have more usage than what's shown in the original paper, as it has great feature extraction ability.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

# Motivition 

Imagine you are a machine learning engineer. One day, you want to add some new features for a classification task. For example you may need to do some fine grained classification on chopsticks like identify whether it is a Sushi chopstick, disposable chopstick or hot pot chopstick, but you do have any data. Then, one simple way to solve this problem is to do some data collection on your own, because your boss think find some new people to label it is too complex and time consuming. So, you followed the advice of your boss and then collect some chopstick data and train it. But things can happen over and over again, maybe today is chopsticks, tomorrow is soups and then your inference pipeline become more and more difficult to maintain. 

But could we just maintain some easily accessible data like image with language supervision and then train a model like clip instead of maintain lots of specialized model and data? For example, CLIP representation is very powerful and able to outperform many existing model with only CLIP image feature and linear classifier. Could we just maintain CLIP model and its training data and do simple ML classification instead of train deep model each time? 


# What we hope to demonstrate
Recently, there is a trend in industry to train large-scale self-supervised models. Such models utilize huge amount of unlabeled data to learn the intrinsic features, and in this way get more general and robust prediction result. CLIP is such a model consisting of an image encoder and a text encoder. In the forward stage, it calculates the loss based on the difference between the image/text feature vector pair with an image/text-description pair as input, and optimizes the two encoders' parameters simultaneously. Typical usage of the CLIP model includes using the image encoder as pretrained model to finetune, and formulizing the prediction task as text queries, together with the image feeding into CLIP to get a score vector. 

We believe CLIP have more usage than what's shown in the original paper, as it has great feature extraction ability.

First we'll compare CLIP with other models trained with general labeled dataset. We want to see if finetuning CLIP as pretrained model results in higher accuracy, fewer labeled data for a specific task.

Second, we'll use CLIP image encoder to extract feature vectors for a specific task/dataset, and unsupervised-learn to cluster the vectors. We'll examinate whether the cluster reflects the actual data label. If so, such clustering method can be used for auto-labeling and label imbalance elimination. 

Third, we'll also further explore the generality of CLIP. Whether its good features is sensitive to the specific data domain (whether similar data exists in original unlabeled training data or not)

# Related work 

## Tested model: 

The experiment is run on the following model, there are 3 type of model

Type 1: language supervised model
- CLIP model( VIT-32 ) [1]

![clip](team19/clip.png)

Type 2: model trained on open scource dataset (ImageNet)
- Vision Transformer: VIT-32 (base and large version), VIT-16 (base and large version) [2]
- New CNN: convnext_small, convnext_base [3]
- Traditional CNN: efficientnet_b4,efficientnet_b6 [4]

Type 3: Model with self-supervised pretraining
- Beit [5]
- ViTMAE [6]
- DINO [7]
- Data2vec [8]

## Summary of tested model


|    | model               |   year | Key technology                                    | Fine tune on imagenet                     | Good for representation learning    | pretrain dataset                           | dataset reference                                                                                                                                                                                                                                                                                 |
|---:|:--------------------|-------:|:--------------------------------------------------|:------------------------------------------|:------------------------------------|:-------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | CLIP                |   2021 | Contrastive learning, vision-language supervision |                                           | Y                                   | 400 million (image,text) pairs from openai |                                                                                                                                                                                                                                                                                                   |
|  1 | Data2vec            |   2022 | teacher-student Mask                              | 86.6 (L) 84.2(B)                          |                                     | ImageNet 1k                                | Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.                                                                                                                                                        |
|  2 | DINO                |   2021 | teacher student                                   | 82.8 (B)                                  | Y                                   | ImageNet                                   | Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. IJCV, 2015.                                                |
|  3 | Beit                |   2021 | dVAE mask                                         | 83.4 (B) 86.3(L)                          |                                     | training set of ImageNet 1k (1.2 M)        | Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.                                                                                                                                                        |
|  4 | MAE                 |   2021 | Mask                                              | 83.6 (B) 85.9 (L)                         |                                     | ImageNet-1K                                | Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.                                                                                                                                                        |
|  5 | vit_b_16 & vit_b_32 |   2021 | VIT, pretrain                                     | 84.15 (16 B JFT300M) 80.73 (32 B JFT300M) |                                     | JFT300M                                    | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. preprint arXiv:2010.11929, 2020. |
|  6 | convnext            |   2022 | CNN modification                                  | 85.1 (B) 85.5 (L)                         |                                     | ImageNet-22K.                              |                                                                                                                                                                                                                                                                                                   |
|  7 | EfficientNet        |   2020 | NAS                                               | 80.1 (B6) 84.0 (B6)                       |                                     |                                            |                                                                                                                                                                                                                                                                                                   |


## Dataset for general classification

There are two type of dataset we will test:

Dataset similar to CLIP pretraining data
- Flowers102: 102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Some examples are shown below. 


<img src="team19/Dataset_Flowers102.png" width="400"/>

- FGVCAircraft: The dataset contains 10,200 images of aircraft, with 100 images for each of 102 different aircraft model variants, most of which are airplanes. 

<img src="team19/Dataset_FGVCAircraft.png" width="400"/>


Dataset different from CLIP pretraining data
- PatternNet(torch geo): Dataset from torchgeo package [9] The PatternNet dataset is a dataset for remote sensing scene classification and image retrieval. It have 38 scene classes, 800 images per class. 


<img src="team19/PatternNet_Dataset.png" width="400"/>

- UCMerced(torch geo): Dataset from torchgeo package [9]. The UC Merced dataset is a land use classification dataset of 2.1k 256x256 1ft resolution RGB images of urban locations around the U.S. extracted from the USGS National Map Urban Area Imagery collection with 21 land use classes (100 images per class).

<img src="team19/UCMerced_Dataset.png" width="400"/>

- ISIC [10]: Data from SIIM-ISIC Melanoma Classification 2020. This competition aim to predicting a binary target for each image, 0 denotes benign, and 1 indicates malignant.

<img src="team19/ISIC.png" width="400"/>


## Dataset for data drift

<!-- - rxrx1dataset (wilds) -->
- iwildcamdataset (wilds): Dataset from wilds [11] for animal classification. The input x is a photo from a camera trap, the label y is one of 182 animal species, and the domain d specifies the identity of the camera trap. The training, validation and in-distribution data contain the images from different camera, but out-of-distribution test data are not capture by the camera used in in-distribution camera.

<img src="team19/iwildcam_dataset.jpg" width="400"/>


- fmowdataset (wilds): Dataset from wilds [11] for satellite image classification. As satellite data constantly changes due to human activity and environmental processes, these models must be robust to distribution shifts over time. The input x is an RGB satellite image, the label y is one of 62 building or land use. The in-distribution data comprises data from before 2013, while the out-of-distribution test set comprises data from 2016 and after.

<img src="team19/fmow_dataset.jpg" width="400"/>


# Experiment: 

### Classification experiment on general dataset

The experiment is to test the performance of different models on the general dataset mentioned before. In this part, two types of experiment are conducted:

Classification with feature and simple ML algorithm:
 - This experiment aim to test the respresentation performance of each model. For each dataset, extract feature from last layer of the network and perform KNN and logistic regression. 

Fine tuning on each dataset:
 - This aim to test each model performance after fine-tuning. For each dataset, fine tune each network on each dataset.

The evaluation metric for this experiment is accuracy. In this case, we want to test the performance of each model on each general dataset with respect to different amount of training and validation data. The amount of training and validation data increased in order of 10 from 10 samples per class to 1000 samples per class. But for general dataset, not all dataset have enough data. Thus, we detailed designed the training, validation and testing split for dataset in following table. The number listed in (10)train:val:test, (100)train:val:test and (1000)train:val:test means the number of data used in training, validation and testing dataset. In simple ML algorithm, we only use (10)train:val:test and (100)train:val:test for training and testing. 

|    | dataset              |   total data |   Number of labels | (10)train:val:test   | (100)train:val:test   | (1000)train:val:test   |
|---:|:---------------------|-------------:|-------------------:|:---------------------|:----------------------|:-----------------------|
|  0 | ISIC                 |         1150 |                  2 | 14:4:1132            | 148:50:952            | 688:230:232            |
|  1 | PatternNet_Dataset   |        30400 |                 38 | 266:76:30058         | 2812:950:26638        | 18202:6080:6118        |
|  2 | UCMerced_Dataset     |         1260 |                 21 | 147:42:1071          | 567:189:504           | 567:189:504            |
|  3 | Dataset_FGVCAircraft |         3334 |                100 | 700:200:2434         | 1900:600:834          | 1900:600:834           |
|  4 | Dataset_DTD          |         1880 |                 47 | 329:94:1457          | 1081:376:423          | 1081:376:423           |
|  5 | Dataset_Flowers102   |         1020 |                102 | 510:204:306          | 510:204:306           | 510:204:306            |

From the table, Dataset_Flowers102 only have one type of split way. UCMerced_Dataset, Dataset_FGVCAircraft and Dataset_DTD have two type of split way. This is because these dataset have not enought data for each label (e.g. Dataset_Flowers102 on have 10 data for each label, Dataset_DTD have 40 samples per label). Thus, in this experiment, we just repeat the the last experiment for Dataset_Flowers102, UCMerced_Dataset, Dataset_FGVCAircraft and Dataset_DTD for those experiment that do not have enough labels.

### Classification experiment on Data Drift dataset


Same as before, the experiment is to test the performance of different models on the data drift dataset. In this case, the model is trained on the in-distribution data and tested on both in-distribution and out-of-distribution data. 

Classification with feature and simple ML algorithm:
 - This experiment aim to compare the model representation performance in terms of data drift. For each dataset, extract feature from last layer of the network and perform KNN and logistic regression. 

Fine tuning on each dataset:
 - This aim to test the model sensitivity in terms of fine tuning for data drift. For each dataset, fine tune each network on each dataset.

The evaluation metric for this experiment is still accuracy on in-distribution test set and out-of-distribution test set. Same as previous experiment, we split training data in order of 10 that start from 10 samples and more detailed split is listed below. In simple ML algorithm, we only use (10)train:val and (100)train:val for training and testing. 

|    | dataset         |   total training data |   total val data |   total id test data |   total ood test data |   Number of labels | (10)train:val   | (100)train:val   | (1000)train:val   |
|---:|:----------------|----------------------:|-----------------:|---------------------:|----------------------:|-------------------:|:----------------|:-----------------|:------------------|
|  0 | FMoWDataset     |                 76863 |            19915 |                11327 |                 22108 |                 62 | 434:124         | 4517:1527        | 40150:12721       |
|  1 | IWildCamDataset |                129809 |            14961 |                 8154 |                 42791 |                 87 | 1090:149        | 6694:1394        | 33628:7079        |



### Clustering experiment 

In clustering experiment, we firstly extract feature from each model. The feature extraction method is same as the one used in logistic regression and KNN experiment. To have better clustering performance, we firstly do dimension reduction with UMAP with euclidean metric. Then, we use k-means clustering to cluster the data. In this case, we perform clustering on all dataset in general dataset and test data in data drift dataset. The evaluation metric for this experiment is adjusted rand score, which can be considered as accurcy measure in clustering task.

### Training and preprocessing

The training pipeline is built based on pytorch and pytorch-lightning. Each training samples are cropped and normalized to the 244x244 with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

The GPU used for fine tuning is NVIDIA GeForce RTX 3070 laptop GPU. The batch size for training is 16 and optimizer is Adam. The learning rate is 1e-5. The number of epochs depend on early stopping. The early stopping is used to stop training when the validation loss does not decrease for 30 epoch.


# Results


## Result for classification experiment on general dataset

The result for training representation on KNN and logistic regression is shown in the following tables. The row is ordered by accuracy of logistic regression at 100. 

### knn_10
|    | Dataset_Flowers102    | UCMerced_Dataset      | ISIC                  | Dataset_DTD           | PatternNet_Dataset    | Dataset_FGVCAircraft   |
|---:|:----------------------|:----------------------|:----------------------|:----------------------|:----------------------|:-----------------------|
|  0 | CLIP 0.803            | DINO 0.781            | DINO 0.650            | DINO 0.523            | CLIP 0.864            | DINO 0.202             |
|  1 | DINO 0.826            | vit_l_32 0.646        | vit_b_32 0.598        | CLIP 0.493            | DINO 0.863            | vit_l_16 0.122         |
|  2 | vit_b_16 0.568        | vit_b_32 0.675        | ViTMAE 0.720          | convnext_base 0.355   | vit_l_16 0.729        | vit_b_16 0.133         |
|  3 | vit_l_16 0.513        | vit_l_16 0.587        | vit_l_16 0.570        | vit_b_32 0.411        | convnext_tiny 0.787   | convnext_tiny 0.111    |
|  4 | vit_l_32 0.486        | CLIP 0.787            | convnext_base 0.534   | vit_b_16 0.387        | vit_b_16 0.800        | vit_l_32 0.131         |
|  5 | convnext_base 0.447   | convnext_base 0.651   | vit_b_16 0.605        | convnext_tiny 0.350   | vit_l_32 0.794        | CLIP 0.235             |
|  6 | convnext_tiny 0.526   | convnext_tiny 0.613   | convnext_tiny 0.571   | vit_l_16 0.323        | vit_b_32 0.786        | vit_b_32 0.124         |
|  7 | vit_b_32 0.532        | vit_b_16 0.714        | vit_l_32 0.553        | vit_l_32 0.310        | convnext_base 0.709   | convnext_base 0.078    |
|  8 | ViTMAE 0.179          | ViTMAE 0.348          | data2vec 0.691        | ViTMAE 0.117          | ViTMAE 0.474          | ViTMAE 0.024           |
|  9 | Beit 0.140            | Beit 0.383            | Beit 0.683            | Beit 0.087            | Beit 0.478            | Beit 0.020             |
| 10 | data2vec 0.078        | data2vec 0.198        | efficientnet_b2 0.499 | data2vec 0.044        | data2vec 0.246        | data2vec 0.014         |
| 11 | efficientnet_b6 0.013 | efficientnet_b6 0.053 | CLIP 0.582            | efficientnet_b6 0.030 | efficientnet_b6 0.029 | efficientnet_b6 0.011  |
| 12 | efficientnet_b2 0.013 | efficientnet_b2 0.062 | efficientnet_b6 0.5   | efficientnet_b2 0.019 | efficientnet_b2 0.029 | efficientnet_b2 0.008  |



### knn_100
|    | Dataset_Flowers102    | UCMerced_Dataset      | ISIC                  | Dataset_DTD           | PatternNet_Dataset    | Dataset_FGVCAircraft   |
|---:|:----------------------|:----------------------|:----------------------|:----------------------|:----------------------|:-----------------------|
|  0 | CLIP 0.830            | DINO 0.904            | DINO 0.538            | DINO 0.661            | CLIP 0.942            | DINO 0.287             |
|  1 | DINO 0.761            | vit_l_32 0.831        | vit_b_32 0.591        | CLIP 0.633            | DINO 0.947            | vit_l_16 0.220         |
|  2 | vit_b_16 0.584        | vit_b_32 0.852        | ViTMAE 0.723          | convnext_base 0.520   | vit_l_16 0.866        | vit_b_16 0.207         |
|  3 | vit_l_16 0.562        | vit_l_16 0.791        | vit_l_16 0.590        | vit_b_32 0.550        | convnext_tiny 0.889   | convnext_tiny 0.148    |
|  4 | vit_l_32 0.513        | CLIP 0.896            | convnext_base 0.509   | vit_b_16 0.550        | vit_b_16 0.888        | vit_l_32 0.200         |
|  5 | convnext_base 0.408   | convnext_base 0.813   | vit_b_16 0.551        | convnext_tiny 0.529   | vit_l_32 0.881        | CLIP 0.312             |
|  6 | convnext_tiny 0.509   | convnext_tiny 0.769   | convnext_tiny 0.512   | vit_l_16 0.522        | vit_b_32 0.878        | vit_b_32 0.178         |
|  7 | vit_b_32 0.503        | vit_b_16 0.834        | vit_l_32 0.535        | vit_l_32 0.484        | convnext_base 0.854   | convnext_base 0.106    |
|  8 | ViTMAE 0.153          | ViTMAE 0.594          | data2vec 0.701        | ViTMAE 0.219          | ViTMAE 0.671          | ViTMAE 0.038           |
|  9 | Beit 0.137            | Beit 0.624            | Beit 0.623            | Beit 0.160            | Beit 0.649            | Beit 0.031             |
| 10 | data2vec 0.098        | data2vec 0.310        | efficientnet_b2 0.502 | data2vec 0.066        | data2vec 0.394        | data2vec 0.033         |
| 11 | efficientnet_b6 0.013 | efficientnet_b6 0.051 | CLIP 0.637            | efficientnet_b6 0.023 | efficientnet_b6 0.037 | efficientnet_b6 0.009  |
| 12 | efficientnet_b2 0.022 | efficientnet_b2 0.075 | efficientnet_b6 0.5   | efficientnet_b2 0.035 | efficientnet_b2 0.031 | efficientnet_b2 0.017  |



### lr_10
|    | Dataset_Flowers102    | UCMerced_Dataset      | ISIC                  | Dataset_DTD           | PatternNet_Dataset    | Dataset_FGVCAircraft   |
|---:|:----------------------|:----------------------|:----------------------|:----------------------|:----------------------|:-----------------------|
|  0 | CLIP 0.905            | DINO 0.885            | DINO 0.715            | DINO 0.602            | CLIP 0.944            | DINO 0.383             |
|  1 | DINO 0.937            | vit_l_32 0.837        | vit_b_32 0.623        | CLIP 0.600            | DINO 0.955            | vit_l_16 0.312         |
|  2 | vit_b_16 0.856        | vit_b_32 0.843        | ViTMAE 0.701          | convnext_base 0.573   | vit_l_16 0.923        | vit_b_16 0.314         |
|  3 | vit_l_16 0.869        | vit_l_16 0.851        | vit_l_16 0.632        | vit_b_32 0.555        | convnext_tiny 0.914   | convnext_tiny 0.280    |
|  4 | vit_l_32 0.810        | CLIP 0.878            | convnext_base 0.665   | vit_b_16 0.548        | vit_b_16 0.910        | vit_l_32 0.309         |
|  5 | convnext_base 0.830   | convnext_base 0.821   | vit_b_16 0.636        | convnext_tiny 0.586   | vit_l_32 0.901        | CLIP 0.345             |
|  6 | convnext_tiny 0.830   | convnext_tiny 0.818   | convnext_tiny 0.624   | vit_l_16 0.544        | vit_b_32 0.887        | vit_b_32 0.275         |
|  7 | vit_b_32 0.810        | vit_b_16 0.807        | vit_l_32 0.687        | vit_l_32 0.548        | convnext_base 0.897   | convnext_base 0.222    |
|  8 | ViTMAE 0.542          | ViTMAE 0.607          | data2vec 0.654        | ViTMAE 0.406          | ViTMAE 0.817          | ViTMAE 0.096           |
|  9 | Beit 0.320            | Beit 0.531            | Beit 0.735            | Beit 0.137            | Beit 0.683            | Beit 0.043             |
| 10 | data2vec 0.183        | data2vec 0.283        | efficientnet_b2 0.518 | data2vec 0.075        | data2vec 0.393        | data2vec 0.029         |
| 11 | efficientnet_b6 0.013 | efficientnet_b6 0.080 | CLIP 0.662            | efficientnet_b6 0.025 | efficientnet_b6 0.050 | efficientnet_b6 0.016  |
| 12 | efficientnet_b2 0.035 | efficientnet_b2 0.055 | efficientnet_b6 0.506 | efficientnet_b2 0.023 | efficientnet_b2 0.042 | efficientnet_b2 0.011  |



### lr_100
|    | Dataset_Flowers102    | UCMerced_Dataset      | ISIC                  | Dataset_DTD           | PatternNet_Dataset    | Dataset_FGVCAircraft   |
|---:|:----------------------|:----------------------|:----------------------|:----------------------|:----------------------|:-----------------------|
|  0 | CLIP 0.921            | DINO 0.970            | DINO 0.719            | DINO 0.742            | CLIP 0.979            | DINO 0.516             |
|  1 | DINO 0.921            | vit_l_32 0.948        | vit_b_32 0.676        | CLIP 0.737            | DINO 0.978            | vit_l_16 0.461         |
|  2 | vit_b_16 0.852        | vit_b_32 0.944        | ViTMAE 0.670          | convnext_base 0.718   | vit_l_16 0.972        | vit_b_16 0.447         |
|  3 | vit_l_16 0.849        | vit_l_16 0.942        | vit_l_16 0.657        | vit_b_32 0.690        | convnext_tiny 0.968   | convnext_tiny 0.437    |
|  4 | vit_l_32 0.839        | CLIP 0.934            | convnext_base 0.654   | vit_b_16 0.685        | vit_b_16 0.964        | vit_l_32 0.434         |
|  5 | convnext_base 0.839   | convnext_base 0.934   | vit_b_16 0.638        | convnext_tiny 0.683   | vit_l_32 0.964        | CLIP 0.420             |
|  6 | convnext_tiny 0.826   | convnext_tiny 0.932   | convnext_tiny 0.627   | vit_l_16 0.680        | vit_b_32 0.961        | vit_b_32 0.377         |
|  7 | vit_b_32 0.797        | vit_b_16 0.932        | vit_l_32 0.615        | vit_l_32 0.652        | convnext_base 0.961   | convnext_base 0.364    |
|  8 | ViTMAE 0.558          | ViTMAE 0.888          | data2vec 0.612        | ViTMAE 0.633          | ViTMAE 0.925          | ViTMAE 0.149           |
|  9 | Beit 0.307            | Beit 0.787            | Beit 0.575            | Beit 0.267            | Beit 0.829            | Beit 0.061             |
| 10 | data2vec 0.140        | data2vec 0.429        | efficientnet_b2 0.536 | data2vec 0.113        | data2vec 0.477        | data2vec 0.037         |
| 11 | efficientnet_b6 0.026 | efficientnet_b6 0.149 | CLIP 0.529            | efficientnet_b6 0.033 | efficientnet_b6 0.085 | efficientnet_b6 0.029  |
| 12 | efficientnet_b2 0.022 | efficientnet_b2 0.081 | efficientnet_b6 0.525 | efficientnet_b2 0.028 | efficientnet_b2 0.066 | efficientnet_b2 0.020  |





From the results above, in most cases in terms of ML algorithms performance, logistic regression with 100 samples > logistic regression with 10 samples > knn with 100 samples > knn with 10 samples, but if each samples in dataset is not too similar like DTD, UCMerced and PatternNet, sometimes logistic regression with 10 samples and knn with 100 samples can have same or better performance for best model. In industry, knn-like information retrieval is much cheaper and fast than logistic regression, so if our samples are not too similar and we have enough samples, KNN maybe a better option. 

As for model representation, DINO is superisingly the best model in most cases and then CLIP or VIT trained on imagenet. And in most cases, efficient net and mask based self supervised model perform poorly. Thus, from this point of view, DINO is more promising, because it use less data for pretraining but the result is still good.  


The result for fine tuning and comparation with simple ML + feature training is shown below table:


### Result for classification experiment on Data Drift dataset




### Result for clustering experiment 

The result of clustering experiment is shown below. For each column, each model is arranged in the order of model performance and the number in right of model name is adjusted rand score.

|    | Dataset_DTD         | Dataset_FGVCAircraft   | Dataset_Flowers102   | ISIC                | FMoWDataset ID      | FMoWDataset OOD     | IWildCamDataset ID   | IWildCamDataset OOD   | PatternNet_Dataset   | UCMerced_Dataset    |
|---:|:--------------------|:-----------------------|:---------------------|:--------------------|:--------------------|:--------------------|:---------------------|:----------------------|:---------------------|:--------------------|
|  0 | CLIP 0.364          | CLIP 0.147             | CLIP 0.654           | data2vec 0.177      | CLIP 0.086          | CLIP 0.102          | DINO 0.213           | vit_b_32 0.182        | vit_b_32 0.846       | CLIP 0.684          |
|  1 | DINO 0.343          | DINO 0.106             | DINO 0.623           | Beit 0.164          | DINO 0.084          | DINO 0.079          | vit_b_32 0.173       | DINO 0.165            | convnext_base 0.841  | vit_b_16 0.649      |
|  2 | convnext_tiny 0.303 | vit_b_16 0.072         | convnext_tiny 0.380  | ViTMAE 0.120        | vit_b_32 0.055      | convnext_tiny 0.057 | CLIP 0.166           | vit_b_16 0.148        | convnext_tiny 0.834  | DINO 0.645          |
|  3 | convnext_base 0.291 | vit_b_32 0.062         | vit_b_16 0.363       | CLIP 0.087          | vit_b_16 0.049      | vit_b_16 0.057      | vit_b_16 0.161       | convnext_tiny 0.127   | vit_b_16 0.824       | vit_b_32 0.616      |
|  4 | vit_b_16 0.284      | convnext_tiny 0.046    | vit_b_32 0.344       | vit_b_32 0.029      | convnext_tiny 0.047 | vit_b_32 0.052      | convnext_tiny 0.155  | CLIP 0.126            | DINO 0.796           | convnext_tiny 0.538 |
|  5 | vit_b_32 0.275      | convnext_base 0.031    | convnext_base 0.257  | vit_b_16 0.020      | convnext_base 0.038 | convnext_base 0.048 | convnext_base 0.135  | convnext_base 0.097   | CLIP 0.744           | convnext_base 0.508 |
|  6 | ViTMAE 0.030        | ViTMAE 0.006           | Beit 0.058           | DINO 0.020          | Beit 0.009          | Beit 0.009          | Beit 0.117           | Beit 0.089            | ViTMAE 0.394         | Beit 0.172          |
|  7 | Beit 0.022          | Beit 0.006             | ViTMAE 0.053         | convnext_base 0.002 | ViTMAE 0.004        | ViTMAE 0.005        | data2vec 0.072       | data2vec 0.058        | Beit 0.365           | ViTMAE 0.148        |
|  8 | data2vec 0.013      | data2vec 0.005         | data2vec 0.047       | convnext_tiny -0.00 | data2vec 0.004      | data2vec 0.004      | ViTMAE 0.058         | ViTMAE 0.040          | data2vec 0.085       | data2vec 0.064      |


From the table, CLIP model achieved the best performance and then DINO. Also, except ISIC dataset, those mask based self-supervised model (VITMAE, Beit, Data2vec) perform poorly in most cases. 

The model that trained in supervised way in imageNet (VIT, convnext) perform good in IWildCamDataset dataset. This might because the class labels and samples are too similar to imageNet. Also, the DINO model that self-supervised pretrained on imageNet perform even better than CILP. So one interesting thing would be what would happend if DINO is trained on CLIP's dataset, will DINO representation better than CILP in terms of clustering?

For ISIC dataset, mask based self-supervised model like Data2vec which perform not good in the other dataset perform very good this time. Because in ISIC dataset, the difference between different classes are very small compared to other datasets, it might be possible that mased self-supervised model tend to maintain those tiny information. But ISIC is the only that kind of dataset we tested, it might possible that this is just an coincidence.



<!-- IWildCamDatase -->


# Discussion and Conclusion

# Future work


<!-- ### Evaluation

Classification task:
 - Accuracy: Percentage of correction prediction

Clustering task: 
 - homogeneity score: each cluster contains only members of a single class.
 - completeness score: all members of a given class are assigned to the same cluster.
 - v measure score: (2 * homogeneity * completeness)/(homogeneity + completeness)
 - adjusted rand score: Rand index adjusted for chance. The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings. -->





<!-- 
## Project Timeline
- ~~Week 1-2: Define problems and writing proposal.~~
- ~~Week 3: Do presentation and test feasibility of different kind of models.~~
- Week 4-6: Write training pipeline and clustering pipeline for different models
  - In week 5, we should have a runable pipeline for training and testing
  - In week 6, we should have some result for classification and clustering tasks.
- Week 7-8: Wait for training and test result and focus on further testing on label noise or distribution shift.
- Week 9-10: Start writing report and prepare for presentation. -->


## Reference

[1] Radford, Alec, et al. "Learning Transferable Visual Models From Natural Language Supervision" *arXiv*. 2021.

[2] Alexey, Lucas, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" 

[3] Zhuang, Hanzi Mao1 et al. "A ConvNet for the 2020s"

[4] Mingxing, Quoc "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

[5] Hangbo, Li, Furu "BEiT: BERT Pre-Training of Image Transformers"

[6] Kaiming, Xinlei et al "Masked Autoencoders Are Scalable Vision Learners"

[7] Mathilde, Hugo et al "Emerging Properties in Self-Supervised Vision Transformers"

[8] Alexei, Wei-Ning et al  "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language"

[9] Adam J., Caleb et al "TorchGeo: deep learning with geospatial data" https://arxiv.org/abs/2111.08872

[10] SIIM-ISIC Melanoma Classification https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview/siim-ai-conference

[11] WILDS: A Benchmark of in-the-Wild Distribution Shifts. Pang Wei Koh*, Shiori Sagawa*, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. ICML 2021.


## Appendix
