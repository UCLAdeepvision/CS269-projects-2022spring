---
layout: post
comments: true
title: Zero-shot weakly Supervised Localization using CLIP
author: Jingdong Gao, Jiarui Wang
date: 2022-04-19
---


> Our project focuses on module weak supervision. And we are mainly interested in the area of zero-shot weakly supervised localization. We aimed at using CLIP as backbone because of its outstanding performance on zero shot classification tasks. We would like to extend other methods such as class activation/attention map, LCTR transformer and TS transformer to CLIP so that we can achieve localization prediction on novel datasets.
 
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
We aimed at using CLIP as backbone because of its outstanding performance on zero shot classification tasks. We would like to extend other methods such as class activation/attention map, LCTR transformer and TS transformer to CLIP so that we can achieve localization prediction on novel datasets. This page will introduce this project from the aspect of background introduction and method analysis. 

## Background 

Weakly supervised localization is important as it can effectively reduce the cost of annotation for bounding boxes. Zero shot learning is attractive because it can not only solve problems targeting at pre-determined object category, but also non-predetermined category. 

### What is weakly supervised localization 

Object localization is a well known research problem in computer vision. Within the area, weakly supervised localization had attracted a huge amount of efforts. It only needs image level annotation compared with supervised localization as shown in Figure. 

![Weakly supervised localization]({{ '/assets/images/team03/1.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 1. Weakly supeprvised localization only needs image label. There is no need to provide localization ground truth.*


### What is zero shot
The idea of Zero shot was first rasied in paper Dataless classification ([Chang, et al. 2008](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)). Zero shot learning means the model learns from pre-determined class categories, but at the same time, it has the ability to predict and test non-defined classes. Our idea is driven by CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)). The pre-trained neural network can be applied to other dataset which including non-trained categories. As shown in Figure for example, to classify animals, network is trained to recognized some classes including horse, but has never been given a zebra during training. In zero shot localization, ideally it should be able to recognize a zebra.

![Zero Shot]({{ '/assets/images/team03/2.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 2. Zero shot model can be applied to any other dataset during test.*


## Method Overview
We plan to use CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)) as our main backbone. To tackle the weakly supervised localization problem, we want to combine the zero-shot generalizability enabled by CLIP and saliency map approaches that handle the localization task. Addtionally, we may focus on saliency approaches originate from two directions: CNN based image encoders and vision transformer based encoders, since CLIP pretrained both. For CNN based encoders, we may will adapt algorithms such as ScoreCAM and GradCAM. Methods such as LCTR and TS CAM may be explored for ViT based encoders if we have sufficient time and computation resources, since these methods require additional training of extra layers.

### CLIP

CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)) combines zero shot transfer and natural language supervision. After pretraining, CLIP is supposed to have competitive performance on other non-predetermined image categories. The overall workflow is as following: CLIP learns an enumorous amount of images and their corresponding labels. And then, different classification tasks will be applied to CLIP models. For each test image, text description "a photo of category" will be provided in order to find the nearest answer. 

The detailed structure is shown in Fig.3. Model consists of an image encoder and and text encoders. Inputs are image, text pairs where texts are image captions. The model is trained with contrastive loss where matching pairs are positive examples and unmatched pairs are negative examples. And then, inner product of text embeddings and image embeddings will be calculated for training purpose.

During a new classification test, given a set of labels and input image, CLIP uses the pretrained text encoder to embed the labels. Then compute the inner product between the text embeddings and the image embedding from the image encoder. Finally, the class that generates the highest similarity score will be the prediction answer. 



![CLIP]({{ '/assets/images/team03/3.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 3. CLIP's structure. From CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf))*



CLIP pretrained image classification model with language supervision. Then by combining CLIP with class activation/attention map methods, we assume the activated area can be regarded as localization on novel datasets. Similarly, if we apply LCTR transformer or TS-CAM transformer into CLIP, it may still have the ability of weakly supervised localization.


### Score CAM

Score-CAM ([Wang, et al. 2020](https://arxiv.org/pdf/1910.01279.pdf)) is a popular method to generate saliency maps for CNN based image classifiers. It often produces more accurate and stable results than gradient based methods. According to Fig. 4, given an input image, score cam takes the activation maps from the last convolution layer, and uses these maps as masks over the the input image. The masked input images are processed by the CNN again to generate a weight for each activation map. The weights are normalized the weighted sum of the maps is used as the final output. We can naively adapt the method to CLIP by changing the definition of the score from the logit of the target class to the inner product between the masked input image embedding resulted from the CLIP image encoder and the text embedding generated by the text encoder. 


![Score-CAM]({{ '/assets/images/team03/4.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 4. Score CAM's structure. From Score-CAM ([Wang, et al. 2020](https://arxiv.org/pdf/1910.01279.pdf))*
### LCTR transformer

### TS-CAM transformer
Vision transformer has become an important architecture in the field of computer vision. However, producing saliency maps from ViT architectures is not as straightforward as with CNN architectures, since the attention map among difference patches does not directly encode semantic information, which resides in the class token. TS-CAM is one of the first methods that addresses this problem and shows SOTA performance on weakly supervised object localization tasks. It consists of two modules: a semantic reallocation module that seeks to transfer class semantics from the class token to the patch tokens, which requires training of additional CNN layers, and an attention module that extracts global relationship between image patches from attention maps in each transformer block. We may apply this approach to accomodate ViT image encoders pretrained in CLIP. 



## Data Plan
To compare with previous results from  works in the area of weakly supervised object localization, we plan to test our method on the ILSVRC2012 dataset that was commonly used in the field. The dataset is accessible for public download. However, due to the size of dataset and our limited compututaion resources, we are going to manually select a subset of ten classes from the entire dataset, and go into detailed analysis and optimization of the proposed on the selected data. We aim to select a classes that represent distinct concepts(e.g. animals and tools) and have difference visual features(e.g. flowers and boat), so that our analysis will generalize to the larger set of classes.

To ensure consistency, we use the same data preprocessing scheme from the original CLIP model. 

## Reference
[1] Gao, Wan, et al. "TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization." *International Conference on Computer Vision*. 2021.

[2] Radford, Kimm et al. "Learning Transferable Visual Models From Natural Language Supervision." *Arxiv*. 2021.

[3] Wang et al. "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks." *Conference on Computer Vision and Pattern Recognition*. 2019.

[4] Zhou et al. "Learning Deep Features for Discriminative Localization." *Conference on Computer Vision and Pattern Recognition*. 2016.

[5] Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *International Conference on Computer Vision*. 2017.

[6] Chen et al. "LCTR: On Awakening the Local Continuity of Transformer for Weakly Supervised Object Localization." *the Association for the Advancement of Artificial Intelligence*. 2017.

---
