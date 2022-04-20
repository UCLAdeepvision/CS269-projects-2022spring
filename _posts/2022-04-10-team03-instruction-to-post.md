---
layout: post
comments: true
title: Zero-shot weakly Supervised Localization using CLIP
author: Jingdong Gao, Jiarui Wang
date: 2022-04-19
---


> Our project focuses on module weak supervision. And we are mainly interested in the area of zero-shot weakly supervised localization.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Weakly supervised localization is important as it can effectively reduce the cost of annotation for bounding boxes. Zero shot learning is attractive because it can not only solve problems targeting at pre-determined object category, but also non-predetermined category. 

## Background 


### What is weakly supervised localization 

Object localization is a well known research problem in computer vision. Within the area, weakly supervised localization had attracted a huge amount of efforts. It only needs image level annotation compared with supervised localization as shown in Figure. 

![Weakly supervised localization]({{ '/assets/images/team03/1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Weakly supeprvised localization only needs image label. There is no need to provide localization ground truth.*




### What is zero shot

The idea of Zero shot was first rasied in paper Dataless classification ([Chang, et al. 2008](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)). Zero shot learning means the model learns from pre-determined class categories, but at the same time, it has the ability to predict and test non-defined classes. Our idea is driven by CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)). The pre-trained neural network can be applied to other dataset which including non-trained categories. As shown in Figure for example, to classify animals, network is trained to recognized some classes including horse, but has never been given a zebra during training. In zero shot localization, ideally it should be able to recognize a zebra.

![Zero Shot]({{ '/assets/images/team03/2.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Zero shot can be applied to any other dataset.*


## Method Overview
We plan to use CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)) as our main backbone. For the weakly supervised localization problem, we will focus on two direction: image based encoder and transformer based encoder.

### CLIP

CLIP pretrained image classification model with language supervision. Model consists of an image encoder and and text encoders. Inputs are image, text pairs where texts are image captions. The model is trained with contrastive loss where matching pairs are positive examples and unmatched pairs are negative examples. Inner product of text embeddings and image embeddings. 
During inference time, given a new classification task with a set of labels and input image, we can use the pretrained text encoder to embed the labels and compute the inner product between the text embeddings and the image embedding from the image encoder. Then the class that generate the highest similarity score will be predicted. 

Then by combining CLIP with class activation/attention map methods, we attempt zero-shot object localization on novel datasets.

CLIP has pretrained two types of image encoders, CNN based and transformer based.
https://arxiv.org/pdf/1910.01279.pdf

https://arxiv.org/pdf/2103.00020.pdf

Image encoder可以用resnet， 当成一个cnn，就可以把cam加进去。把text encoder I*T的结果当成是weight, 然后apply back to activation map


### Score CAM Image Encoder

### LCTR transformer

### TS CAM transformer
### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).






## Reference
[1] Gao, Wan, et al. "TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization." *International Conference on Computer Vision*. 2021.

[2] Radford, Kimm et al. "Learning Transferable Visual Models From Natural Language Supervision." *Arxiv*. 2021.

[3] Wang et al. "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks." *Conference on Computer Vision and Pattern Recognition*. 2019.

[4] Zhou et al. "Learning Deep Features for Discriminative Localization." *Conference on Computer Vision and Pattern Recognition*. 2016.

[5] Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *International Conference on Computer Vision*. 2017.

[6] Chen et al. "LCTR: On Awakening the Local Continuity of Transformer for Weakly Supervised Object Localization." *the Association for the Advancement of Artificial Intelligence*. 2017.

---
