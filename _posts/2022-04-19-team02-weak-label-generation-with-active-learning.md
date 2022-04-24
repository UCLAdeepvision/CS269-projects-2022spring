---
layout: post
comments: true
title: Weak Supervision through GANs with Active Learning
author: Andrew Choi and Arvind Vepa
date: 2022-04-19
---

> With recent advances of deep learning, unprecedented performance has been achieved for the task of image segmentation. Despite this, image segmentation is often hampered by being quite data hungry. This problem is exacerbated by the fact that manually generating accurate ground truth masks is extremely tedious and time-consuming. Due to this, the development of efficient and effective weak supervision methods has become crucial. 
> Generative Adversarial Networks (GANs) have shown surprisingly good results in Image-to-Image Translation. Given this, we would like to explore whether GANs can be used in a weak supervision pipeline to generate realistic weak labels such as scribbles, bounding boxes, and points. We hypothesize that a GAN capable of generating convincing weak labels can in practice produce several weak label samples, resulting in an increasingly dense weak label. This in turn should lead to improved performance when training image segmentation models.
> In addition to this, we also propose an active learning pipeline to further improve performance which takes into consideration areas of uncertainty in the segmented image. Additionally, with active learning, we consider heterogeneous weak annotations, including bounding boxes, points, image tags, and scribbles.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Motivation
Although weak supervision has been a popular research avenue in recent years, most methods are specifically catered towards a specific type of weak label. For ease of use and generalizability, universal weak supervision methods have become increasingly attractive. We hypothesize that GANs can act as a universal weak supervision method. By training on datasets with prelabeled weak annotations, we propose to produce a GAN capable of generating additional weak annotations. Given this, we would like to use weak annotations produced by a GAN to improve performance for an image segmentation model. 

After this, we would like to incorporate active learning to further improve results. From a trained segmentation model, we can generate an uncertainty map of the outputted segmentation mask. The uncertainty of a pixel can be decided by how close the prediction is to a uniform distribution. Given the regions of uncertainty, we can then generate further annotation candidates and then rank these candidates based on their resultant change in prediction uncertainty. A tentative metric for this can be \
$$
\Delta u = \sum^{N-1}_{n=0} \sum^{h-1}_{j=0} \sum^{w-1}_{i=0} p^n_{i,j} log(p^n_{i,j}) - \sum^{N-1}_{n=0} \sum^{h-1}_{j=0} \sum^{w-1}_{i=0} \bar{p}^n_{i,j} log(\bar{p}^n_{i,j}) 
$$ 

where $$N$$ is the number of classes; $$h$$ is the height of the image; $$w$$ is the width of the image; $$p$$ is the new prediction, and $$\bar{p}$$ is the old prediction. This is equivalent to the change in entropy.
A human annotator can then choose the best weak annotation candidate. This active learning procedure can be repeated until the desired performance is achieved.

## Contributions
### Contributions concerning GANs
To the best of our knowledge, no previous work has used GANs to generate additional weak supervision annotations before. GANs have primarily been used to generate entire segmentations. Weak label generation should in theory be simpler and thus, more feasible. We hope to show that this novel idea can significantly benefit model training for image segmentation.
### Contributions concerning Active Learning
To the best of our knowledge, there has been a lack of focus on suggesting weak annotations through active learning as most active learning focuses on suggesting pixel-wise annotations or regions. In addition to this, most active learning models evaluate annotation candidates based on uncertainty. We propose to instead use the change in uncertainty $$\Delta u$$ (entropy). We hope to show that this metric is both novel and practical towards evaluating annotation candidates.

### Contributions towards Weak Supervision
To the best of our knowledge, no previous work has considered training models on different types of weak annotations, especially in an active learning setting.


## General Project Methodology
1. Given some weak annotations, use a GAN to generate additional weak annotation candidates that will be used by a model to improve weak supervision segmentation performance.
2. Additionally, we use an active learning model to generate additional weak annotation candidates for human annotators to consider. 
3. Steps 1-2 can be repeated until desired performance is achieved.

## Project Timeline
- ~~Week 1-2: Formulate a relevant and interesting research problem.~~
- ~~Week 3: Research datasets as well as annotation materials needed.~~
- Week 4-6: Re-implement SOTA methods, train baseline models, and generate results. Furthermore, start training GAN network for weak label generation.
  - Ideally, we discover whether GANs are capable of producing plausible weak labels by the end of week 5.
  - Ideally, we re-implement all methods by the end of week 5.
  - All models should be trained and have generated results by end of week 6.
- Week 7-8: Focus on active learning component of the project.
- Week 9-10: Gather all results and start writing research report / presentation.

## Related Papers
We can compare out method to other universal supervision methods as well as baseline methods. In particular, we'd like to compare our work to [1], which showed impressive results against SOTA methods catered towards specific weak labels as shown below.

![]({{ '/assets/images/team02/example.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Results for Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning.* [1].

The code for this project can be found [here](https://github.com/twke18/SPML). We will also compare our approach to weak supervision using image tags [2], bounding boxes [3], labeled points and scribbles [4].

There is also related work in active learning. For example, there is prior work that finds both samples and regions of the image based on uncertainty to annotate for biomedical image segmentation [5]. Additionally, in later work,  active learning is employed in per-pixel annotation for segmentation [6]. However, both works do not consider active learning in terms of suggesting weak supervision and, especially, different forms of weak supervision, which is much more cost-effective.

## Datasets
Below is a list of the datasets we plan to use as well as information concerning them.
- [Scribble Sup Dataset](https://jifengdai.org/downloads/scribble_sup/) consists of 
  - PASCAL VOC 2012 set that involves 20 object categories (aeroplane, bicycle, ...) and one background category. There are 12,031 images annotated, including 10,582 images in the training set and 1,449 images in the validation set.
  - PASCAL-CONTEXT set that involves 59 object/stuff categories and one background category. Besides the 20 object categories in the first dataset, there are 39 extra categories (snow, tree, ...) included. We follow this protocol to annotate the PASCAL-CONTEXT dataset. We have 4,998 images in the training set annotated.
  - Image tags, bounding boxes, points, scribbles, and groundtruth annotations available
- DensePose
  - Dense correspondences from 2D images to surface-based representations of the human body.
  - Annotations involve points inferred from keypoints using a Gaussian model.
  - Image tags, bounding boxes, points, and groundtruth annotations available
- Automatic Cardiac Diagnosis Challenge (ACDC) Dataset
  - Contains cine-MR images obtained by 100 patients with different MR scanners and acquisition protocols. Manual segmentations are provided along with the images, containing pixel-wise annotations for the end-diastolic (ED) and end-systolic (ES) cardiac phases. The annotated structures are left ventricle (LV), right ventricle (RV) and myocardium (MYO).
  - Scribbles can be found [here](https://vios-s.github.io/multiscale-adversarial-attention-gates/data).
  - Image tags, bounding boxes, points, scribbles, and groundtruth annotations available
- DRIVE, STARE, and DSA (for vessel segmentation)
  - Image tags, bounding boxes, points, and groundtruth annotations available
  - Contains no scribbles but as the datasets are rather small, manually annotation may be feasible.

## What we hope to show
By the end of this project, we hope to show that GANs can be trained to generate effective weak labels for image segmentation. These weak labels can be sampled repeatedly in order to produce a more dense weak label, which should then result in improved performance when training image segmentation models. Along with this, we introduced an active learning model that takes into consideration areas of uncertainty in the segmented image. 

As GANs are quite difficult to train, we have prepared for the outcome where the GANs do not produce useful labels. In this scenario, we will focus on the active learning component and the benefits this can produce.

## Reference

[1] Ke, Tsung-Wei and Hwang, Jyh-Jing and Yu, Stella X. "Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning." *International Conference on Learning Representations*, 2021. \
[2] Yu-Ting Chang, Qiaosong Wang, Wei-Chih Hung, Robinson Piramuthu, Yi-Hsuan Tsai, and Ming- Hsuan Yang. "Weakly-supervised semantic segmentation via sub-category exploration." *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2020. \
[3] Chunfeng Song, Yan Huang, Wanli Ouyang, and Liang Wang. "Box-driven class-wise region masking and filling rate guided loss for weakly supervised semantic segmentation." *In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 3136–3145, 2019. \
[4] Meng Tang, Federico Perazzi, Abdelaziz Djelouah, Ismail Ben Ayed, Christopher Schroers, and Yuri Boykov. "On regularized losses for weakly-supervised cnn segmentation." *In Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 507–522, 2018b. \
[5] Lin Yang, Yizhe Zhang, Jianxu Chen, Siyuan Zhang, Danny Z. Chen. "Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation." *International Conference on Medical Image Computing and Computer Assisted Intervention*, 2017. \
[6] Soufiane Belharbi, Ismail Ben Ayed, Luke McCaffrey, Eric Granger. "Deep Active Learning for Joint Classification & Segmentation with Weak Annotator." *Winter Conference on Applications of Computer Vision*, 2021.
---
