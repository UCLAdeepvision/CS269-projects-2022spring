---
layout: post
comments: true
title: Independent Causal Mechanism for Robust Deep Neural Networks
author: Vishnu Devarakonda, Ting-Po Huang
date: 2022-04-23
---

> In order to generalize the machine learning models and solve the "Distribution Shift" problem, we want to propose a different solution with independent mechanisms.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


## Motivation

Almost all machine Learning models have issues with generalization that stems from a phenomenon called distribution shift. This problem is due to the difference between the testing and training distributions which could both be subsequently different from the final real world data distribution. There are several well known approaches to improving model generaliztion including but not limted to, adding more data, balancing catgeories in the datasets, and data augmentation. Despite the improvements made by these approaches, one can easy find examples of real world data or through techniques like advesarial learning that can still fool the model. The issue could be due to not considering the causal effects on data generation[1].

In this paper, the authors propose an insight that could potentially lead to broad improvements in machine learning. It requries a shift in perspective and thinking about the causal effects that produce the distribution itself. In causal modeling, "the structure of each distribution is induced by physical mechanisms that give rise to dependences between observables." These mechanisms or experts have several unique properties of interest. First, they are independent and do not inform one another. Second, they are autonomous generative models effecting the distribution itself which grants them the ability transfer between problems. The goal of this project is to try and identify the mechansims that effect the distribution of human faces.

The distribution of human faces contains a variety of unqiue features that can be driven by genetics (hair, eye color, shape, etc), by culture (Turban, Hijab, etc), personality (beards, glasses, masks, etc), etc. These features can be thought of as "augmentations" that are driven by causal mechansims which effect the distribution of inherent human features(features all humans share). Finding these causal mechanisms could lead to greater ability to generalize across human faces.

## Project timeline

- Week 3 : Search for the human face datasets
- Week 4-5: Re-implement the digits detection work (with MNIST dataset) on paper.
- Week 6-9: propose and experiment with different architectures and datasets to reach our face detection task.
  - try the dataset in the below section in the current architecture.
  - propose our architecture to find the best independent mechanisms
- Week 10: wrap up and prepare for the presentation and compose the report.

## General Project Methodology

1. We will reproduce the digits detection work (with MNIST dataset) on paper.
2. Our goal is to identify the causal mechanisms that impact the distribution of human face. The origin paper uses data augmentation to sample noise input data. We want to use physical characteristics (hair, color, eyes) as our noise input data to help us train these mechanisms.
3. We will try the same architecture in paper with a different dataset. And see how it works. Keep finding datasets and try different architecture until our mechanisms reach the goal.

## Related Papers

[1] Goyal A, Lamb A, Hoffmann J, et al. "Recurrent independent mechanisms" arXiv preprint arXiv:1909.10893, 2019.

## Datasets

Below is the dataset, we would work on for training our mechanisms.

- Face pictures with featuress driven by genetics and culture
  - [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
    - high-quality image dataset of human faces, originally created as a benchmark for generative adversarial networks (GAN),
    - The dataset consists of 70,000 high-quality PNG images at 1024×1024 resolution and contains considerable variation in terms of age, ethnicity and image background. It also has good coverage of accessories such as eyeglasses, sunglasses, hats, etc.
  - [Tufts Face Database](https://www.kaggle.com/datasets/kpvisionlab/tufts-face-database)
    - the most comprehensive, large-scale (over 10,000 images, 74 females + 38 males, from more than 15 countries with an age range between 4 to 70 years old) face dataset that contains 7 image modalities: visible, near-infrared, thermal, computerized sketch, LYTRO, recorded video, and 3D images.
  - [UTKFace](https://susanqq.github.io/UTKFace/)
    - The dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity.
  - [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    - CelebA is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.
    - The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including 10,177 number of identities and 202,599 number of face images.
- Faces picture with real augumentation
  - [Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
    - This dataset contains expert-generated high-quality photoshopped face images.The images are composite of different faces, separated by eyes, nose, mouth, or whole face.
  - [Face Mask](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
    - This dataset contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.The classes are With mask, Without mask, and Mask worn incorrectly.
  - [Labelled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
    - LFW is a database of face photographs designed for studying the problem of unconstrained face recognition. This database was created and maintained by researchers at the University of Massachusetts, Amherst.
    - 13,233 images of 5,749 people were detected and centered by the Viola Jones face detector and collected from the web. 1,680 of the people pictured have two or more distinct photos in the dataset
  - [Yale Face Database](http://vision.ucsd.edu/content/yale-face-database)
    - The Yale Face Database contains 165 grayscale images in GIF format of 15 individuals. There are 11 images per subject, one per different facial expression or configuration: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.

## What we hope to show

We want to find effective independent mechanisms or experts to help us generalize across human faces. With the casual mechanisms we implemented, we can use them to detect human faces more robustly.
Also, we will test whether we can use these mechanisms in a different domain.

## Reference

[1] Giambattista Parascandolo, Niki Kilbertus, Mateo Rojas-Carulla, Bernhard Schölkopf. "Learning Independent Causal Mechanisms" Proceedings of the 35th International Conference on Machine Learning, PMLR. 2018.
