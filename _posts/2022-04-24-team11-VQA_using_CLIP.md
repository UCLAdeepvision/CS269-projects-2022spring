---
layout: post
comments: true
title: Visual Question Answering using CLIP
author: Rakesh Dal and Sri Keerthi Bolli
date: 2022-04-24
---

> We propose to leverage CLIP for VQA (visual question answering) and further enhance the performance using language level semantic segmentation, prompt engineering. We would also like to experiment with multilingual CLIP on multilingual VQA. Additionally, we would also like to study the performance of the models on different types of questions (yes/no questions, count based questions, etc).


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Background
CLIP (Contrastive Language Image Pre-Training) is a neural network architecture that efficiently learns visual concepts from natural language supervision. While standard image models jointly train an image feature extractor and a linear classifier to predict some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes. The Visual Question Answering (VQA) task combines challenges for processing data with both Visual and Linguistic processing, to answer basic ‘common sense’ questions about given images. Given an image and a question in natural language, the VQA system tries to find the correct answer using visual elements of the image and inference gathered from textual questions.


## Motivation
CLIP has been trained on a wide variety of images with a wide variety of natural language supervision that’s abundantly available on the internet. Hence, it is much more representative and has shown good zero-shot capabilities on various vision-language tasks. Although the transformer-based language models are highly expressive, they are relatively weak at zero-shot ImageNet classification. The model has been non-trivially transferred to most tasks and is often competitive without the need for any dataset specific training. The encoders used in VQA models are trained on manually annotated data which is very costly to collect. Hence, using the pre-trained CLIP encoders could be used in place of them. Semantic segmentation has shown to have improved the performance of VQA by adding additional features into the VQA encoders. Formulating different kinds of prompts by combining the questions and answers in VQA and feeding that to CLIP has also shown to have a significant effect on the performance of VQA.


## Datasets
We plan to use the VQA 2.0 dataset containing real and abstract images. It has different types of questions like yes/no questions, count based questions, etc. The dataset can be found [here](https://visualqa.org/). Few examples from VQA 2.0 dataset are shown in Figure [2].

We also intend to test the performance of CLIP based VQA Models on Multilingual VQA using multilingual variant of the Conceptual-12M dataset and multilingual variants of the VQAv2 train and validation sets available [here](https://github.com/gchhablani/multilingual-vqa).

![]({{ '/assets/images/team11/dataset.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Examples of VQA 2.0 dataset*.


## CLIP Architecture
The architecture diagram of CLIP is shown in Figure [2]. It has been trained on 400M image-text pairs crawled from internet.

![]({{ '/assets/images/team11/CLIP.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. CLIP's image, text encoders and it's application for zero shot prediction*.


## Approach
Pre-trained CLIP models i.e. the visual and text encoders of CLIP are available. We want to replace both the visual and text encoders of a few state-of-the-art VQA models like Pythia, MCAN, LXMERT with those of pre-trained CLIP models built on various baseline models and study the effectiveness of CLIP in this regard. We would further obtain the segmentation maps of the images and use them on the VQA models and study their effectiveness. Also, we plan to create different types of prompts by combining the question and answers from the dataset samples and feed those prompts to the text encoders and study their performance. We also want to split the dataset based on the type of questions and perform the above mentioned experiments on each of the categories. At the end, we would also want to extend these experiment to multilingual VQA.


## Preliminary Results
We tried to reproduce the results of the paper, “How Much Can CLIP Benefit Vision-and-Language Tasks?” where the authors use clip during inference on existing VQA models like Pythia. We used the Visual Genome and the VQA 2.0 datasets for the VQA Task. We were able to successfully extract the features using CLIP which have to be fed by the models like Pythia to obtain results on the VQA Task. We expect to obtain the results very soon.


## Expected Results
Initially, we plan to run the state-of-the-art VQA models and get the baseline results. We would then replace the encoders of those VQA models with pre-trained encoders of CLIP. And the next step would be use semantic maps generated by semantic segmentation of the visual images of the datasets on VQA + CLIP models. We also plan to experiment the above mentioned VQA models with various types of prompts. And finally, we will present a detailed study of the effectiveness of all the performed experiments on different types of questions present in the dataset. 


## References

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,. "Learning transferable visual models
from natural language supervision." *arXiv preprint
arXiv:2103.00020.*, et al. 2021. \
[2] Yash Goyal, Tejas Khot, Douglas Summers-Stay,
Dhruv Batra, and Devi Parikh. 2017. "Making the
v in vqa matter: Elevating the role of image understanding in visual question answering." *In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition*, pages 6904–6913. \
[3] Hao Tan and Mohit Bansal. 2019. "Lxmert: Learning
cross-modality encoder representations from transformers." *arXiv preprint arXiv:1908.07490.*, [cs.CL] 3 Dec 2019 \
[4] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and
Qi Tian. 2019. "Deep modular co-attention networks
for visual question answering." *In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6281–6290. \
[5] Yu Jiang, Vivek Natarajan, Xinlei Chen, Marcus
Rohrbach, Dhruv Batra, and Devi Parikh. 2018.
"Pythia v0. 1: the winning entry to the vqa challenge
2018." *arXiv preprint arXiv:1807.09956.*, cs.CV] 27 Jul 2018 \
[6] Pham, VQ., Mishima, N., Nakasu, T. "Improving Visual Question Answering by Semantic Segmentation." *In: Farkaš, I., Masulli, P., Otte, S., Wermter, S. (eds) Artificial Neural Networks and Machine Learning – ICANN 2021. ICANN 2021. Lecture Notes in Computer Science(), vol 12893. Springer, Cham. https://doi.org/10.1007/978-3-030-86365-4_37*, 2021.

