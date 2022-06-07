---
layout: post
comments: true
title: Object Removal using combination of segmentation and image inpainting
author: Sicheng Jiang, Andong Hua
date: 2022-04-24
---


> The project mainly focuses on the problem of object removal using a combination of segmentation and image inpainting techniques. We plan to develop a system that automatically detects target objects to remove using weakly supervised language-driven semantic or instance segmentation models and remove the object to recover the background image with image inpainting techniques.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}
## Key words
Object removal, language-driven semantic segmentation/instance segmentation, image inpainting 

## Motivation
Object removal techniques are widely used to remove unwanted objects. Nowadays, lots of Apps and picture tools can be found online to remove objects. Howevers, most of them are designed to let users select image regions to remove and replace them with corresponding backgrounds. For example, users need to manually draw a boundary or  to select target regions, which is time-consuming. Our project aims at developing an algorithm to automatically detect the regions to remove using segmentation technique. We would explore image inpainting since it plays an important role in applications such as photo restoration, image editing, especially object removal. 

Instead of using normal segmentation methods, we would explore language-driven segmentation methods ([1], [2]) to detect the target objects. Existing semantic segmentation methods require human annotators to label each pixel to create training dataset which is labor intensive and costly, while the language-driven models label the image with only words or phrases describing specific objects and relationships between them to reduce the cost of annotation.


## Related work
1. LANGUAGE-DRIVEN SEMANTIC SEGMENTATION
2. PhraseCut- Language-based Image Segmentation in the Wild
3. Image Inpainting Guided by Coherence Priors of Semantics and Textures
4. Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional 5. Encoding
5. Text-Guided Neural Image Inpainting(https://arxiv.org/pdf/2004.03212.pdf)
6. Resolution-robust Large Mask Inpainting with Fourier Convolutions 


## Methodology
![pipeline]({{ '/assets/images/team18/methodology.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Main process for the object removal task
The proposed method can be formulated as follows: given an image Ix and a sentence describing an object s, our model outputs a reconstructed image without the object  Iy. There are two stages of our model: segmentation part and inpainting part. We plan to work on two stages separately and then combine them into one model.
### Segmentation
In this part, the original image Ix and the sentence s is fed into the text-based segmentation model. The output is a mask m, which is a binary image. We plan to use the model LSeg[1] to do semantic level segmentation first and then improve the performance by segmenting objects into instance level based on HULANet[2]. 
### Inpainting
After obtaining the mask of the object in sentence s, the inpainting model can output Iy from input image Ix and mask m. We plan to use the model LaMa[6] to do image inpainting. To improve the performance, we will try to give the more information by inputting the sentence s. 


## Expected output
We expect our model can be end-to-end and can produce both structural and textural images. We hope to achieve SOTA performance in both segmentation and inpainting tasks. Additionally, we want to figure out image inpainting performance with different mask size as text-aided inpainting is good at large masks while traditional inpainting model performs better at small masks. We also want to find the influence on inpainting model performance with soft mask and hard mask.


## Reference
[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.