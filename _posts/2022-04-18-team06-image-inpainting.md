---
layout: post
comments: true
title: Image Inpainting
author: Weichong Ling, Yanxun Li
date: 2022-04-18
---


> Image inpainting is to fill in missing parts of images precisely based on the surrounding area using deep learning. Our goal is to implement a GAN-based model that takes an image as input and changes objects in the image selected by the user while keeping the realisticness.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Image inpainting is a popular topic of image generation in recent years. The goal of image inpainting is to fill in missing parts of images precisely based on the surrounding area using deep learning. Currently, image inpainting models using Generative Adversarial Network (GAN) can achieve such realistic results that humans cannot visually detect the reconstructed area. We intend to dive into image inpainting and explore how to utilize it to complete other image generation tasks. Our high-level idea is to implement a GAN-based model that takes an image as input and changes objects in the image selected by the user while keeping the realisticness. One possible direction is to implement a model that takes two images where the first image has a region masked and the second image has the target object that the user wants to put on the masked area of the first image. The user can roughly sketch the target object on the masked region, based on which the model can generate the target object in the first image with realisticness. We may change our direction once we find other more interesting and feasible directions.

## Related works
Recently, GAN-based structure [1] and deep learning are increasingly integrated into image inpainting techniques.  Iizuka et al [2] designed the first state-of-art algorithm by introducing global and local adversarial loss to the structure. [2] also proposed the dilated convolution to enlarge the receptive field in order to capture global information. Yu et al [3] further improved this structure with a contextual attention layer. Their method consists of a two-step structure that first generates a coarse inpainting. In the next step, a trainable contextual attention layer can use the features of known patches as convolutional filters to refine the generated patches. A year later, Yu et al [4] proposed gated convolution to make convolution weights for masked areas a learnable parameter. With gated convolution, [4] is capable of free-from mask, user guided generation, and photorealistic inpainting. 

We observe that all the works before [4] do not support user guided generation. However, user participation is important to generate inpainting that meets users' expectation. EdgeConnect [5] proposed a different structure to generate high quality image inpainting with free-form mask and user participation. It used canny edge detection to preprocess the input image, generate edges in the missing area, and use another GAN to synthesize the final output. Users can guide inpainting by directly sketching to edit the edge map. Our work is based on EdgeConnect [5] and the Contextual Attention [3]. The main difference is that our work supports masked object replacement, while the aforementioned two works only generate inpainting with respect to one input image. We believe our work can provide users with more freedom to edit and create.

## Approach
![EC-Structure]({{ '/assets/images/06/CA-structure.png' | relative_url }})
{: style="width:70%; margin-left:15%;"}
<center><i>Fig 1. Contextual Attention demonstration</i></center>  
Our model takes two images as input. The first image has a region masked, the second image has the target object that the user wants to put on the masked area. We then generate edge maps for both images by Canny edge detection. In the next step, users can sketch the target object on the masked region. Here we utilize the idea of Contextual Attention [3] to use user sketch as convolutional filters to borrow edges from the second edge map. Then we pass this unfinished edge map to EdgeConnect for further processing.

![EC-Structure]({{ '/assets/images/06/EC-structure.png' | relative_url }})
{: style="width:150%; margin-left:-25%;"}
<center><i>Fig 2. EdgeConnect model structure</i></center>  

We choose EdgeConnect [5] as the backbone of our project. The first advantage of EdgeConnect is that G2 [Fig.1] takes a complete edge map as input and output as the final inpainted image. This means we only need to deal with the object replacing on the edge map level. The rest of the work could be leveraged to the original code. Another advantage is that it’s easy to incorporate user sketches into the edge map. 

Though our idea heavily relies on EdgeConnect, we still plan to try some new techniques on its structure. Notice that EdgeConnect only uses dilated convolution. [4] has shown gated convolution can generate inpainting with better quality. Thus, if time permits, we also plan to test gated convolution in our project.


## Expectation

In the next six weeks, we hope we can at least build our method and generate lo-fi output. For example, if we want to replace the bowl on the image with a cup, we hope our generated image can at least show a cup and have global consistency in the original image. Local consistency such as clean edges and no artifacts is significant for a good image inpainting model. However, in this project, we will not prioritize it. 


## Reference

[1] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. *In Advances in neural information processing systems*, pages 2672–2680, 2014.  
[2] S. Iizuka, E. Simo-Serra, and H. Ishikawa. Globally and locally consistent image completion. *ACM Transactions on Graphics (TOG)*, 36(4):107, 2017.  
[3] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang. Generative image inpainting with contextual attention. *In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.  
[4] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu and T. Huang, "Free-Form Image Inpainting With Gated Convolution," *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 4470-4479, doi: 10.1109/ICCV.2019.00457.  
[5] Nazeri, Kamyar, et al. "Edgeconnect: Generative image inpainting with adversarial edge learning." *arXiv preprint arXiv*:1901.00212 (2019).  

---