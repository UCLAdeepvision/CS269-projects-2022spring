---
layout: post
comments: true
title: Pokemon GAN
author: Yu-Hsuan Liu and Jiayue Sun
date: 2022-06-10
---


> Character design is a challenging process: artists need to create a new set of characters tailored to the specific game or animated feature requirements while still following the basic anatomy and perspective rules. The key question is: can AI help humans to project their ideas into concrete drawings? In this project, we will investigate (1) How well can the existing network StyleGAN generalize to design drawing generation with a clear outline and high-contrast coloring. (2) If adding more discriminator branches of different drawing tasks can further disentangle the GAN latent space and make it more human controllable. 

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Motivation
To create a new and unique character for games, anime, etc. it takes years of the art training to master drawing skills and digital art software for virtualizing the design ideas. Even acquiring these skills, the process of designing a character takes days, even months, to refine and finalize the design. If we can utilize automation, we can ease the creation process. For example, some research uses neural network based model to do automatic coloring for sketches or line art. If we provide a segmentation map, some models can generate the illustration. Models like GAN can generate characters' pictures to have a starting point for designing a character or even characters that can be directly put into practice.

The challenge is when people are trying to design a new character for new work, it is a new concept of art. There are only few data to reference. We are wondering if we can still utilize automation to help with the character design. For example, Pokemon series tends to have a unified color for a Pokemon due to the type system. Also, the line art of Pokemon is cleaner compared to Digimon. To design a new Pokemon, there are only 905 existed Pokemon for us to train. However, the design of the characters for both works is based on humans, animals, plants, or items. We want to investigate if we can distill knowledge from all similar designs and apply them to new concept arts.

## Related Work

### Line Art Recognition
[HED](https://openaccess.thecvf.com/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf) [1] utilizes a new edge detection algorithm. It generates line art by  image-to-image prediction with a deep learning model. The structure of HED is shown in Fig. 1. The input image will go through several convolutional layers. For each convolutional layer, HED has a side-output layer along with deep supervision to guide the side outputs towards edge predictions with the characteristics we want. Therefore, The outputs of HED are multi-scale and multi-level.

![HED Architecture]({{ '/assets/images/team16/HED_architecture.png' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 1. HED structure</i>
</div>

Other than HED, we would like to try [PhotoSketch](https://arxiv.org/pdf/1901.00542.pdf) [5]. This model is released recently and also gained a lot of attention. HED cannot capture the edge well when the colors of items are close in the image. PhotoSketch uses conditional GAN to predict the most salient contours in images and reflects imperfections in ground truth human drawing, which is different from traditional edge detection algorithms. As shown in Fig 2, PhotoSketch is trained on a diverse set of image-contour pairs that generate conflicting gradients. The discriminator will average the GAN loss across all image-contour pairs, while the regression-loss finds the minimum-cost contour to pair with the train image.

![PhotoSketch Architecture]({{ '/assets/images/team16/PhotoSketch_architecture.png' | relative_url }}) 
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 2. PhotoSketch structure</i>
</div>

### Image Generation

Due to the success of [StyleGAN](https://arxiv.org/pdf/1812.04948.pdf) [2] and [StyleGAN2](https://arxiv.org/pdf/1912.04958.pdf) [3] on photo and anime-style art generation, we are going to focus on the extension and analysis of StyleGAN in this process. In this section we will introduce what is StyleGAN and the direction discovery can be performed on StyleGAN. 

![StyleGAN Architecture]({{ '/assets/images/team16/StyleGAN_architect.png' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 3. Traditional GAN vs. StyleGAN structure (Image source: <a> https://arxiv.org/abs/1812.04948 </a>)</i>
</div>

Unlike the traditional GAN where the generator takes an random latent input $$z$$, StyleGAN genereator takes a constant input of size 4x4x512 and its latent input $$z$$ is only used to generate the styles. Specifically, $$z$$ is mapped to an intermediate late space $$\mathcal{W}$$, whose vector $$w$$ will be affine transformed into style $$y$$. We can see in Fig. 3 that these feature map-specific styles are used to modulate adaptive instance normalization (AdaIN) operations.

Another type of input for StyleGAN generator is the noise image injected into each layer of the network (module B in StyleGAN). These noise are introduced to generate stochastic scale-specific details (i.e. hair and freckels) into the image generation.

### Latent Space Analysis
Many recent research have explored methods to understand the semantics of GAN latent space. Among these works, disentangled direction or manifold discovery in GAN latent space enables human to control certain aspects of GAN generation results without retraining GAN. 

In the lecture module, we've seen [SeFa](https://genforce.github.io/sefa/) from Shen and Zhou [4], which explores the GAN latent space through finding the $$k$$ most important directions that can cause the largest variations after the projection of transformation step $$A$$. Since SeFa had relative good performance on StyleGAN trained on anime art, we are going to use it for analysis of latent space disentanglement and see whether adding new discriminator branches can increase the degree of disentanglement.

![SeFa Direction Control Results]({{ '/assets/images/team16/SeFa.png' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 4. SeFa Direction Control Results </i>
</div>



## Contributions
1. Curate a clean collection of Pokemon image data from all 8 generations along with metadata in the game. The labels will include line art, part segmentation mask, and pokemon metadata labels. 
2. Analyze how well StyleGAN performed on the Pokemon dataset after direct fine-tuning or learning the styles from pokemon images instead of random input.
3. Extend the existing StyleGAN model with more discriminator branches of different prediction tasks and analyze how the latent space of StyleGAN changes: is it more disentangled? 
4. Possible Extension: see if we can allow human input to control line art and coloring discretely as separate steps. 

## Dataset

| Dataset                 | Image Number| Link       |
| :---                    | :---:        |    :----:   | 
| Veekun Sprites 256x256  | 819         | [kaggle link](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)|
| Images without label    | 7357        | [kaggle link](https://www.kaggle.com/datasets/djilax/pkmn-image-dataset)       |
| Official Art mix sizes  | 833         | [kaggle link](https://www.kaggle.com/datasets/daemonspade/pokemon-images)|
|Images with label        | 10K+        | [kaggle link](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) |


### Data Preprocessing
We are working on combining the images from different datasets. We unify the size and the format of Pokemon images from different datasets. We also include the newest generation Pokemon images that do not exist in any current dataset.


## Current Progress

### Sketch
We have successfully run HED to extract line art from the Pokemon image.

![Pokemon Line Art]({{ '/assets/images/team16/HED_result.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 5. Pokemon line art generated by HED on Veekun Sprites</i>
</div>


### StyleGAN
We have succesfully trained StyleGAN on pokemon sprites (256x256 images) and get some preliminary results. Some sample generated images are shown in the figure below:

![Generated Pokemon]({{ '/assets/images/team16/uncurated_pokemon.png' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 6. Pokemon generated by StyleGAN trained on pokemon sprites only</i>
</div>


### Latent Space Analysis





## Reference

[1] Xie, Saining and Tu, Zhuowen. "Holistically-Nested Edge Detection" *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2015.

[2] Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.

[3] Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2020.

[4] Shen, Yujun, and Bolei Zhou. "Closed-form factorization of latent semantics in gans." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021.

[5] Li, Mengtian, et al. "Photo-Sketching: Inferring Contour Drawings from Images." *WACV*. 2019.

---
