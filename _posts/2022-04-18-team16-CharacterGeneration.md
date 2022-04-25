---
layout: post
comments: true
title: Pokemon GAN
author: Yu-Hsuan Liu and Jiayue Sun
date: 2022-04-24
---


> Character design is a challenging process: artists need to create a new set of characters tailored to the specific game or animated feature requirements while still following the basic anatonomy and perspective rules. The key question is: can AI help human to project their ideas into concrete drawings? In this project, we will investigate 1) How well can the existing network StyleGAN generalize to design drawing generation with clear outline and high-contrast coloring 2) if adding more discriminator branches of different drawing tasks can further disentangle the GAN latent space and make it more human controllable. 

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Motivation
To create a new and unique character for games, anime, etc. it takes years of the art training to master drawing skills and digital art software for virtualizing the design ideas. Even acquiring these skills, the process of designing a character takes days, even months, to refine and finalize the design. If we can utilize automation, we can ease the creation process. For example, some research uses neural network based model to do automatic coloring for sketches or line art. If we provide a segmentation map, some models can generate the illustration. Models like GAN can generate characters' pictures to have a starting point for designing a character or even characters that can be directly put into practice.

The challenge is when people are trying to design a new character for new work, it is a new concept of art. There are only few data to reference. We are wondering if we can still utilize automation to help with the character design. For example, Pokemon series tends to have a unified color for a Pokemon due to the type system. Also, the line art of Pokemon is cleaner compared to Digimon. To design a new Pokemon, there are only 905 existed Pokemon for us to train. However, the design of the characters for both works is based on humans, animals, plants, or items. We want to investigate if we can distill knowledge from all similar designs and apply them to new concept arts.

## Related Work

### Line Art Recognition


### Image Generation

Due to the success of StyleGAN [2] and StyleGAN2 [3] on photo and anime-style art generation, we are going to focus on the extension and analysis of StyleGAN in this process. In this section we will introduce what is StyleGAN, StyleGAN2 and the direction discovery can be performed on StyleGAN. 

![StyleGAN Architecture]({{ '/assets/images/team16/StyleGAN_architect.png' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 1. Traditional GAN vs. StyleGAN structure. (Image source: <a> https://arxiv.org/abs/1812.04948 </a>)</i>
</div>

Unlike the traditional GAN where the generator takes an random latent input $z$, StyleGAN genereator takes a constant input of size 4x4x512 and its latent input $z$ is only used to generate the styles. Specifically, $z$ is mapped to an intermediate late space $\mathcal{W}$, whose vector $w$ will be affine transformed into style $y$. We can see in figure 1 that these feature map-specific styles are used to modulate adaptive instance normalization (AdaIN) operations.

Another type of input for StyleGAN generator is the noise image injected into each layer of the network (module B in StyleGAN). These noise are introduced to generate stochastic scale-specific details (i.e. hair and freckels) into the image generation.

### Latent Space Analysis
In the lecture module, we've seen the work SeFa from Shen and Zhou, which is the PCA of 



## Contribution
1. Curate a clean collection of Pokemon image data from all 8 generations along with metadata in the game. The labels will include line art, part segmentation mask and pokemon metadata labels. 
2. Analyze how well StyleGAN performed on the Pokemon dataset after direct fine-tuning or learning the styles from pokemon images instead of random input.
3. Extend existing StyleGAN model with more discriminator branches of different prediction tasks and analyze how the latent space of StyleGAN change: is it more disentangled? 
4. Possible Extension: see if we can allow human input to control line art and coloring discretely as separate steps. 


## Dataset

| Dataset                 | Image Number| Tasks       |
| :---                    | :---:        |    :----:   | 
| Veekun Sprites 256x256  | 819         | [kaggle link](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)|
| Images without label    | 7357        | [kaggle link](https://www.kaggle.com/datasets/djilax/pkmn-image-dataset)       |
| Official Art mix sizes  | 833         | [kaggle link](https://www.kaggle.com/datasets/daemonspade/pokemon-images)|
|Images with label        | 10K+        | [kaggle link](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) |



## Current Progress
### Data Preprocessing
We are working on combining the images from different dataset. We have to unify the size of the image and the format of the image among different datasets. 


### Sketch
We have successfully run HED [1] to extract line art from the Pokemon image.

![Pokemon Line Art]({{ '/assets/images/team16/HED_result.jpg' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 2. Pokemon line art generated by HED on Veekun Sprites</i>
</div>


### StyleGAN
We have succesfully trained StyleGAN on pokemon sprites (256x256 images) and get some preliminary results. Some sample generated images are shown in the figure below:

![Generated Pokemon]({{ '/assets/images/team16/uncurated_pokemon.png' | relative_url }})
{: style="width: 600px; max-width: 100%; display: block; margin-left: auto; margin-right: auto;"}
<div style="text-align: center;">
  <i>Fig 3. Pokemon generated by StyleGAN trained on pokemon sprites only</i>
</div>




## Plan

|               | Tasks       |
| :---          |    :----:   | 
| Week 3        | Project Proposal Presentation |
| Week 4        | Decide Topic and Project Proporsal Report  |
| Week 5        | Clean Data and Test Existed Models |
| Week 6        | Train Models |
| Week 7        | Fine-tune & Adjust Models  |
| Week 8        | Analyze Model Latent Space |
| Week 9-11     | Prepare Project Presentation and Finish Project Report |


## Reference

[1] Xie, Saining and Tu, Zhuowen. "Holistically-Nested Edge Detection" *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2015.

[2] Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

[3] Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.





---
