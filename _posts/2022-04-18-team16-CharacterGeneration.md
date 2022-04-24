---
layout: post
comments: true
title: Post Template
author: 
date: 2022-04-18
---


> Introduction for the project(or Abstract)

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Motivation
To create a new and unique character for games, anime, etc. it takes years of the art training to master drawing skills and digital art software for virtualizing the design ideas. Even acquiring these skills, the process of designing a character takes days, even months, to refine and finalize the design. If we can utilize automation, we can ease the creation process. For example, some research uses neural network based model to do automatic coloring for sketches or line art. If we provide a segmentation map, some models can generate the illustration. Models like GAN can generate characters' pictures to have a starting point for designing a character or even characters that can be directly put into practice.

The challenge is when people are trying to design a new character for new work, it is a new concept of art. There are only few data to reference. We are wondering if we can still utilize automation to help with the character design. For example, Pokemon series tends to have a unified color for a Pokemon due to the type system. Also, the line art of Pokemon is cleaner compared to Digimon. To design a new Pokemon, there are only 905 existed Pokemon for us to train. However, the design of the characters for both works is based on humans, animals, plants, or items. We are wondering if we can distill knowledge from all similar designs and apply them to new concept arts.


## Goals
1. 
2. 
3. 

## Datasets
1. Pokemon Images Dataset (https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)

2. 

## Related Work





## Current Progress

### Data Preprocessing
We are working on combining the images from different dataset. We have to unify the size of the image and the format of the image among different datasets. 


### Line Art of Image
We have successfully run HED [1] to extract line art from the Pokemon image.

(Add images later on)


### StyleGAN




## Plan
|             | Tasks    |
| :---        |    :----:   | 
| Week 3        | Project Proposal Presentation |
| Week 4        | Decide Topic and Project Proporsal Report  |
| Week 5        | Clean Data and Test Existing Models |
| Week 6        | Modify and Train Models |
| Week 7        | Modify and Train Models |
| Week 8        | Finalize Models |
| Week 9 - 11   | Prepare Project Presentation and Finish Project Report |


## Reference

[1] Xie, Saining and Tu, Zhuowen. "Holistically-Nested Edge Detection" *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2015.





---
