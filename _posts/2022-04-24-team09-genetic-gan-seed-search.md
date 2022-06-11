---
layout: post
comments: true
title: Evolutionary Search for GAN Seed Identification
author: Keli Huang and Tingfeng Xia
date: 2022-04-24
---


> Identifying subspaces of generator latent spaces for GANs is a quintessential step toward improving the understanding and training of better GANs. Spaces of interest usually correspond to certain wanted features on the output, for example the pose and orientation of human faces. We aim to introduce a class of evolutionary algorithms for the systematic discovery of these subspaces. 

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction  
GANs provide a powerful framework for training generative models. Generator trained this way takes a random seed vector from the latent space, usually sampled from a high dimensional gaussian distribution, and outputs a generated sample.  In computer vision, the output sample given a generator is usually an image for different subsequent tasks. 

Identifying subspaces of generator latent spaces for GANs is a quintessential step toward improving the understanding and training of better GANs which  has attracted a lot of research interest in recent years. It has been shown that there exists subspaces associated with pose and/or identity shift of human or objects [1]; subspaces that correspond to change of painting style [2]; and subspaces that generate catastrophic examples [3]. Discovering and understanding these subspaces will benefit us a lot by training with better GANs. 

Finding desired GAN latent subspaces that correspond to certain features Spaces of interest is also an important problem which can be done in many ways varying from heuristic method to some reversed ML method. Some papers utilize genetic algorithms to achieve style transferring tasks on human faces in a more efficient way [4] by finding the latent subspaces for two individual images.

In this work, we try to find a better generator in a human-in-the-loop fashion.  We here defined the desired subspaces as those spaces can generate some bad examples given generators. We will use the genetic algorithm to find those bad latent subspaces in a more efficient way.
After finding such subspaces, we can utilize some statistical method and linear algebra trying to exclude those spaces by changing the original  high dimensional gaussian distribution, which as a result improves the generator’s performances by some evaluation metrics as FID. Such a method should be able to work with varied (computer vision GAN) generators that transform a random noise vector into an image.

## Related Works
GAN attracts much research interest these days for its ability to generate much more data by noise vectors. StyleGAN2 [5] and StyleGAN3 [6] show its strong ability to generate images such as human faces, both by human evaluations and some predefined metrics. 

Genetic algorithm, as a kind of heuristic method, can be utilized in many non-optimization methods. Some work utilized genetic algorithms in finding a more reasonable model architecture [7]. In order to find a desired latent space in a more efficient way, CG-GAN [5] uses a genetic algorithm to identify style transferring spaces for two images. 

## Methodologies and Expected Results
In this work, we are planning to combine genetic algorithm with GAN to result in a GAN model 
with better performance. We can decompose our work into three sections.  

**ML Pipeline.**
We will implement our work based on some mature GAN pipeline, such as progressive GAN [8] and StyleGAN3 [6]. We are trying to utilize their open models and introduce some of our own modifications. After that, we can achieve a better performance in generating image tasks from metric as FID. 

**Genetic Algorithm.**
Instead of randomly sampled from a high dimensional gaussian distribution, in this work, we are trying to modify the original latent vector distribution since apparently some subspaces can largely destroy our model performance. We can use the genetic algorithm to find such bad subspaces in a more efficient way. After identifying such subspaces, we can use some mathematics-based method to tell the GAN trying to avoid sampling from those areas.

**Human-ML Interative Interface.**
Selecting the predefined bad examples by humans requires us to implement a Human-ML interact interface. We here try to achieve such human selection operations by clicking the shown figures with some predefined aims such as pure selection or adding to bad latent subspaces.

We plan on using human face datasets, such as CelebA and MetFaces [9, 10]. When generating these human face images, it is easier for us to define some bad patterns such as only one eye or no mouth. We are trying to evaluate our modified model with metric FID, which is also a widely used method to evaluate how good a generator is.

We expect to achieve a goal that, by human interaction with the ML model identifying some bad latent subspaces, the generator will sample from a more reasonable latent space generating human images, which will result in a better FID score.

## Milestones
- **Week 4 - 5:** Select and train GAN. Explore searching options. UI element build up.  
- **Week 6 - 7:** Implement and tune search.  
- **Week 8:** Search result exploration.  
- **Week 9:** Pipeline integration.  
- **Week 10:** Final presentation.  
- **Week 11:** Final report.  

## Progress Update (Sunday, Apr 24, 2022)
Out of the many types of GANs available, we have selected StyleGAN as the GAN to start experimenting with progressive GAN, which is relatively small and easy to implement. The GAN has been successfully trained and tested on our local device. We are not able to utilize GPUs on Google Colab due to the need of a local running GUI. 

We have implemented a version of Frechet Inception Distance (FID) that is portable to the GANs that we are experimenting with. FID is a metric used to compare the distribution between the generated samples and the distribution of real images in the input dataset. 

We have implemented a vanilla version of genetic search for seed discovery. However, we are still working on ways to overcome the non-uniform (gaussian, in fact) nature of the seed latent space. In particular, we hypothesize that linear interpolation in a high dimensional gaussian space would yield poor results. 

As the rest of the project is written in mostly Python, we resort to the same language for building our image selecting user interface. This UI component is designed to be maximally adaptable to any number of images input and is fully customizable in terms of the number of desired chosen images for next iteration. Below is an image of how our simple image selecting user interface looks for now. 

![Image selector ui]({{ '/assets/images/team09/image_selector_ui_ver1.png' | relative_url }}) 
{: style="max-width: 70%;"}
   
## Reference
[1] Shen, Yujun & Zhou, Bolei. (2021). Closed-Form Factorization of Latent Semantics in GANs. 1532-1540. 10.1109/CVPR46437.2021.00158.   
[2] Hiçsönmez, Samet & Samet, Nermin & Akbas, Emre & Duygulu, Pinar. (2020). GANILLA: Generative adversarial networks for image to illustration translation. Image and Vision Computing. 95. 103886. 10.1016/j.imavis.2020.103886.   
[3] Thanh-Tung, Hoang & Tran, Truyen & Venkatesh, Svetha. (2018). On catastrophic forgetting and mode collapse in Generative Adversarial Networks.  
[4] Zaltron, Nicola & Zurlo, Luisa & Risi, Sebastian. (2020). CG-GAN: An Interactive Evolutionary GAN-Based Approach for Facial Composite Generation. Proceedings of the AAAI Conference on Artificial Intelligence. 34. 2544-2551. 10.1609/aaai.v34i03.5637.   
[5] Karras, Tero & Laine, Samuli & Aittala, Miika & Hellsten, Janne & Lehtinen, Jaakko & Aila, Timo. (2020). Analyzing and Improving the Image Quality of StyleGAN. 8107-8116. 10.1109/CVPR42600.2020.00813.   
[6] Karras, Tero & Aittala, Miika & Laine, Samuli & Härkönen, Erik & Hellsten, Janne & Lehtinen, Jaakko & Aila, Timo. (2021). Alias-Free Generative Adversarial Networks.   
[7] Wang, Chaoyue & Xu, Chang & Yao, Xin & Tao, Dacheng. (2018). Evolutionary Generative Adversarial Networks. IEEE Transactions on Evolutionary Computation. PP. 10.1109/TEVC.2019.2895748.     
[8] Karras, Tero & Aila, Timo & Laine, Samuli & Lehtinen, Jaakko. (2017). Progressive Growing of GANs for Improved Quality, Stability, and Variation.      
[9] Liu, Ziwei & Luo, Ping & Wang, Xiaogang & Tang, Xiaoou. (2015). Deep Learning Face Attributes in the Wild. Proceedings of International Conference on Computer Vision (ICCV).   
[10] Karras, Tero & Aittala, Miika & Hellsten, Janne & Laine, Samuli & Lehtinen, Jaakko & Aila, Timo. (2020). Training Generative Adversarial Networks with Limited Data.   

---
