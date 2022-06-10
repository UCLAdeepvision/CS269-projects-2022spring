---
layout: post
comments: true
title: Virtual Try-on on Videos
author: Manish Reddy Gottimukkula, Shardul Shailendra Parab, Vishnu Vardhan Bachupally
date: 2022-04-23
---

> Image-based Virtual Try-On focuses on transferring a desired clothing item on to a person's image seamlessly without using 3D information of any form. A key objective of Virtual Try-On models is to align the in-shop garment with the corresponding body parts in the person image. The problem at hand becomes challenging due to the spatial misalignment between the garment and the person's image. With the recent advances of deep learning and Generative Adversarial Networks(GANs), intensive studies were done to accomplish this task and were able to achieve moderately succesful results. The subsequent task in this direction would be is to apply the Virtual Try-On on videos. This has many applications in fashion, e-commerce etc sectors. We have started by using an existing state of the art image virtual tryon model and included various state of the art techniques to improve the performance of the videos. We have included Flow obtained from Flownet model to improve the overall smoothness of the video. Previously in video virtual tryon tasks, depth has never been taken into consideration to improve the video quality. In a novel approach, To improve the fitting of the cloth, we have used depth information and trained on various models including ResNet, DenseNet and CSPNet. The video quality has improved after adding these training tasks. Finally we have augmented the dataset by adding different backgrounds in the videos and trained on the above models to understand the effect of background in virtual tryon.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Motivation
With the rise of online shopping and e-retail platforms, Virtual Try-On of clothing garments is a key feature to enhance the experience of the online shoppers. Many intensive studies are done to solve this problem. Some initial methods focussed on image to image translation models without spatial aligment. Later some studies addressed the spatial alignment by performing garment warping using local appearance flow. One of latest studies also proposed a global appearance flow using StyleGANs for better spatial adjustment. So overall, a lot of work has been done in the are of image based Virtual Try-On and results have suprassed the minimum satisfiable requirements. While there are still room for improvement in image based Virtual Try-On methods, generating videos with Virtual Try-On is also very relevant in e-commerce and should be addressed hand in hand with image based approaches. Most existing video-based virtual try-on methods usually require clothing templates and they can only generate blurred and low-resolution results. Very few methods exist which uses relatively new versions of GAN to solve the video-based virtual try-on problem. Hence in this project, we would like to address this problem by taking a closer look of existing approaches and solving the problems with them.

## Related Work
Apart from the previously mentioned related work in this direction, we would like to particularly highlight few recent papers which are directly related to this project. 
[[1]](#references) focuses on fitting an in-shop garment into a clothed person image for which garment warping is the core step. Existing methods face the problem of failing difficult body poses,occlusions and large mis-alignments between person and garment images. The authors propose a novel global appearance flow estimation model. This helps us leverage a global style vector which encodes the whole-image context.

[[2]](#references) tries to tackle temporal incoherency for using GANs to tackle video genration.
The paper uses tendency of neural networks to learn low frequency functions,
the natural alignment of StyleGAN  to provide a strongly consistent prior. It is observed that the technique does produce highly consistent manipulations with respect to the temporal domain as well as is highly generalisable. 

StyleGAN is one of the most well studied generative models and results in impressive performance in image generation. [[3]](#references) explores the recent styleGAN3 architecture and its latent space patterns. [[1]](#references) also uses StyleGAN based architecture for appearance flow estimation. We could also potentially use styleGAN to generate intermittent frames in video generation as necessary.

## Proposal
Our proposal is to combine the work previously done in image to image Virtual Try-On and addressing temporal coherency in video generation methods to result in high quality video Virtual Try-On output. Below are the brief steps:
- Given an in-shop garment image and a person's image, align the garment on to the person using existing dense appearance flow estimation methods.
- Naive approach for video generation : Take frames from input video, applying image-to-image virtual try-on algorithm and stitch the frames. This is going to result in temporal misalignments due to independent stitching. Find ways to estimate the decoherency in the output.
- Address temporal coherencey : Generate output video from input combining virtual try-on and adjusting time based frames flow (as proposed in [[2]](#references)) to smoothen the temporal misalignments.

## Datasets
- **Original Dataset:**

  - VVT - dataset designed for video virtual try-on task, contains 791 videos of fashion model catwalk.
  - Most widely used videos dataset for virtual try on applications.
  - All videos have white background and hence models might overfit the background information while detecting the cloth position.

- **VVT subset dataset used for Training:**

  Considering the computation power and timelines, we have hand picked 40 videos which has an average of 250 frames per video. So in total we had trained our models on around 10000 frames. Out of which 30 videos data is used for training and 5 videos each are used for validation and test. Frame size in these videos is (256, 192). 

- **Augmented VVT dataset with non-plain backgrounds**:

  All the videos present in the original VVT dataset have white background and hence we have observed many artifacts when we test on  outdoor videos. So we have augmented the original VVT dataset by adding different backgrounds. We have trained a `person-segmentation` model to segment the person in the foreground and thus replacing the background seamlessly with different backgrounds. Here are the details of the `person-segmentation` model we have trained for foreground-background separation:

  - Person Segmentation Model details:

    - ConvNet based model

    - Reference: `[person-segmentation](https://github.com/dkorobchenko-nv/person-segmentation)`

    - Trained on COCO segmentation (person) dataset for 100 epochs.

      


  Here are the results from the person segmentation model:

  ![]({{ '/assets/images/team10/person_segmentation_result.png' | relative_url }})

  Figure1: a) Original frame b) Person segmentation result c) foreground mask

  

  Once the foreground mask is detected, we take background from a different image and combine it to result in the new augmented image. Here are some of the augmentation results:

  ![]({{ '/assets/images/team10/background_addition_results.png' | relative_url }})

  Figure2: 	a1) original 	a2) with background 	b1) original 	b2) with background

  

- **Generating DepthMap training data:**

  

## Timeline
- Week 1-2: Perform extensive research on possible options in the different modules
  and shortlist a few unique and relevant problems.
- Week 3: Research available datasets and codebases and understand
   the various challenges to solve the shortlisted problems and zero down 
  one option.
- Week 4-5: Work on running and understanding available codebases for the three 
 papers and isolate salient features from them. Provide prelimnary results for 
  the virtual try-on model and understand the challenges in to combine ideas.
- Week 6-8: Combine temporal stitching and style warping ideas together. Write code
 for various dataloaders and also the combined models and utilities. Train models.
- Week 9-10: Improve model, obtain results and write the report and presentation.

## Baseline runs
- We have done tests using the pretrained checkpoints provided by Style-Based global appearance flow paper [[1]](#references) [github repo](https://github.com/SenHe/Flow-Style-VTON/). This is to do image based virtual try-on over VITON dataset. Below are some of the results we have obtained:

![]({{ '/assets/images/team10/res1.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
![]({{ '/assets/images/team10/res2.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
![]({{ '/assets/images/team10/res3.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Baseline results for image based virtual try-on using model proposed in* [[1]](#references).

## References
[1] He, S., Song, Y.-Z., and Xiang, T. Style-based global ap-
pearance flow for virtual try-on, 2022. URL [https://arxiv.org/abs/2204.01046](https://arxiv.org/abs/2204.01046).

[2] Tzaban, R., Mokady, R., Gal, R., Bermano, A. H., and
Cohen-Or, D. Stitch it in time: Gan-based facial editing
of real videos, 2022. URL [https://arxiv.org/abs/2201.08361](https://arxiv.org/abs/2201.08361).

[3] Alaluf, Y., Patashnik, O., Wu, Z., Zamir, A., Shechtman,
E., Lischinski, D., and Cohen-Or, D. Third time’s the
charm? image and video editing with stylegan3, 2022.
URL [https://arxiv.org/abs/2201.13433](https://arxiv.org/abs/2201.13433).

[4] Zhong, X., Wu, Z., Tan, T., Lin, G., and Wu, Q. MV-TON:
Memory-based video virtual try-on network. In Proceed-
ings of the 29th ACM International Conference on Multi-
media. ACM, oct 2021. doi: 10.1145/3474085.3475269.
URL [https://doi.org/10.1145%2F3474085.3475269](https://doi.org/10.1145%2F3474085.3475269).

[5] Han, X., Wu, Z., Wu, Z., Yu, R., and Davis, L. S. Viton: An
image-based virtual try-on network, 2017. URL [https://arxiv.org/abs/1711.08447](https://arxiv.org/abs/1711.08447)
