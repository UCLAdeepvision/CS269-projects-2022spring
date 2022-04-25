---
layout: post
comments: true
title: Image Captioning with CLIP
author: team14
date: 2022-04-10
---


>
Image captioning is a fundamental task in vision-language understanding, which aims to provide a meaningful and valid caption for a given input image in a natural language. Most existing image captioning model rely on pre-trained visual encoder. CLIP is a neural network which demonstrated a strong zero-shot capability on many vision tasks. In our project, we want to further investigate the effectiveness of CLIP models for image captioning.
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Background
Image captioning is a fundamental task in vision-language understanding, which aims to provide a meaningful and valid caption for a given input image in a natural language. The general pipeline is composed of a visual encoder and a language model. Visual encoders should provide an effective representation of the visual content and the goal of  language models is to predict the probability of a given sequence of words to occur in a sentence. Most existing methods rely on pre-trained visual encoders. From existing Vision and Language models and experiments, we observed that large-scale pre-training usually can lead to an improvement in generalization performance of the model. The recently proposed large-scale pre-trained neural network, CLIP, has demonstrated a strong zero-shot capability on many vision tasks. As a result, we propose to use CLIP’s image encoder as the visual encoder in the image captioning task.

## Related work
### CLIP
CLIP is a neural network which efficiently learns visual concepts from natural language supervision resulting in rich semantic latent space shared by both visual and textual data.It is trained on 400M image-text pairs crawled from the Internet which required little human annotation. It demonstrated a strong  zero-shot capability on vision tasks, and was popularly adopted in many vision and language models.  
![YOLO]({{ '/assets/images/team14/CLIP.jpeg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. We then use this behavior to turn CLIP into a zero-shot classifier. We convert all of a dataset’s classes into captions such as “a photo of a dog” and predict the class of the caption CLIP estimates best pairs with a given image.*
<br/><br/>

### CLIP-VIL
CLIP-VIL uses CLIP as the visual encoder in various V&L models in two typical scenarios: 1) plugging CLIP into task-specific fine-tuning; 2) combining CLIP with V&L pre-training and transferring to downstream tasks. The architecture consists of visual feature extraction from the pretrained CLIP model, and a single transformer taking the concatenation of visual features and text embeddings as inputs, as the architecture and training procedure shown in following the figure. In this paper, the author experiment the effectiveness of the CLIP model for image captioning with a variant of self-critical sequence training, which is a reinforcement-learning-based image captioning method.
![YOLO]({{ '/assets/images/team14/Clip-ViL.jpeg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. The training process of a V&L model typically consists of three steps*.

## Method
Based on the results obtained from CLIP-VIL, we observed that using CLIP’s visual encoder as a source of visual features can outperform other widely-used region-based methods and grid-based methods. As a result, we want to further investigate the effectiveness of the CLIP model as image encoder combining with attention-based image captioning methods, such as Object Relation Transformer[3] and Self Attention network for image captioning[4].
1. Replace the visual encoder (features) in the traditional image captioning models with CLIP’s visual encoder (features).
2. (optional, depending on whether the initial language model is pre-trained or not) Vision and language pre-train the model.
3. Train and test the new model on the same specific task, and compare with the original language model.

## Experiment
### Dataset
#### MS COCO
The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images. COCO Captions contains over one and a half million captions describing over 330,000 images. For the training and validation images, five independent human generated captions are be provided for each image.

### Metrics
By referencing from the Clip-ViL paper, we will use the standard automatic evaluation metrics including CIDEr[5], BLEU[6], and METEOR[7].

### Expected result
We want to show that the image captioning task can benefit from the CLIP model as a visual encoder by showing that the image captioning models with CLIP as image encoder has better performance than the original image captioning models.

## Schedule
Week 5: Do the experiments in ClIP models and CLIP-ViL to get the baseline.

Week 6: Research on what attention-based image captioning models we could try.

Week 7 and 8: Implement the image captioning models with CLIP.

Week 9: Complete the experiments and evaluation.

Week 10: Final presentation and report.

## Reference

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning Transferable Visual Models From Natural Language Supervision,” arXiv preprint arXiv:2103.00020, 2021.

[2] S. Shen, L. H. Li, H. Tan, M. Bansal, A. Rohrbach, K.-W. Chang, Z. Yao, and K. Keutzer, “How Much Can CLIP Benefit Vision-and-Language Tasks?” arXiv preprint arXiv:2107.06383, 2021.

[3] S. Herdade, A. Kappeler, K. Boakye, and J. Soares, “Image Captioning: Transforming Objects into Words,” in NeurIPS, 2019.

[4] L. Guo, J. Liu, X. Zhu, P. Yao, S. Lu, and H. Lu, “Normalized and
Geometry-Aware Self-Attention Network for Image Captioning,”
in CVPR, 2020.

[5] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. 2016. Spice: Semantic proposi- tional image caption evaluation. In European confer- ence on computer vision, pages 382–398. Springer.

[6] Kishore Papineni, Salim Roukos, Todd Ward, and Wei- Jing Zhu. 2002. Bleu: a method for automatic eval- uation of machine translation. In Proceedings of the 40th annual meeting of the Association for Compu- tational Linguistics, pages 311–318.

[7] Alon Lavie and Abhaya Agarwal. 2007. Meteor: An automatic metric for mt evaluation with high levels of correlation with human judgments. In Proceed- ings of the second workshop on statistical machine translation, pages 228–231.


---
