---
layout: post
comments: true
title: Zero-shot Object Localization With Image-Text Models and Saliency Methods
author: Jingdong Gao, Jiarui Wang
date: 2022-04-19
---


> Recent works in computer vision have proposed image-text foundation models that subsume both vision and language pre-training. Methods such as CLIP, ALIGN, and COCA demonstrated success in generating powerful representations for a variety of downstream tasks, from traditional vision tasks such as image classification to multimodal tasks such as visual question answering and image captioning. In particular, these methods showed impressive zero-shot capabilities. CLIP and COCA was able to achieve 76.2\% and 86.3\% top-1 classification accuracy on Imagenet, respectively, without explicitly training on any images from the dataset. Motivated by these observations, in this work, we explore the effectiveness of image-text foundation model representations in zero-shot object localization. We propose a variant of Score-CAM that generates a saliency map for an image conditioning on a user-provided textual query, from intermediate features of pre-trained image-text foundation models. When the query asks for an object in the image, our method would return the corresponding localization result. We quantitively evaluate our method on the ImageNet validation set, and demonstrate comparative ground truth localization accuracy with state of the art of weakly supervised object localization methods. We also provide a streamLit interface that enable users to experiment with different image and text combinations. The code is released on Github.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction and Motivation

Deep neural networks have achieved notable success in the field of object recognition. However, the most accurate models require large number of annotations. For the object localization task that seeks to find the area of the object of interest in an given image, fully supervise methods need instance level labels such as bounding boxes and pivot points, which are highly costly. This leads to the emergence of weakly supervised object localization. With only image-level supervision, it can still achieve good localization accuracy. While traditional weakly supervised localization methods provide an alternative that saves annotation time, in the test time, they are limited to predict on the predetermined object classes observed during training. Therefore, to locate novel object categories, the model has to be finetuned on a new dataset. Zero-shot methods[7, 8] aim to address this issue and achieve comparative localization results for out of distribution labels without additional training. In this work, we propose a framework that combines pre-trained image-text models from web data without human annotation, and saliency methods, for the zero-shot object localization task.

Recent works in computer vision have proposed image-text foundation models that subsume both vision and language pre-training. Methods such as CLIP, ALIGN, and COCA demonstrated success in generating powerful representations for a variety of downstream tasks, from traditional vision tasks such as image classification to multimodal tasks such as visual question answering and image captioning. In particular, these methods showed impressive zero-shot capabilities. CLIP and COCA was able to achieve 76.2\% and 86.3\% top-1 classification accuracy on Imagenet, respectively, without explicitly training on any images from the dataset. Motivated by these observations, in this work, we explore the effectiveness of image-text foundation model representations in zero-shot object localization. We propose a variant of Score-CAM that generates a saliency map for an image conditioning on a user-provided textual query, from intermediate features of pre-trained image-text foundation models. When the query asks for an object in the image, our method would return the corresponding localization result. We quantitively evaluate our method on the ImageNet validation set, and demonstrate comparative ground truth localization accuracy with state of the art of weakly supervised object localization methods. While our framework may be applied to image-text models in general, in this report we build our implementation on top of CLIP, since we do not have access to other pre-trained models at the time of this work. We examine internal features of both the CNN and ViT architecture for CLIP's image encoders, and evaluate the influence of different features on our localization objective, both qualitatively and quantitively. We also provide an streamLit interface that enable to users to experiment with different image and text combinations. The code is released on Github.

## Related Literature

### Image-Text Foundation Models

A recent line of research in computer vision focuses on pre-training with image and text jointly to learn multimodal representations. One paradigm of works demonstrated success by training two encoders simultaneuously with contrastive loss. The dual-encoder model consists of an image and text encoder and embed images and texts in the same latent space. These models are typically trained with noisy webly data that are image and text pairs. Another line of work follows an encoder-decoder architecture. During pre-training, the encoder takes images as input, and the decoder is applied with a language modeling loss. As a further step, the most recent work combines both architectures and achieves the state of art results on a variety of vision and language tasks.

In this work, we utilize CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)), an image-text foundation model with a dual-encoder architecture. The encoders are trained in parallel with web collected images and corresponding captions. During training, CLIP aligns the outputs from the image and text encoder in the same latent space. 

The detailed structure is shown in Fig.1. For each training step, a batch of n image and text pairs is sampled. The matching pairs are treated as positive examples, and unmatching are negative examples. Then the model computes the inner product of embeddings from the encoders for each pair and applies a contrastive loss.

In the test time, given a set of possible labels and an input image, CLIP places the each label within a prompt that can be in the form of "an image of a \[label\]." and uses the pretrained text encoder to embed the prompts. Then it computes the inner product between the text embeddings and the image embedding from the image encoder. Finally, the label whose prompt generates the highest similarity score will be the prediction answer. 

![CLIP]({{ '/assets/images/team03/final/3.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 1. CLIP's structure. From CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf))*

### Saliency Methods
Saliency methods such as CAM and Grad-CAM have shown to generate class dependent heapmaps that highlight areas containing objects of the given class in an input image from intermediate outputs of pre-trained image classification models, such as VGG and ResNet. The generated saliency maps can be directly used to perform localization tasks by drawing bounding boxes the most highlighted region.

Score-CAM ([Wang, et al. 2020](https://arxiv.org/pdf/1910.01279.pdf)) is a another method to generate saliency maps for CNN based image classifiers. It often produces more accurate and stable results than gradient based methods from experimental results. According to Fig. 2, given an input image, score cam takes the activation maps from the last convolution layer, and uses these maps as masks over the the input image. The masked input images are processed by the CNN again to generate a weight for each activation map. The weights are normalized the weighted sum of the maps is used as the final output. 
![Score-CAM]({{ '/assets/images/team03/final/4.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 2. Score CAM's structure. From Score-CAM ([Wang, et al. 2020](https://arxiv.org/pdf/1910.01279.pdf))*

More detaily, given an input image and a CNN based image processor f, ScoreCAM will use feature maps from convolutional layer of f as a mask and applied the mask over the input. With regard to the masked image, ScoreCAM uses the image processor to process the masked image and use the logit for class c as the feature map's contribution score to c. In the end, the weighted sum of the feature maps are calculated using the contribution scores. The result will be a saliency map for class c.

Comparing to saliency methods that utilizes gradients, Score-CAM often results in more stable saliency maps. Additionally, it adapts to CLIP naturally, by changing the definition of the score from the logit of the target class to the inner product between the masked input image embedding resulted from the CLIP image encoder and the text embedding generated by the text encoder. 

### ViT
ViT is a competitive alternative to convolutional neural networks to perform vision processing tasks. Transformers measure the relationships between pairs of input tokens. Image is split into patches of fixed-size images. With position embeddings, the corresponding vectors will be sent to a standard transformer encoder.

As shown in Fig 3, token embedding in layer l-1 will be first linear projected to query, key, value pairs. Then we will perform multihead attention and inter token attention sequentially to generate layer l's embedding.
![Vit]({{ '/assets/images/team03/final/vit.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 3. ViT's structure.*

## Methodology

### Combine Image-Text Models and Saliency Methods

To generate text dependent saliency maps from intermediate features of an image-text model, we propose VL-Score, a variant of Score-CAM that accounts for both vision and language inputs. Given an image and a class label embedding in a prompt, we first apply the image and text encoder to embed the image and text inputs into the same latent space. After this forward pass, we extract intermdiate features from a specific layer of the image encoder. For instance, if the image encoder builds upon a CNN architecture, we can extract the activations after the a specific convolutional layer l. The extracted features will be normalized to the scale between 0 and 1 and reshaped into masks that have the shape of the input image. After applying the feature masks to the original input, the masked images will be sent to image encoder to compute the masked image embeddings. These embeddings are then dot producted with the prompt's embedding to compute the similar of each feature map to the given sentence. The similarity score are then normalized and used to compute the final saliency map with the feature maps, as shown in Figure 4.


![Pipieline]({{ '/assets/images/team03/final/pipeline2.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 4. Our method pipeline.*

Similar to Score-CAM, our method uses a base image, which is by default an image whose pixel values are all zero, and computes the dot product of the base image's embedding and the prompt's embedding as the base score. However, we do not substract this base score from the scores of each feature map. Instead, we use the base score as cap such that we only keep the feature maps whose scores are higher than the base. If none of the feature map has a higher score, our method determines that the object of interest is not present in the given image. The scores that are higher than the base are put into a softmax with a temperature. The results are used to compute the weight sum of corresponding feature maps as the final output.

![Pipieline, base image]({{ '/assets/images/team03/final/pipelinebasic.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 5. Our method pipeline's base image.*

### Image Encoder CNN
In this project, we use CLIP as the image-text model since it has released pre-trained versions. In particular, it has two types of pretrained image encoders, based on CNN and ViT architectures. For CNN based encoders, it releases models including ResNet 50[9], Resnet 101, Resnet 50 * 4, etc. The definition of feature maps in the CNN architecture is natural. Typically, the features are extracted from last convolutional layer of the encoder.

### Image Encoder ViT
On the other hand, the definition of feature maps in ViT base encoders is not as obvious. Here, we explore several methods of extracting intermediate features. One way of feature extraction analagous to the case in CNN encoders is to directly use the output of a transformer block. The patches embeddings of shape n x d, where n is equal to w x h, can be reshaped into w x h x d. Then we can treat each slice of the embeddings in the last dimension as a feature. Another common way is to utilize the attention outputs from an transformer block. For a block that has T heads, there are T x n attention scores between the class token to each of the n patch tokens. Then these attentions can be reshaped into T x w x h and sliced along the last dimension. Besides directly using a transformer block's output, [] has shown that the intermediate outputs such as keys, queries, and values inside a transformer block can preserve structural information for ViT in DINO. Therefore, we also experimented with using these intermediate features and applied PCA in attempt to obtain more structured feature maps.

Fig 5 illustrates the feature maps acquired by each of the four methods. We observe that for CLIP ViT-base/16, the inter-token attention, token embedding and query, key, value(QKV) maps highlight sparsely connected regions. After applying PCA, the most principle components of query embeddings show more structured regions such as a human's head or a fish. However, within a single feature, multiple semantic regions can be highlighted simultaneuously.


![ViT activation map]({{ '/assets/images/team03/final/feature.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 5. Four methods are experimented to be .*


## Data and Evaluation Plan

Due to computation limit, we use ImageNet 2012 validation dataset. The experiments focus on validation datasets' 50000 images and 1000 classes. 

There are 4 evaluation metrics. $box_iou = \frac{area of intersection}{area of union}$. In imagenet, there are images contain several ground truth bounding boxes. We will calculate box_iou with regard to any pair of predicted target and ground truth target, then pick the largest one out to be the result. $box_iou_final = max([box_iou(gt_i, predict_j) i \in m, j \in n])$. If box_iou of the image is larger than 0.5, it is considered as true localization. For top 1 localization accuracy, box_iou needs to be larger than 0.5 and the predicted top 1 label should be the correct label. Top 5 localization accuracy means the box_iou needs to be larger than 0.5 and correct label should fall into top 5 labels.



## Experiment Analysis 
As shown in following table, our pipeline reach 66.60% for ground truth localization accuracy. The data from first 7 columns comes from Eunji Kim[10]. Considering this method is zero shot localization, it beats first 4 methods using weakly supervised localization method. However, there are some situiations in our favor. Firstly, for ground truth localization accuracy, it is assumed that correct label is given. The same definition applies to other tested method of GT-know localization accuracy. However, this is not as similar as classical localization definition. Classical localization task predicts both classification and localization result. Thus, top 1 localization accuracy and top 5 localizatio accuracy are more suitable for classical localization task for comparision. If the user gives certain label input during test, ground truth localization accuracy will be suitable.

| Method                | Backbone |  GT-known Loc | Top-5 Loc        | Top-1 Loc        |
|-----------------------|----------|---------------|------------------|------------------|
| ResNet50-CAM(cvpr 16) | ResNet50 | 51.86         | 49.47            | 38.99            |
| ADL(cvpr 19)          | ResNet50 | 61.04         | -                | 48.23            |
| FAM(iccv 21)          | ResNet50 | 64.56         | -                | 54.46            |
| PSOL(iccv 21)         | ResNet50 | 65.44         | 63.08            | 53.98            |
| I^2C(eccv 20)         | ResNet50 | 68.50         | 64.60            | 54.83            |
| SPOL(ICCV 21)         | ResNet50 | 69.02         | 67.15            | 59.14            |
| BGC (cvpr 22)         | ResNet50 | 69.89         | 65.75            | 53.76            |
| Ours                  | ResNet50 | 66.60         | 39.50(cls 58.64) | 57.20(cls 85.26) |
Table 6: Comparision with other classical work. Our pipeline's best performance can reach 66.60% ground truth localization.

Specific cases are shown in Fig. 7 and Fig. 8. During these cases, user gives correct label input to check its localization ability. For successful cases, we can see the most activated area focuses on the interested object and captures features, such as head and body pretty well. However, there are also cases focusing on the non-ideal features. For instance in Fig 8's second column, the most activated area is somewhere on the wall. But the interested object is the dog.

![Promising result]({{ '/assets/images/team03/final/good-performance.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 7. Good performance experiment result.*


![Non-ideal result]({{ '/assets/images/team03/final/bad-performance.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 8. Non-ideal performance experiment result.*



Table 9 shows detailed experiment result using different backbone and CAM threshold. ResNet101 performs better than ResNet50 and ViT in terms of ground truth localization accuracy 68.57%. When it comes to both classification and localization, Resnet50 performs best. ViT is not ideal at all because the method extracting activation map, which is QKV with PCA, is brute force. The activation map fails to identify the entire entity as interested object. Therefore, the predicted bounding box is sparse and small, which causes the box_iou to be smaller than 0.5 and localization accuracy to be low.

| Method                | Backbone |  GT-known Loc | Top-5 Loc        | Top-1 Loc        |
|-----------------------|----------|---------------|------------------|------------------|
| ResNet50-CAM(cvpr 16) | ResNet50 | 51.86         | 49.47            | 38.99            |
| ADL(cvpr 19)          | ResNet50 | 61.04         | -                | 48.23            |
| FAM(iccv 21)          | ResNet50 | 64.56         | -                | 54.46            |
| PSOL(iccv 21)         | ResNet50 | 65.44         | 63.08            | 53.98            |
| I^2C(eccv 20)         | ResNet50 | 68.50         | 64.60            | 54.83            |
| SPOL(ICCV 21)         | ResNet50 | 69.02         | 67.15            | 59.14            |
| BGC (cvpr 22)         | ResNet50 | 69.89         | 65.75            | 53.76            |
| Ours                  | ResNet50 | 66.60         | 39.50(cls 58.64) | 57.20(cls 85.26) |
Table 9: Overall experiment result.

In Fig 9, we can see CAM threshold also has influence over the localization accuracy. Fig 10 gives more detailed explaination. Heatmap has different intensity in pixel representing the activation level of the area. The higher the intensity, the color will be more red than blue, and the pixel are regarded as more activated features. Threshold and common component method will be used to find the bouding box of interested obejct. When the threashold is higher, the bounding box will be sliced smaller and more likely to be seperated into multiple boxes if read areas are seperated by yellow areas. When there are multiple ground truth instances, such as multiple hens in the example image, higher threashold can help slice the most activated areas in heatmap. However, threashold is not the higher, the better. In the upper histogram in Fig 10, for both ResNet 50 and ResNet 101, we can see when threshold = 0.6, the accuracy with known ground truth is highest compared with case threshold 0.55 and 0.65. Thus, threshold can be a hyperparameter affecting the localization accuracy.

![Box_IOU relationship with threshold]({{ '/assets/images/team03/final/threshold.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 10. Box iou's relationship with CAM threshold.*

Of course, users can try as many label inputs as they want. As shown in Fig 12, label 'man', 'face','man's face', and 'fish' are given seperately. Different label generates different contribution scores after text embedding. Then the weighted sum changes correspondingly. The sample image's ground truth is 'tench, Tinca tinca'. From language perspective, 'fish' is similar to 'tench'. Then the most activated part of 'fish' mostly matched to the fish.

![Case study cnn]({{ '/assets/images/team03/final/cnn-casestudy.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 11. Case study given different label input, with backbone ResNet101.*


Counter example will be ViT backbone as shown in Fig 12. Even with correct label "fish", it fails to capture related features as a whole.

![Case study ViT]({{ '/assets/images/team03/final/vit-casestudy.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 12. Case study given different label input, with backbone ViT.*



## Future Work
CLIP with Score-CAM is proved to be successful in zero shot object localization when using CNN image encoder. Visual explaination has other classical methods, including Grad CAM, Grad CAM++, etc. Replacing Score-CAM with other saliency method can be one future work.

Also, testing with the whole validation dataset will be helpful in understanding its overall comparision with other methods.


## StreamLit Interface
To make our work more interactive, we provide a StreamLit Interface for users to upload their own image and experiment with different text descriptions to the localization and saliency results. Note that running with CPU can be extremely slow. Detailed instructions on how to setup this interface is provided in https://github.com/mxuan0/zeroshot-localization.git. 

Figure shows an illustration of how to use the interface. The user can select their intended model, layer, and bounding box thresholds. After uploading an image and enter a text description, the program can be run by clicking 'Find it!'.

## Conclusion
CLIP with Score-CAM can perform object localization with promising accuracy, especially considering it is a zero-shot procedure. At the same time, it has the following uneligible restrictions. For instance, when there are m multiple ground truth bounding boxes, the pipeline is far from predict exactly m bounding boxes. Most cases it only predicts one whole box or two boxes. At the same time, finding suitable activation maps from ViT image encoder remains a problem. Direct extraction feature layers is brute force and has poor performance. 






## Reference
[1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

[2] Radford, Kimm et al. "Learning Transferable Visual Models From Natural Language Supervision." *Arxiv*. 2021.

[3] Wang et al. "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks." *Conference on Computer Vision and Pattern Recognition*. 2019.

[4] Zhou et al. "Learning Deep Features for Discriminative Localization." *Conference on Computer Vision and Pattern Recognition*. 2016.

[5] Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *International Conference on Computer Vision*. 2017.

[6] Chen et al. "LCTR: On Awakening the Local Continuity of Transformer for Weakly Supervised Object Localization." *the Association for the Advancement of Artificial Intelligence*. 2017.

[7] Rahman, Shafin, Salman Khan, and Fatih Porikli. "Zero-shot object detection: Learning to simultaneously recognize and localize novel concepts." Asian Conference on Computer Vision. Springer, Cham, 2018.


[8] Rahman, Shafin, Salman H. Khan, and Fatih Porikli. "Zero-shot object detection: joint recognition and localization of novel concepts." International Journal of Computer Vision 128.12 (2020): 2979-2999.

[9] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.


[10] Kim, Eunji, et al. "Bridging the Gap between Classification and Localization for Weakly Supervised Object Localization." arXiv preprint arXiv:2204.00220 (2022).

---
