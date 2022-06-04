---
layout: post
comments: true
title: Zero-shot weakly Supervised Localization using CLIP
author: Jingdong Gao, Jiarui Wang
date: 2022-04-19
---


> Recent works in computer vision have proposed image-text foundation models that subsume both vision and language pre-training. Methods such as CLIP, ALIGN, and COCA demonstrated success in generating powerful representations for a variety of downstream tasks, from traditional vision tasks such as image classification to multimodal tasks such as visual question answering and image captioning. In particular, these methods showed impressive zero-shot capabilities. CLIP and COCA was able to achieve 76.2\% and 86.3\% top-1 classification accuracy on Imagenet, respectively, without explicitly training on any images from the dataset. Motivated by these observations, in this work, we explore the effectiveness of image-text foundation model representations in zero-shot object localization. We propose a variant of Score-CAM that generates a saliency map for an image conditioning on a user-provided textual query, from intermediate features of pre-trained image-text foundation models. When the query asks for an object in the image, our method would return the corresponding localization result. We quantitively evaluate our method on the ImageNet validation set, and demonstrate comparative ground truth localization accuracy with state of the art of weakly supervised object localization methods. We also provide a streamLit interface that enable users to experiment with different image and text combinations. The code is released on Github.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

 

## Introduction

Deep neural networks have achieved notable success in the field of object recognition. However, the most accurate models require large number of annotations. For the object localization task that seeks to find the area of the object of interest in an given image, fully supervise methods need instance level labels such as bounding boxes and pivot points, which are highly costly. This leads to the emergence of weakly supervised object localization. With only image-level supervision, it can still achieve good localization accuracy. While traditional weakly supervised localization methods provides an alternative that saves annotation time, in the test time, they are limited to predict on the predetermined object classes observed during training. Therefore, to locate novel object categories, the model has to be finetuned on a new dataset. Zero-shot localization methods[7, 8] aim to address this issue and achieve comparative localization results for out of distribution labels.

Recent works in computer vision have proposed image-text foundation models that subsume both vision and language pre-training. Methods such as CLIP, ALIGN, and COCA demonstrated success in generating powerful representations for a variety of downstream tasks, from traditional vision tasks such as image classification to multimodal tasks such as visual question answering and image captioning. In particular, these methods showed impressive zero-shot capabilities. CLIP and COCA was able to achieve 76.2\% and 86.3\% top-1 classification accuracy on Imagenet, respectively, without explicitly training on any images from the dataset. Motivated by these observations, in this work, we explore the effectiveness of image-text foundation model representations in zero-shot object localization. We propose a variant of Score-CAM that generates a saliency map for an image conditioning on a user-provided textual query, from intermediate features of pre-trained image-text foundation models. When the query asks for an object in the image, our method would return the corresponding localization result. We quantitively evaluate our method on the ImageNet validation set, and demonstrate comparative ground truth localization accuracy with state of the art of weakly supervised object localization methods. While our framework may be applied to image-text models in general, in this report we build our implementation on top of CLIP, since we do not have access to other pre-trained models at the time of this work. We examine internal features of both the CNN and ViT architecture for CLIP's image encoders, and evaluate the influence of different features on our localization objective, both qualitatively and quantitively. We also provide an streamLit interface that enable to users to experiment with different image and text combinations. The code is released on Github.



## Related Literature
We plan to use CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)) and ScoreCAM[3] as our main backbone. To tackle the weakly supervised localization problem, the pipeline combines zero-shot generalizability enabled by CLIP and saliency map approaches that handle the localization task. Addtionally, we may focus on saliency approaches originate from two directions: CNN based image encoders and vision transformer based encoders, since CLIP pretrained both. For CNN based encoders, we may will adapt Resnet50 and Resnet101.


### CLIP

CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf)) combines zero shot transfer and natural language supervision. After pretraining, CLIP is supposed to have competitive performance on other non-predetermined image categories. The overall workflow is as following: CLIP learns an enumorous amount of images and their corresponding labels. And then, different classification tasks will be applied to CLIP models. For each test image, text description "a photo of category" will be provided in order to find the nearest answer. 

The detailed structure is shown in Fig.1. Model consists of an image encoder and and text encoders. Inputs are image, text pairs where texts are image captions. The model is trained with contrastive loss where matching pairs are positive examples and unmatched pairs are negative examples. And then, inner product of text embeddings and image embeddings will be calculated for training purpose.

During a new classification test, given a set of labels and input image, CLIP uses the pretrained text encoder to embed the labels. Then compute the inner product between the text embeddings and the image embedding from the image encoder. Finally, the class that generates the highest similarity score will be the prediction answer. 

![CLIP]({{ '/assets/images/team03/final/3.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 1. CLIP's structure. From CLIP ([Radford, et al. 2021](https://arxiv.org/pdf/2103.00020.pdf))*

CLIP pretrained image classification model with language supervision. Then by combining CLIP with class activation/attention map methods, we assume the activated area can be regarded as localization on novel datasets.


### Score-CAM

Score-CAM ([Wang, et al. 2020](https://arxiv.org/pdf/1910.01279.pdf)) is a popular method to generate saliency maps for CNN based image classifiers. It often produces more accurate and stable results than gradient based methods from experimental results. According to Fig. 2, given an input image, score cam takes the activation maps from the last convolution layer, and uses these maps as masks over the the input image. The masked input images are processed by the CNN again to generate a weight for each activation map. The weights are normalized the weighted sum of the maps is used as the final output. 
![Score-CAM]({{ '/assets/images/team03/final/4.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 2. Score CAM's structure. From Score-CAM ([Wang, et al. 2020](https://arxiv.org/pdf/1910.01279.pdf))*

More detaily, given an input image and a CNN based image processor f, ScoreCAM will use feature maps from convolutional layer of f as a mask and applied the mask over the input. With regard to the masked image, ScoreCAM uses the image processor to process the masked image and use the logit for class c as the feature map's contribution score to c. In the end, the weighted sum of the feature maps are calculated using the contribution scores. The result will be a saliency map for class c.


Comparing to saliency methods that utilizes gradients, Score-CAM adapts to CLIP more naturally due to its simple implementation. We can naively adapt the method to CLIP by changing the definition of the score from the logit of the target class to the inner product between the masked input image embedding resulted from the CLIP image encoder and the text embedding generated by the text encoder. However, this naive adaptation can lead suboptimal result as shown in Figure 5 and Figure 6. Therefore, we would need come up with more sophisticated strategies by conducting and analyzing more experiments. 


### ViT
ViT is a competitive alternative to convolutional neural networks to perform vision processing tasks. Transformers measure the relationships between pairs of input tokens. Image is split into patches of fixed-size images. With position embeddings, the corresponding vectors will be sent to a standard transformer encoder.

As shown in Fig 3, token embedding in layer l-1 will be first linear projected to query, key, value pairs. Then we will perform multihead attention and inter token attention sequentially to generate layer l's embedding.
![Vit]({{ '/assets/images/team03/final/vit.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 3. ViT's structure.*

## Methodology

### Pipieline

Given an image, our pipeline will use CLIP pretrained image encoder to process the image. Feature map will be hooked out as mask and applied to input image to serve as masked image. Then masked images will be sent to image encoder to calculate the feature maps. The image's label will be transformed into logit number and serve as the contribution score of corresponding category. Finally, we multiply the feature maps with label text embeddings to get the saliency map. The saliency map shows the most activated areas for the given label. With certain threshold, common component can be sliced out to serve as the localization bounding box. 


![Pipieline]({{ '/assets/images/team03/final/pipeline2.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 4. Our method pipeline.*

As shown in Fig.5, we use a base image as well. Every pixel of base image is 0 and it generates weighted scores wb. For input image in test dataset, if certain label's weighted score is smaller then the weighted score of base image, we will ignore this label. 


![Pipieline, base image]({{ '/assets/images/team03/final/pipelinebasic.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 5. Our method pipeline's base image.*



### Image Encoder CNN
CLIP has several pretrained image encoder. For CNN, it provides ResNet 50[9], Resnet 101, Resnet 50 * 4, etc. Typically, activation layer is hooked from last fully connected layers of the model.

### Image Encoder ViT
Activation map of ViT is not as obvious as in CNN because of its structure. Several methods are tested to check the performance of feature extracting.

As shown in Fig 5, we tested four methods seperately to be activation maps. Inter-token attention layer, token embedding layer and query, key, value(QKV) layer detect features sparsely. There are 12 maps in each method's layer, and we only use the first 4 layers for visualization and test. For intern token attention,the most activated area is sparse and doesn't match with the input image's features in human understanding. Same applies to token embedding and QKV method. After combining PCA method with QKV, the result map still not satisfying but can extract the human's head to some extent. In our experiment, we choose to use QKV with PCA as the activation layers for ViT image encoder.


![ViT activation map]({{ '/assets/images/team03/final/feature.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 5. Four methods are experimented to be .*


## Data and Evaluation Plan

Due to computation limit, we use ImageNet 2012 validation dataset. The experiments focus on first 10000 images and 200 classes. 

There are 4 evaluation metrics. $box_iou = \frac{area of intersection}{area of union}$. In imagenet, there are images contain several ground truth bounding boxes. We will calculate box_iou with regard to any pair of predicted target and ground truth target, then pick the largest one out to be the result. $box_iou_final = max([box_iou(gt_i, predict_j) i \in m, j \in n])$. If box_iou of the image is larger than 0.5, it is considered as true localization. For top 1 localization accuracy, box_iou needs to be larger than 0.5 and the predicted top 1 label should be the correct label. Top 5 localization accuracy means the box_iou needs to be larger than 0.5 and correct label should fall into top 5 labels.



## Experiment Analysis 
As shown in Fig. 6, our pipeline reach 66.33% for ground truth localization accuracy. The data from first 7 columns comes from Eunji Kim[10]. Considering this method is zero shot localization, it beats first 4 methods using weakly supervised localization method. However, there are some situiations in our favor. First, our pipeline only tests over the first 10000 images in ImageNet because of computation limit. The overall accuracy may be lower if the test is performed over the whole dataset. Secondly, for ground truth localization accuracy, it is assumed that correct label is given. The same definition applies to other tested method of GT-know localization accuracy. However, this is not as similar as classical localization definition. Classical localization task predicts both classification and localization result. Thus, top 1 localization accuracy and top 5 localizatio accuracy are more suitable for classical localization task for comparision. If the user gives certain label input during test, ground truth localization accuracy will be suitable. 


![Experiment comparision with previous work]({{ '/assets/images/team03/final/experiment-others.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 6. Experiment comparision with previous work.*


Specific cases are shown in Fig. 7 and Fig. 8. During these cases, user gives correct label input to check its localization ability. For successful cases, we can see the most activated area focuses on the interested object and captures features, such as head and body pretty well. However, there are also cases focusing on the non-ideal features. For instance in Fig 8's second column, the most activated area is somewhere on the wall. But the interested object is the dog.

![Promising result]({{ '/assets/images/team03/final/good-performance.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 7. Good performance experiment result.*


![Non-ideal result]({{ '/assets/images/team03/final/bad-performance.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 8. Non-ideal performance experiment result.*



Fig 9 shows detailed experiment result using different backbone and CAM threshold. ResNet101 performs better than ResNet50 and ViT in terms of ground truth localization accuracy 68.57%. When it comes to both classification and localization, Resnet50 performs best. ViT is not ideal at all because the method extracting activation map, which is QKV with PCA, is brute force. The activation map fails to identify the entire entity as interested object. Therefore, the predicted bounding box is sparse and small, which causes the box_iou to be smaller than 0.5 and localization accuracy to be low.

![Experiment result]({{ '/assets/images/team03/final/experiment-self.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
*Fig 9. Overall experiment result.*




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
