---
layout: post
comments: true
title: Measuring and Mitigating Bias in Vision-and-Language Models
author: Feiyang Chen and Zi-Yi Dou
date: 2022-04-24
---


> Models pre-trained on large amounts of image-caption data have demonstrated impressive performance across vision-and-language (VL) tasks. However, only a few of the recent works have paid attention to the social bias problem in these models. In this project, we propose to first measure whether and to what extent the biases exist in representative VL models. Then, we will investigate ways to mitigate these biases and evaluate the model on the FairFace dataset, consisting of faces with balanced race, gender, and age.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Vision-and-language (VL) tasks require models to understand both vision and language inputs. The pretraining-finetuning paradigm has proven to be effective for VL models [1]. Typically, large amounts of image-caption pairs are used to pretrain representations that contain rich multimodal information and are helpful for downstream tasks.

As pointed out by several researchers [2, 3, 4], the social biases encoded in the training data can be amplified by machine learning models and potentially harm marginalized populations. While the VL model performance has been significantly improved with increasing amounts of model parameters and training data, few of the existing works have paid attention to the social bias problem in these models. Among them, Agarwal et al. [5] presents a preliminary study on the racial and gender bias problem in the CLIP model [1]. Cho et al. [6] investigate various aspects of text-to-image generation models, including the social bias problem. However, most of these existing works lack quantitative analyses of the social biases encoded in VL models and ways to mitigate these biases.

In this project, we propose to first quantitatively analyze the biases in representative VL models. To this end, we can borrow ideas from existing work in natural language processing [7] and apply them in representative VL models such as CLIP [1] and METER [8]. Afterwards, we can investigate how we can mitigate the biases. For example, we can identify which neurons respond to gender, racial information as in [9] and if we can edit these neurons during inference time.

We can experiment on the FairFace dataset [10], which consists of people faces with balanced race, gender, and age. We can use the lexicon in [6] and compute the similarity between each face and text prompt (e.g. a photo of a doctor). The similarity differences between different genders and races can be used to quantify the biases, and we hope we can propose ways to mitigate these biases.

## Related Work
### Bias in Language Models
Works on bias in language models can be broadly divided into three sources: language representations, language understanding, and language generation. For language representations, researchers mainly focus on measuring and mitigating biases in text embedding spaces, including word [11, 12, 13, 14] and sentence embeddings [15, 16, 17]. For language understanding, existing works mostly apply bias detection and mitigation methods to some natural language understanding (NLU) tasks, such as hate speech detection [18, 19], relation extraction [20], sentiment analysis [21], and commonsense inference [22]. There are also work on addressing some bias amplification issues [3, 23]. For language generation, existing approaches mainly lie in identifying and reducing biases in the generated text of machine translation [24, 25] and dialogue generation [26, 27], as well as other natural language generation (NLG) tasks [28, 29]. Although recent works have made great progress in debiasing language models, they are still limited to the text modality.


### Bias in Vision Models
Bias in vision models mainly comes from visual recognition and image generation. Recent works [30] study the origins and prevalence of texture bias in ImageNet-trained CNNs, indicating that vision models prefer to classify images by shape rather than texture. In visual recognition, Wang et al. [31] design a simple yet effective visual recognition benchmark for studying bias mitigation, and provide a comprehensive analysis of bias mitigation techniques in visual recognition models. More recently, Chen et al. [32] focus on understanding and mitigating annotation bias in facial expression recognition and analyze systematic biases in the human annotations of public datasets. In image generation, Katja et al. [33] study on the frequency bias of generative models and provide a thorough analysis of existing explanations for systematic artifacts in the spectral statistics of generated images.


### Bias in Vision-and-Language Models
There are relatively a few works focusing on bias in vision-and-language models. Several researchers have found bias on the dataset level [34, 35, 36]. On the model level, Tejas et al. [37] study biases compound in pre-trained vision-and-language models, extending text-based bias analysis methods to multimodal language models like VL-BERT [38]. Zhang et al. [39] focus on diagnosing the environment bias in vision-and-language navigation through environment re-splitting and feature replacement, to search possible reasons for environment bias. More recently, Agarwal et al. [5] presents a preliminary study on the racial and gender bias problem in the CLIP model [1]; and Cho et al. [6] explore biases in text-to-image generative transformers and proposed two new evaluation aspects of text-to-image generation: visual reasoning skills and social biases.

## Methods
### Quantifying the Bias
We first present three ways of measuring the bias. In the following paragraphs, we assume that we use data consisting of people faces, where each instance is annotated with its corresponding race and gender information. Therefore, we can obtain images of people faces of different groups.

To measure the bias, we also need to obtain different text concepts. To this end, we can use the lexicon in [6]. Specifically, Cho et al. [6] construct four categories of words, including 85/6/39/15 profession/political/object/other words. We refer the readers to their paper for a detailed description of their constructed lexicon.

**Similarity-based Bias Measurement.** We first propose a similarity-based bias measurement method. Specifically, we can compute the similarity score between any image and text pair using a pre-trained VL model. For example, given a profession ‘doctor’, we can create a text prompt ‘a photo of a doctor’ and use it to measure the similarity between the concept of ‘doctor’ and different groups of people. If the similarity between this concept and a group of people is significantly larger or smaller than another group, we can conclude that there exists bias towards different groups of people regarding this concept.

Formally, given a text concept $$c$$, we compute the similarity between $$c$$ and all the images in the data, and compute the mean $$\mu$$ and standard deviation 
$$\sigma$$. Then, for a group of images $$\{m_i\}$$, we compute the mean of the similarities $$\mu_m$$ and normalize it with $$ p_m = \frac{\mu_m - \mu} {\sigma}$$. We finally compute the mean absolute deviation of $$mad = \sum_j |p_j - \bar{p}| $$ as in [6] and use it to quantify the bias. 

**Retrieval-based Bias Measurement.** We can also perform image-text retrieval for a given text concept. For example, given a text concept $$c$$, we can retrieve the top 100 most similar images from the dataset. Then, we can compute the proportions 
$$
p_j
$$ of people from different groups and compute the mean absolute deviation of $$ mad = \sum_j |p_j - \bar{p}| $$ as before, which is used to quantify the bias.

**Differential Association Measurement.** We can also use the Word Embedding Association Test (WEAT) [7], which is a popular metric in measuring the bias of word and sentence embeddings, to quantify the bias in VL models. For example, given two *text* concepts 
$$c_1=$$'doctor' and $$c_2=$$'nurse' as well as two *image* groups of people faces $$M=$$'male'$$=\{m_i\}$$ and $$F=$$'female'$$=\{f_i\}$$, we can measure the bias with
$$
s(c_1, c_2, M, F) = t(c_1, M, F) - t(c_2, M, F),
$$
where
$$
t(c, M, F) = \text{avg}_{m_i \in M} \text{sim}(c, m_i) - \text{avg}_{f_i \in F} \text{sim}(c, f_i),
$$
and the similarities between text concepts and images are obtained by VL models. Permutation tests can be used to measure the significance of this differential association. 

### Reducing the Bias
One of the most common causes of bias in models is that there exists bias in the training data [4]. However, it can be costly and sometimes even impossible to carefully curate an ‘unbiased’ data, and is computational to re-train or fine-tune a pre-trained model on such data. Therefore, in this project, we mainly focus on inference-time de-biasing methods.
We propose to first identify which neurons respond to gender or race inputs. Following [9], we can use both feature visualization [40] and dataset examples [41] to identify such neurons. Specifically, feature visualization maximizes the neuron’s activation by performing gradient-based optimization on inputs, and dataset examples generates the distribution of maximal activating images from a dataset for a specific neuron. After we identify such neurons, we can manually edit these neurons during inference time. For example, if a group of neurons $$\{n^1_i\}$$ responds to male inputs and another group $$\{n^2_i\}$$ responds to female inputs, we can balance their activations given an input during inference. In this way, the model can be ignorant of the gender information when it should not take it into consideration. Similar techniques such as [42] can also be explored.

## Experiments
In this section, we present the datasets and models we want to evaluate and our evaluation metric. We also present some preliminary results.

### Settings
**Datasets.** We propose to experiment on the FairFace dataset [10], which has been used to measure the bias in CLIP [5]. The dataset consists of 108,501 images of model generated faces. Each person face is annotated by its race, gender, and ages. In this project, we mainly focus on the race and gender information. The seven ethnicities included are: White, Black, Indian, East Asian, South East Asian, Middle East and Latino. The two genders included are male and female.

**Models.** We propose to experiment with two representative VL models, including CLIP [1] and METER [8]. CLIP is trained with a image-text contrastive objective on large image-text corpora. Its image and text encoder are independent of each other except that we treat the dot-products between the image and text representations on the top as image-text similarities. The METER model, on the other hand, fuses the image and text encoders at the top layers, thus the image and text modalities entangle with each other in the backbone, which can be helpful for a wider range of VL tasks.

**Evaluation Metrics.** We can use the three proposed methods to quantify the bias as in Section 3. We can first use the lexicon in [6] to measure the biases and then investigate if our methods can reduce the biases as reflected in our metrics. We also need to compute if the retrieval accuracy will be affected when we use debiasing methods.

### Preliminary Results
We first perform study on the gender and racial bias issues in the CLIP-ViT-224/32 model on the FairFace validation set using both similarity-based and retrieval-based methods.

| Term | Male || Female || Bias Measurements |
|---------------------------|-------------------------------|---------------------------------|-------------------------------------------|
|                           | Norm. Sim.                | Rt. Per.                   | Norm. Sim.                            | Rt. Per. | Sim. | Rt. |
| doctor                    | 0.2148                        | 0.89                            | -0.2412                                   | 0.11         | 0.4560   | 0.78     |
| nurse                     | -0.5075                       | 0.03                            | 0.5695                                    | 0.97         | 1.0770   | 0.94     |
| basketball player         | 0.4153                        | 0.89                            | -0.4660                                   | 0.11         | 0.8813   | 0.78     |
| golf player               | 0.2446                        | 0.85                            | -0.2745                                   | 0.15         | 0.5191   | 0.70     |
| homemaker                 | -0.6056                       | 0.02                            | 0.6794                                    | 0.98         | 1.2850   | 0.96     |

Table 1: Evaluation of gender biases in CLIP. As in Section 3, ‘Norm. Sim.’ and ‘Rt. Per.’ stands for the normalization similarity and the percentage in the retrieval results. ‘Sim.’ and ‘Rt.’ stands for the similarity and retrieval-based bias measurements respectively.

**Gender Bias.** As shown in Table 1, we find that there exists concerning gender bias issues for the CLIP model. Specifically, the similarity between female faces and concept 'doctor' is significantly larger than that of male faces ($$p<0.01$$ using permutation tests), while for the concept 'nurse' the situation is reversed. The same phenomenon can also be discovered by the retrieval-based method. We mostly retrieve male faces when searching for 'doctor' while mostly retrieve female faces when search for 'nurse'. We also find that there are biases towards male faces when searching for sports-related terms. 

| Term | White || Black || Bias Measurements |
|---------------------------|--------------------------------|--------------------------------|-------------------------------------------|
|                           | Norm. Sim.                 | Rt. Per.                  | Norm. Sim.                            | Rt. Per. | Sim. | Rt. |
| doctor                    | 0.0831                         | 0.19                           | -0.3011                                   | 0.01         | 0.8148   | 0.5086   |
| nurse                     | -0.0161                        | 0.22                           | -0.0530                                   | 0.09         | 0.6840   | 0.3371   |
| basketball player         | -0.1563                        | 0.10                           | 0.2340                                    | 0.32         | 1.4431   | 0.5029   |
| golf player               | 0.2474                         | 0.28                           | -0.4669                                   | 0.03         | 1.8378   | 0.6629   |
| homemaker                 | -0.2315                        | 0.04                           | -0.0850                                   | 0.12         | 1.5626   | 0.7086   |

Table 2: Evaluation of racial biases in CLIP. While there are seven races in the dataset, here we only list the results for black and white faces. As in Section 3, ‘Norm. Sim.’ and ‘Rt. Per.’ stands for the normalization similarity and the percentage in the retrieval results. ‘Sim.’ and ‘Rt.’ stands for the similarity and retrieval-based bias measurements respectively.

**Racial Bias.** As in Table 2, we find that black faces are less associated with 'doctor’. Also, black faces have higher associations with 'basketball player' than others while lower associations with 'golf player', indicating there are racial biases existed in the CLIP model.




## Reference
[1] AlecRadford,JongWookKim,ChrisHallacy,AdityaRamesh,GabrielGoh,SandhiniAgarwal,Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. <br>

[2] Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In Advances in Neural Information Processing Systems (NeurIPS), 2016. <br>

[3] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like shopping: Reducing gender bias amplification using corpus-level constraints. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2017. <br>

[4] Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In ACM Conference on Fairness, Accountability, and Transparency (FAccT), 2021. <br>

[5] SandhiniAgarwal,GretchenKrueger,JackClark,AlecRadford,JongWookKim,andMilesBrundage. Evaluating clip: towards characterization of broader capabilities and downstream implications. arXiv preprint, 2021. <br>

[6] Jaemin Cho, Abhay Zala, and Mohit Bansal. Dall-eval: Probing the reasoning skills and social biases of text-to-image generative transformers. arXiv preprint, 2022. <br>

[7] Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. Semantics derived automatically from language corpora contain human-like biases. Science, 2017. <br>

[8] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, Zicheng Liu, and Michael Zeng. An empirical study of training end-to-end vision-and-language transformers. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. <br>

[9]GabrielGoh,NickCammarata,ChelseaVoss,ShanCarter,MichaelPetrov,LudwigSchubert,AlecRadford, and Chris Olah. Multimodal neurons in artificial neural networks. Distill, 2021. <br>

[10] Kimmo Karkkainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2021. <br>

[11] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Ryan Cotterell, Vicente Ordonez, and Kai-Wei Chang. Gender bias in contextualized word embeddings. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 629–634, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. <br>

[12] Thomas Manzini, Lim Yao Chong, Alan W Black, and Yulia Tsvetkov. Black is to criminal as caucasian is to police: Detecting and removing multiclass bias in word embeddings. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 615–621, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. <br>

[13] Kawin Ethayarajh, David Duvenaud, and Graeme Hirst. Understanding undesirable word embedding associations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1696–1705, Florence, Italy, July 2019. Association for Computational Linguistics. <br>

[14] Keita Kurita, Nidhi Vyas, Ayush Pareek, Alan W Black, and Yulia Tsvetkov. Measuring bias in contextual- ized word representations. In Proceedings of the First Workshop on Gender Bias in Natural Language Processing, pages 166–172, Florence, Italy, August 2019. Association for Computational Linguistics. <br>

[15] PaulPuLiang,IreneMengzeLi,EmilyZheng,YaoChongLim,RuslanSalakhutdinov,andLouis-Philippe Morency. Towards debiasing sentence representations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5502–5515, Online, July 2020. Association for Computational Linguistics. <br>

[16] ChandlerMay,AlexWang,ShikhaBordia,SamuelR.Bowman,andRachelRudinger.Onmeasuringsocial biases in sentence encoders. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 622–628, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. <br>

[17] Desislava Aleksandrova, François Lareau, and Pierre André Ménard. Multilingual sentence-level bias detection in Wikipedia. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019), pages 42–51, Varna, Bulgaria, September 2019. INCOMA Ltd. <br>

[18] Thomas Davidson, Debasmita Bhattacharya, and Ingmar Weber. Racial bias in hate speech and abusive language detection datasets. In Proceedings of the Third Workshop on Abusive Language Online, pages 25–35, Florence, Italy, August 2019. Association for Computational Linguistics. <br>

[19] Xiaolei Huang, Linzi Xing, Franck Dernoncourt, and Michael J. Paul. Multilingual Twitter corpus and baselines for evaluating demographic bias in hate speech recognition. In Proceedings of the 12th Lan- guage Resources and Evaluation Conference, pages 1440–1448, Marseille, France, May 2020. European Language Resources Association. <br>

[20] AndrewGaut,TonySun,ShirlynTang,YuxinHuang,JingQian,MaiElSherief,JieyuZhao,DibaMirza, Elizabeth Belding, Kai-Wei Chang, and William Yang Wang. Towards understanding gender bias in relation extraction. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2943–2953, Online, July 2020. Association for Computational Linguistics. <br>

[21] Svetlana Kiritchenko and Saif Mohammad. Examining gender and race bias in two hundred sentiment analysis systems. In Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pages 43–53, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. <br>

[22] Tenghao Huang, Faeze Brahman, Vered Shwartz, and Snigdha Chaturvedi. Uncovering implicit gender bias in narratives through commonsense inference. arXiv preprint arXiv:2109.06437, 2021. <br>

[23] Shengyu Jia, Tao Meng, Jieyu Zhao, and Kai-Wei Chang. Mitigating gender bias amplification in distribution by posterior regularization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2936–2942, Online, July 2020. Association for Computational Linguistics. <br>

[24] Christine Basta, Marta R. Costa-jussà, and José A. R. Fonollosa. Towards mitigating gender bias in a decoder-based neural machine translation model by adding contextual information. In Proceedings of the The Fourth Widening Natural Language Processing Workshop, pages 99–102, Seattle, USA, July 2020. Association for Computational Linguistics. <br>

[25] Hila Gonen and Kellie Webster. Automatically identifying gender issues in machine translation using perturbations. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1991– 1995, Online, November 2020. Association for Computational Linguistics. <br>

[26] Haochen Liu, Wentao Wang, Yiqi Wang, Hui Liu, Zitao Liu, and Jiliang Tang. Mitigating gender bias for neural dialogue generation with adversarial learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 893–903, Online, November 2020. Association for Computational Linguistics. <br>

[27] Emily Dinan, Angela Fan, Adina Williams, Jack Urbanek, Douwe Kiela, and Jason Weston. Queens are powerful too: Mitigating gender bias in dialogue generation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8173–8188, Online, November 2020. Association for Computational Linguistics. <br>

[28] Emily Sheng, Kai-Wei Chang, Prem Natarajan, and Nanyun Peng. Towards Controllable Biases in Language Generation. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3239–3254, Online, November 2020. Association for Computational Linguistics. <br>

[29] Catherine Yeo and Alyssa Chen. Defining and evaluating fair natural language generation. In Proceedings of the The Fourth Widening Natural Language Processing Workshop, pages 107–109, Seattle, USA, July 2020. Association for Computational Linguistics. <br>

[30] Katherine Hermann, Ting Chen, and Simon Kornblith. The origins and prevalence of texture bias in convolutional neural networks. Advances in Neural Information Processing Systems, 33:19000–19015, 2020. <br>

[31] Zeyu Wang, Klint Qinami, Ioannis Christos Karakozis, Kyle Genova, Prem Nair, Kenji Hata, and Olga Russakovsky. Towards fairness in visual recognition: Effective strategies for bias mitigation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8919–8928, 2020. <br>

[32] Yunliang Chen and Jungseock Joo. Understanding and mitigating annotation bias in facial expression recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14980–14991, 2021. <br>

[33] Katja Schwarz, Yiyi Liao, and Andreas Geiger. On the frequency bias of generative models. Advances in Neural Information Processing Systems, 34, 2021. <br>

[34] Shruti Bhargava and David Forsyth. Exposing and correcting the gender bias in image captioning datasets and models. arXiv preprint, 2019. <br>

[35] Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny, pornog- raphy, and malignant stereotypes. arXiv preprint, 2021. <br>

[36] Ruixiang Tang, Mengnan Du, Yuening Li, Zirui Liu, Na Zou, and Xia Hu. Mitigating gender bias in captioning systems. In Proceedings of the Web Conference, 2021. <br>

[37] Tejas Srinivasan and Yonatan Bisk. Worst of both worlds: Biases compound in pre-trained vision-and- language models. arXiv preprint arXiv:2104.08666, 2021. <br>

[38] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pre-training of generic visual-linguistic representations. In International Conference on Learning Representations (ICLR), 2019. <br>

[39] Yubo Zhang, Hao Tan, and Mohit Bansal. Diagnosing the environment bias in vision-and-language navigation. arXiv preprint arXiv:2005.03086, 2020. <br>

[40] Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. Feature visualization. Distill, 2017. <br>

[41] ChristianSzegedy,WojciechZaremba,IlyaSutskever,JoanBruna,DumitruErhan,IanGoodfellow,and
Rob Fergus. Intriguing properties of neural networks. arXiv preprint, 2013. <br>

[42]DamaiDai,LiDong,YaruHao,ZhifangSui,andFuruWei.Knowledgeneuronsinpretrainedtransformers.
In Annual Meeting of the Association for Computational Linguistics (ACL), 2022. <br>

---
