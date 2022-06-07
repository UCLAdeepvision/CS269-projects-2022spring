---
layout: post
comments: true
title: Visual Counting
author: Srinath Naik Ajmeera, Sonia Jaiswal
date: 2022-04-24
---

> Repetition of a task is quite common in our day to day life ranging from a simple pendulum to the periodic day and night pattern of earth, everything repeats. Counting repetitions through time from a video has interesting applications into healthcare, sports and fitness domains for tracking reps in exercises, shots in a badminton rally etc. Through this project, we would like to explore existing literature towards class agnostic counting from video and specifically apply/invent to a scenario of counting paper bills(currency) from a video. Although machines exist for this particular task, being able to do the same just by using a smartphone camera has its advantages of being widely accessible and availability at low cost. We believe that this task is also a perfect fit for this course as it needs both human and AI collaboration!. We also like to mention that this seemingly simple task might be ambitious to achieve and thereby planning to explore other directions to mould existing counting mechanisms which process video as a whole towards real time counting systems, so that they can be used in day to day life.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

We find that the ability to count repetitive tasks from a video is interesting enough. It has associated challenges that the repetition may not be exact, non-uniform period length of the repeating task, being class agnostic for varied actions, handling high to low frequency task variations and feel it is worth exploring and applying to scenario of counting currency. We think that this task of counting currency is achievable because with introduction of sufficient slow motion in the video, human beings are able to count the number of paper bills exactly (We tried this and are able to do it). As humans(processing power of around 24 fps) are able to do this and latest samrtphone camera captures vary around 30-60 fps, we hope its not impossible to achieve this task.   

## Related Work

In our quest for a system which could possibly count currency flippings from a video, we have come across an interesting paper [Counting Out Time: Class Agnostic Video Repetition Counting in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf) [1] which uses a specialised neural net architecture called 'RepNet' to count the repeating task in a video. Below we will go through a brief study of this paper, describing its architecture, training, results and its performance on our expected task of counting.

### RepNet: Class Agnostic Video Repetition Counting in the Wild

#### Architecture

The following is the architecture of 'RepNet'. 
  ![RepNet Architecture]({{ '/assets/images/team12/Archi.png' | relative_url }})
 

The first part of the architecture is to embed the input frames into a 512 dimentional feature vector and create a Temporal Self Similarity Matrix(**TSM**). A TSM can be thought of as an indicator to measure similarity of the video frames within itself, essentially the $$(i, j)'th$$ element of TSM $$(TSM[i][j])$$, represents the similarity score between frame $$i$$ and frame $$j$$ in the input video. This particular idea of using a TSM differentiates this work from others and accounts for its class agnostic nature, we do not need to care about the task which is going on, all we need is its similarity through time!!

The next part is to pass the created TSM as an input to a Transformer based deep neural network(referred as '**Period Predictor**') to predict the period length and periodicity for each frame. Period length is a discrete quantity $$\in \{2,3,...,\frac{N}{2}\}$$ where $$N$$ represents the number of input frames, where as Periodicity is a binary quantity $$\in \{0,1\}$$ indicating whether a frame is part of the repeating task or not.

#### Data && Training

In order to train this network, they have created a synthetic dataset called **Countix**, consisting of real life repeating actions covering diverse set of periods and counts. The dataset is synthetic in a sense that certain actions from Kinetics[2] dataset are selected and their annotated segments are used to create repetition videos by joining same segment multiple times by varying in frequency, reversing the action video etc. The created videos are approximately around 10 seconds. The model is trained for 400K steps with learning rate of $$6e^{-6}$$, using ADAM optimizer and bacth size of 5 videos each with 64 frames.

#### Evaluation Metrics

The existing literature uses two main metrics for evaluating repetition counting in videos:\
**Off-By-One (OBO) count error:** If the predicted count is within one count of the ground truth value, then the video is considered to be classified correctly, otherwise it is a mis-classification. The OBO error is the mis-classification rate over the entire dataset.

**Mean Absolute Error (MAE) of count:** This metric measures the absolute difference between the ground truth count and the predicted count, and then normalizes it by dividing with the ground truth count. The reported MAE error is the mean of the normalized absolute differences over the entire dataset.

#### Results

The paper presents promising results on some actions like *jumping jacks*, *squats*, *slicing onion* etc. The counts are also near perfect in cases of varying periodicity and change of tempo in the repetitions. It also mentions lower error metrics, **MAE** and **OBO** of **0.104** and **0.17** respectively while comparing with other methods. Some of the tasks on which the count is accurate are shown below.
  ![Results]({{ '/assets/images/team12/res.gif' | relative_url }})

#### Performance on our task

We have tested the model on our task of counting currency. As currency bundles are not easily available :(, we just have 1-2 videos of currency flipping for now. Instead of currency, we are targeting the count of paper flippings. Below we show the results of running RepNet on few videos related to our case. **AC** stands for Actual Count, **PC** for Predicted Count. 

<div style="width: 100%; display: table;">
    <div style="display: table-row; height: 100px;">
        <div style="display: table-cell; margin: 5;">
            <iframe width="200" height="300" src="https://www.youtube.com/embed/hklIYi-9ZPY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  
            <p style="margin: 0;">currency - normal speed</p>    
			<p>30 fps | AC > 41 | PC = 28</p>
        </div>
        <div style="display: table-cell; margin: 5;"> 
            <iframe width="200" height="300" src="https://www.youtube.com/embed/cptyDu-wREM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            <p style="margin: 0;">book pages - slow</p>
			<p>30 fps | AC > 0 | PC = 0</p>
        </div>
        <div style="display: table-cell; margin: 5;"> 
            <iframe width="200" height="300" src="https://www.youtube.com/embed/lHKVVW-HnmU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            <p style="margin: 0;">book pages - very slow</p>
			<p>30 fps | AC = 20 | PC = 18</p>
        </div>
    </div>
</div>

## Tentative Approach

As we could clearly observe that RepNet doesn't work to the expectation for our task, the first step we would like to pursue is to create more videos(around 10) of various scenarious of page flippings by varying capture fps, speed of flipping etc. and we will test them using RepNet. These videos will also help us eventually to evaluate our system. There are several videos on which RepNet works well, we would like to speed up them to change the frequency of action and see the performance on different frequencies of the same video (Thanks **Ankur** for the suggestion).

The next step is to analyze the TSM for these tasks and high frequency tasks in general, especially we would like to get some insights into the difference betwen TSM for regular tasks where RepNet works well and for high frequency tasks.

Once we are more confident that the issue is w.r.t high frequency videos, we would like to adapt the RepNet training framework and train using the same dataset but instead speed up the videos to increase the frequency of the associated task and evaluate it for our use case.

Finally, we would like to explore the idea of using/simplifying the model to account for real time counting on a computer or smartphone, which might involve model pruning techniques.

## Note

We are open for ideas, suggestions for this project. Feel free to comment or reach out in case you have some related interesting things to discuss or if you find any papers which might be relevant to solving this particular problem :)   
Contact : ***srinath@g.ucla.edu***, ***soniajaiswal@g.ucla.edu***   

## References

[1] Dwibedi, Debidatta, et al. "Counting out time: Class agnostic video repetition counting in the wild." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.     

[2] Kay, Will, et al. "The kinetics human action video dataset." arXiv preprint arXiv:1705.06950 (2017).

---
