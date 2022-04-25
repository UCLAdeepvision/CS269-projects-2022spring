---
layout: post
comments: true
title: Visual Counting
author: Srinath
date: 2022-04-19
---


> Repetition of a task is quite common in our day to day life ranging from a simple pendulum to the periodic day and night pattern of earth, everything repeats. Counting repetitions through time from a video has interesting applications into healthcare, sports and fitness domains for tracking reps in exercises, shots in a badminton rally etc. Through this project, we would like to explore existing literature towards class agnostic counting from video and specifically apply/invent to a scenario of counting paper bills(currency) from a video. Although machines exist for this particular task, being able to do the same just by using a smartphone camera has its advantages of being widely accessible and availability at low cost. We believe that this task is also a perfect fit for this course as it needs both human and AI collaboration!. We also like to mention that this seemingly simple task might be ambitious to achieve and thereby planning to explore other directions to mould existing counting mechanisms which process video as a whole towards real time counting systems, so that they can be used in day to day life.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Our introduction
various Repetition natures and difficulties
Difficulties w.r.t our targeted problem.
Why we think this task is possible.

## Related Work

In our quest for a system which could possibly count currency flippings from a video, we have come across an interesting paper [Counting Out Time: Class Agnostic Video Repetition Counting in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf) [1] which uses a specialised neural net architecture called 'RepNet' to count the repeating task in a video. Below we will go through a brief study of this paper, describing its architecture, training, results and its performance on our expected task of counting.

### RepNet: Class Agnostic Video Repetition Counting in the Wild

#### Architecture

The following is the architecture of 'RepNet'.  
TO-DO : Add Image   

The first part of the architecture is to embed the input frames into a 512 dimentional feature vector and create a Temporal Self Similarity Matrix(**TSM**). A TSM can be thought of as an indicator to measure similarity of the video frames within itself, essentially the $$(i, j)'th$$ element of TSM $$(TSM[i][j])$$, represents the similarity score between frame $$i$$ and frame $$j$$ in the input video. This particular idea of using a TSM differentiates this work from others and accounts for its class agnostic nature, we do not need to care about the task which is going on, all we need is its similarity through time!!

The next part is to pass the created TSM as an input to a Transformer based deep neural network(referred as '**Period Predictor**') to predict the period length and periodicity for each frame. Period length is a discrete quantity $$\in \{2,3,...,\frac{N}{2}\}$$ where $$N$$ represents the number of input frames, where as Periodicity is a binary quantity $$\in \{0,1\}$$ indicating whether a frame is part of the repeating task or not. Using per frame period and periodicity in a video, it is possible to infer the count and length of a repeating task in a video as described in the inference section.

#### Data && Training

In order to train this network, they have created a synthetic dataset called **Countix**, consisting of real life repeating actions covering diverse set of periods and counts. The dataset is synthetic in a sense that certain actions from Kinetics[2] dataset are selected and their annotated segments are used to create repetition videos by joining same segment multiple times by varying in frequency, reversing the action video etc. The created videos are approximately around 10 seconds. The model is trained for 400K steps with learning rate of $$6e^{-6}$$, using ADAM optimizer and bacth size of 5 videos each with 64 frames.

#### Inference
#### Evaluation Metrics

TO-DO : Write about MAE and OBO error metrics.

#### Results

The paper presents promising results on some actions like *jumping jacks*, *squats*, *slicing onion* etc. The counts are also near perfect in cases of varying periodicity and change of tempo in the repetitions. It also mentions lower error metrics, **MAE** and **OBO** of **0.104** and **0.17** respectively while comparing with other methods. Some of the tasks on which the count is accurate are shown below.
TO-DO : Add Image

#### Performance on our task

We have tested the model on our task of counting currency. As currency bundles are not easily available :(, we just have 1-2 videos of currency flipping for now. Instead of currency, we are targeting the count of paper flippings. Below we show the results of running RepNet on few videos related to our case. **AC** stands for Actual Count, **PC** for Predicted Count. 

<div style="width: 100%; display: table;">
    <div style="display: table-row; height: 100px;">
        <div style="display: table-cell; margin: 5;">
            <iframe width="200" height="300" src="https://www.youtube.com/embed/hklIYi-9ZPY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  
            <p style="margin: 0;">currency - normal speed</p>    
			<p style="margin: 0;">30 fps | AC > 41 | PC = 28</p>
        </div>
        <div style="display: table-cell; margin: 5;"> 
            <iframe width="200" height="300" src="https://www.youtube.com/embed/cptyDu-wREM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            <p style="margin: 0;">book pages - slow</p>
			<p style="margin: 0;">30 fps | AC > 0 | PC = 0</p>
        </div>
        <div style="display: table-cell; margin: 5;"> 
            <iframe width="200" height="300" src="https://www.youtube.com/embed/lHKVVW-HnmU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            <p style="margin: 0;">book pages - very slow</p>
			<p style="margin: 0;">30 fps | AC = 20 | PC = 18</p>
        </div>
    </div>
</div>

## Tentative Approach

As we could clearly observe that RepNet doesn't work to the expectation for our task, the first step we would like to pursue is to create more videos(around 10) of various scenarious of page flippings by varying capture fps, speed of flipping etc. and we will test them using RepNet. These videos will also help us eventually to evaluate our system.

## Expected Results

Our expected results

## References

Our references

---
