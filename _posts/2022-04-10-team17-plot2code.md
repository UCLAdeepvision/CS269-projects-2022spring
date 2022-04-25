---
layout: post
comments: true
title: Visualization plot to code
author: Ankur Kumar, Pranay Shirodkar
date: 2022-04-10
---


> In this work, we will explore sketch to visualization (line plot, bar graph, pie chart etc) task via image generation as well as code generation. This work can be useful to reduce the complexity involved in working with visualization tools such as Matplotlib. Also, this project will allow us to understand the challenges in image generation and program synthesis.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Given an input sketch containing visualization, the task is to create a refined version. This can be achieved by either generating corresponding code, followed by rendering, or creating the final image directly. We use the Matplotlib python package only for code generation due to resource limitations. This problem is a step towards developing alternative ways to interact with visualization libraries, which have grown over time to become popular and significantly complex. Our work will lower the barrier to entry for newcomers by allowing them to generate sophisticated visualizations. It will also prove to be useful for others possessing some knowledge of the library, which generally contains thousands of APIs.

## Related work
There has been a similar attempt for Pandas library, where authors generate program given input and output specifications pertaining to dataframe manipulations [9]. Earlier works on visualization have explored applications such as classification of input visualizaiton into line, bar etc. [6], extracting information from bar chart images [5], and generation of visualization images given input data [2]. [1] tackles a similar problem, where local code context along with natural language instructions are used to generate code for visualization. Code generation from webpage layout [10], code recommendations for Android GUI [3] etc, have also been explored. Our work is directly related to [4], which outputs (*intermediate*) code for input sketch. However, we are interested in directly generating Matplotlib code as opposed to manually defined intermediate language. We also want to explore the effectiveness of conditional image generation methods for this task.

## Dataset
There is no public dataset with human-drawn sketch and corresponding visualization plots. [4] create a synthetic dataset for their task of sketch to *intermediate* code generation. The dataset is not available but the code to create the dataset has been released [11]. It randomly generates data such as `plot type, x, y, legend, color` etc. required to create XKCD plot, which looks cartoonish, using Matplotlib APIs. Then they apply style transfer to generate images which look like hand drawn sketches. We adapt their code to create aligned dataset (10,000 samples) for our task as follows. The resulting dataset is divided into train, dev and test split in the ratio 70:10:20.

### Sketch to image dataset
We plan to experiment with 2 types of sketches, XKCD and style-transferred XKCD, for input. XKCD image is very similar to normal visualization images, whereas many details are missing in style-transferred images, which appears to be rough sketch. A sample image for each of these types are shown below. We have prepared aligned dataset for the first type (i.e. XKCD sketch and corresponding visualization)  and also done preliminary experiments as discussed in next section. Next, we will apply style-transfer to the generated XKCD images to create a more realistic sketch dataset.

![Model Input]({{ '/assets/images/team17/examples.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1: Example XKCD sketch (left), target distribution (middle) and style-transferred image (right).*

### Sketch to code dataset
We are currently modifying the original code to generate matplotlib code for the above images. It requires us to understand the intermediate language specifications and map it back to Matplotlib code. We hope to complete this along with the style-transfer dataset by next week.

#### Sample Input
![Model Input]({{ '/assets/images/team17/sketch_input.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 2: An example sketch as input to the model.*

#### Sample Output
```
import matplotlib.pyplot as plt

## user should define x, y1, y2 here
x = 
y1 = 
y2 = 

# plot the function
plt.plot(x,y1, 'r', label = "y = x^2",
linestyle = 'dashed')
plt.plot(x,y2, 'g', label = "y = x^2 + 2x - 15",
linestyle = 'dashed')
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.xlabel("x")
plt.ylabel("y", rotation=0)
plt.legend(loc="lower right")
plt.title("Two quadratic graphs")

# show the plot
plt.show()
```

## Our Approach

### Sketch to image
It is natural to formulate the task of obtaining refined visualization as sketch to image generation. We use existing pix2pix code repository [12] as it serves as a strong baseline model for supervised conditional image generation. The training loss has two components: 1. GAN loss 2. L1-reconstruction loss. Model performance can be evaluated using automated metrics like FID score as well as human evaluation. 

We show the initial results for model a trained on XKCD sketch for 20 epochs and default configurations from pix2pix repository.
![Model Input]({{ '/assets/images/team17/xkcd_loss.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 3: Training loss for initial conditional-GAN model.*

![Model Input]({{ '/assets/images/team17/xkcd_output.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 4: Sample result for conditional-GAN model trained for 20 epochs.*

### Sketch to code
This is a sequence modeling problem similar to image captioning task, where we train an encoder-decoder model. We hope to use open-source decoder for code generation. The model is trained using cross-entropy loss and evaluated using metrics such as BLEU.


## Next Steps
1. Finish data preparation for style-transferred case and generate corresponding Matplotlib code
2. Parameter tuning for XKCD sketch to image training
3. Train conditional GAN for style-transferred images as input
4. Train encoder-decoder model for sketch-to-code task
5. Evaluations

<!-- 
## Main Content
Your article starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[11] Lachaux, Marie-Anne et al. “Unsupervised Translation of Programming Languages.” ArXiv abs/2006.03511 (2020).

[12] Chen, Mark et al. “Evaluating Large Language Models Trained on Code.” ArXiv abs/2107.03374 (2021).

[13] Nijkamp, Erik et al. “A Conversational Paradigm for Program Synthesis.” ArXiv abs/2203.13474 (2022).
-->

## References
[1] Chen, Xinyun et al. “PlotCoder: Hierarchical Decoding for Synthesizing Visualization Code in Programmatic Context.” ACL (2021).

[2] Dibia, Victor C. and Çagatay Demiralp. “Data2Vis: Automatic Generation of Data Visualizations Using Sequence-to-Sequence Recurrent Neural Networks.” IEEE Computer Graphics and Applications 39 (2019): 33-46.

[3] Zhao, Yanjie et al. “Icon2Code: Recommending code implementations for Android GUI components.” Inf. Softw. Technol. 138 (2021): 106619.

[4] Teng, Zhongwei et al. “Sketch2Vis: Generating Data Visualizations from Hand-drawn Sketches with Deep Learning.” 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA) (2021): 853-858.

[5] Zhou, Fangfang et al. “Reverse-engineering bar charts using neural networks.” Journal of Visualization 24 (2021): 419-435.

[6] Deng, Dazhen et al. “VisImages: A Fine-Grained Expert-Annotated Visualization Dataset.” IEEE transactions on visualization and computer graphics PP (2022).

[7] Al-Hossami, Erfan and Samira Shaikh. “A Survey on Artificial Intelligence for Source Code: A Dialogue Systems Perspective.” ArXiv abs/2202.04847 (2022).

[8] Borkin, Michelle et al. “What Makes a Visualization Memorable?” IEEE Transactions on Visualization and Computer Graphics 19 (2013): 2306-2315.

[9] Bavishi, Rohan et al. “AutoPandas: neural-backed generators for program synthesis.” Proceedings of the ACM on Programming Languages 3 (2019): 1 - 27.

[10] Beltramelli, Tony. “pix2code: Generating Code from a Graphical User Interface Screenshot.” Proceedings of the ACM SIGCHI Symposium on Engineering Interactive Computing Systems (2018).

[11] https://github.com/magnumresearchgroup/Sketch2Vis

[12] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

---
