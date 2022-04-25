---
layout: post
comments: true
title: Improving and Extending Web UI generation from Hand Drawn Sketches
author: Tanmay Sanjay Hukkeri, Lalit Bhagat
date: 2022-04-24
---


> <p style="text-align:justify;"> The area of Web-UI design continues to evolve , often reqiuring a balance of effort between designers and developers to come up with a suitable user interface. One of the key challenges in standard Web UI development involves reaching an interface between desginers and developers, in being able to convert designs to code. To this end, several works in recent years have taken atempts to try and automate this task, or provide for easy conversion from design to code. Works such as Microsoft sketch2code and pix2code , provide automation by converting sketches and screenshots respectively to code. However, there still remains room for further work in this domain, such as including more element types, encoding more information in the sketch and allowing for more variablity.  The proposed project seeks to improve upon the Sketch-to-code framework, by first constructing an enriched dataset of Web-UI samples  and then allowing for embedding more information in the sketch such as color , font-style etc as well as allowing custom generated images using GANs.</p>

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Research Problem and Goal
### Research Problem
<p style="text-align:justify;">
Smooth Web-UI development continues to be an active area of research. Applications such as Microsoft's sketch2Code [1] and pix2code [2] demonstrate great potential as applications that can be used to generate UI code using sketches and images respectively. These models work well on their "element set", which by nature of the problem at hand, is limited to the tags present in the created dataset, as well as restricted to the limited layout forms in the training dataset. The key issue at hand stems from datasets on Web-UI being quite limited as opposed to Android/IOS UI , which has more datasets such as the Rico dataset [3]. Additionally, these models also allow for very-limited to no styling information in the sketches.
</p>  

### Goal
<p style="text-align:justify;">
The goal of this project is to improve and extend upon the existing sketch2Code applications, across several potential lines of work.
</p> 

Some of the key goals of the project include:
1. Generation of a larger and enhanced dataset for Web-UI images, including generating .gui files that allow more layouts. 
2. Allowing users to specify basic styling information such as text color, font-style, shape etc on sketches
3. Leveraging the power of text-to-image GAN models to generate custom images for sketches.

Other more ambitious goals include:
1. Adding improvements to the current models performing object recognition and OCR for converting the sketch to code.  
2. Scraping real websites to generate more realisitc dataset samples.
3. Adding interactivity specification (such as JavaScript embedding) potential in sketches.  

## State of the Art Work
A study of some of the state-of-the-art in the domain can be described as below.  

### From sketch to code  
Some of the state of the art in converting sketches to code include commerical applications such as Microsoft sketch2code [1] as well as custom implemented projects such as [4], [5] However, the projects in [4], [5] suffer from poor generalization to a specific layout because of the train dataset having only a specific layout of examples.

### From screenshot to code 
Projects such as pix2code[2] and well as the work done in [6] translate from UI screenshots to code. 

### Eve: A Sketch-based Software Prototyping Workbench  
As part of the literatre survey, we also study the tool Eve [7], which we found to be the closest that goes from sketches to high-level code, while allowing for custom styling and flexibility. However, we believe there are the following shortcomings.  

- This works on Google Material Design workspace, for Android apps. Mobile apps have slightly more dataset coverage right now because of sources such as Google Material Design [8] , RICO [3] etc. However, we did not come across any such applications for Website design. 
- Adding custom generation of images etc would be new as compared to this application.
- This application performs color and other detailing on the tool, and requires sketching to be done on the tool. We feel that allowing hand-drawn sketches might be more convenient. 
- The tool does not expose any code/results apart from a single set of screenshots, which limits study of its extensiveness. 

![Eve Interation Model]({{ '/assets/images/team04/eve_image.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}  

<p style="text-align:center;">
<i>Fig 1. Eve Interaction Model</i> [7].
</p>

## Datasets
1. Custom dataset from pix2code  
The UI dataset from pix2code is a sequence of (HTML screenshots, GUI files containing DSL) . The sketch2code approach converts this to an image captioning problem of the form (sketch, DSL) by manipulating the CSS of the webpages in the HTML screenshots.

2. Dataset generation using Web Generator  
This module generates a dataset of webpage samples (html files with different properties). A key step would involve writing a method that parses these html files into DSL structure, and adding language embeddings for various styling properties.  

3. Dataset samples from Microsoft sketch2Code  
The sketch2code Github Reporsitory also exposes a training-set of examples that could be promising in improving the current training set.

4. (Ambitious) Scraping from online sites using models such as in [6].  

## Proposed Methodology

### Key Steps 
1. Dataset generation  
The first major step involves dataset creation, generating additional samples and dataset extension. A key task involves converting the Web Generator samples (screenshot, html file) created from Web Generator to the required dataset format. The proposed overall methodology can be detailed as below:  
- Convert the screenshots to hand-drawn sketches based of the method specified in [5]. There is also scope in using projects such as those in [6] to generate the code from the screenshot, and then manipulating it as per [5] to get the required hand-drawn sketch.  
- Convert the HTML code to DSL by writing a custom parser for the same.  
- Extend the above dataset to include information for basic styling such as color etc.  
- Extend the above dataset to include information for generating images (GAN input)  

2. Model re-train and enhancement  
The second major step includes re-training and enhancing the model on this newly created dataset. Specifically, the steps can be detailed as follows:  
- Re-train the model on the new dataset and test against custom samples for each new use-case.
- Incorporate a text-to-image GAN model (such as those in [10],[11],[12]).
- Finetune and update the model as required.  

### Timeline 
The proposed timeline in the upcoming weeks is as below.  
- Week 5-6 : Work on Dataset Generation tasks, including covnerting Web Generator dataset to required format and adding styling based examples.  
- Week 7-8 : Incorporate text-to-image GAN and create an end-to-end application that meets the above goals.  
- Week 9-10: Collect necessary resources and work on the report.  


## Preliminary Results
**Running existing sketch-to-code models**  

Running the model provided in [5] and the sample input in [5], we are able to obtain the output for the sketch as shown below.  

![Sketch1]({{ '/assets/images/team04/sketch_to_code_samples.PNG' | relative_url }})
{: style="width: 60%; align:right;"}  

While the model outputs well for this sketch, one of the key issues observed with this implementation is the lack of generalization to other layouts, since the training dataset only had samples of this form of layout. This raises the importance and significance of the initial step to create a more robust dataset for training purposes.

**Sample Generation using Web Generator**  
By following the steps as per the Github link to the WebGenerator project, we were able to generator random UI samples such as the below two.  

![Sample1]({{ '/assets/images/team04/web_generator_samples.PNG' | relative_url }})
{: style="width: 60%; align:right;"}  

## References
[1] Microsoft Corporation, [Sketch2Code : https://www.microsoft.com/en-us/ai/ai-lab-sketch2code](https://www.microsoft.com/en-us/ai/ai-lab-sketch2code)  
[2] Tony Beltramelli. 2018. Pix2code: Generating Code from a Graphical User Interface Screenshot. In <i>Proceedings of the ACM SIGCHI Symposium on Engineering Interactive Computing Systems</i> (<i>EICS '18</i>). Association for Computing Machinery, New York, NY, USA, Article 3, 1–6. [DOI:https://doi.org/10.1145/3220134.3220135](https://doi.org/10.1145/3220134.3220135)  
[3] Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols, and Ranjitha Kumar. 2017. Rico: A Mobile App Dataset for Building Data-Driven Design Applications. In <i>Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology</i> (<i>UIST '17</i>). Association for Computing Machinery, New York, NY, USA, 845–854. [DOI:https://doi.org/10.1145/3126594.3126651](https://doi.org/10.1145/3126594.3126651)  
[4] sketch2code, Anchen, [https://github.com/mzbac/sketch2code](https://github.com/mzbac/sketch2code)  
[5] SketchCode, Ashwin Kumar, [https://github.com/ashnkumar/sketch-code](https://github.com/ashnkumar/sketch-code)  
[6] Emil Wallner, [https://blog.floydhub.com/turning-design-mockups-into-code-with-deep-learning/?source=techstories.org](https://blog.floydhub.com/turning-design-mockups-into-code-with-deep-learning/?source=techstories.org)  
[7] Sarah Suleri, Vinoth Pandian Sermuga Pandian, Svetlana Shishkovets, and Matthias Jarke. 2019. Eve: A Sketch-based Software Prototyping Workbench. In <i>Extended Abstracts of the 2019 CHI Conference on Human Factors in Computing Systems</i> (<i>CHI EA '19</i>). Association for Computing Machinery, New York, NY, USA, Paper LBW1410, 1–6. [DOI:https://doi.org/10.1145/3290607.3312994](https://doi.org/10.1145/3290607.3312994)  
[8] Google Material Design, [https://material.io/](https://material.io/)  
[9] Andrés Soto, Héctor Mora, Jaime A. Riascos,Web Generator: An open-source software for synthetic web-based user interface dataset generation,SoftwareX,Volume 17,2022,100985,ISSN 2352-7110,[https://doi.org/10.1016/j.softx.2022.100985](https://doi.org/10.1016/j.softx.2022.100985).  
[10] Wang, Peng & Yang, An & Men, Rui & Lin, Junyang & Bai, Shuai & Li, Zhikang & Ma, Jianxin & Zhou, Chang & Zhou, Jingren & Yang, Hongxia. (2022). Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework [https://doi.org/10.48550/arXiv.2202.03052](https://doi.org/10.48550/arXiv.2202.03052)  
[11] Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M. &amp; Sutskever, I.. (2021). Zero-Shot Text-to-Image Generation. <i>Proceedings of the 38th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 139:8821-8831 Available from [https://proceedings.mlr.press/v139/ramesh21a.html](https://proceedings.mlr.press/v139/ramesh21a.html).  
[12] Yutong ZHOU, Awesome-Text-to-Image, [https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image)



---
