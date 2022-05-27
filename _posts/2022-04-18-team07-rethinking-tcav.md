---
layout: post
comments: true
title: Rethinking TCAV
author: Andrew Bai, Yuxin Wang
date: 2022-04-18
---


> (Placeholder for header)

<!--more-->
{: class="table-of-content"}

* TOC
{:toc}
# Rethinking concept-based interpretation with gradients

### Motivation

Interpretation is important for ML systems IRL and concept-based intepretations are easier for humans to understand, achieving the goal of interpretation. TCAV [1] is a SOTA method for concept-based interpretation and is widely adopted for its simplicity and effectiveness. TCAV calculates the alignment of gradients and approximates the concept with a linear function. There are two problems we are attempting to answer. First, TCAV relies on the linear separability assumption which may not hold in general. What happens in that situation does the method still work? Second, there is a line of work of feature attribution based on calculating input gradients. TCAV also uses gradients in its calculation. How does TCAV relate to other gradient-based attribution methods? Our goal is to better understand how TCAV mathematically works and improve on its flaws.

### Expected results

We aim to show (1) how TCAV behaves when concepts are not linearly separable in any model layer and (2) whether interpretations with TCAV is consistent when concepts are linearly separable in many layers. We also aim to mathematically show the connections between TCAV score and the gradient of the model output with respect to concepts. 

### Project timeline

1. Week 4-5: experiment with TCAV on artificially constructed models to examine its interpretations when concepts are not linearly separable.
2. Week 6-7: analyze TCAV mathematically to obtain theoretical insights of how the method works and its limits induced by its implicit assumptions.
3. Week 8-9: propose improved version of TCAV based on the theoretical insights (week 6-7) that solved the empirically shown problems (week 4-5).
4. Week 10: wrap up, present, and compose report.

### Datasets

* [Benchmarking Attribution Methods (BAM)](https://github.com/google-research-datasets/bam): an artificial dataset where the objects from MSCOCO are pasted onto MiniPlaces background scenes. Classifiers trained on the dataset with different ground truth labels are expected to attribute concepts accordingly. For instance, an object classifier should only attribute importance to object concepts and none to scene concepts.
* [Broad and Densely Labeled Dataset (Broden)](https://continental.github.io/hybrid_learning/docs/apiref/generated/hybrid_learning.datasets.custom.broden.html): combination of multiple image datasets with fine-grained semantic labels that can serve as ground truth for concepts if we consider semantic labels as concepts.

## Reference

[1] Kim, Been, et al. "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav)." *International conference on machine learning*. PMLR, 2018.

---
