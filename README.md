# Why Settle for Mid: A Probabilistic Viewpoint to Spatial Relationship Alignment in Text-to-image Models
This repository contains the codebase for the paper [Why Settle for Mid ](https://arxiv.org/abs/2506.23418).

## Abstract
Despite the ability of text-to-image models to generate high-quality, realistic, and diverse
images, they face challenges in compositional generation, often struggling to accurately
representdetailsspecifiedintheinputprompt. Aprevalentissueincompositionalgeneration
is the misalignment of spatial relationships, as models often fail to faithfully generate images
that reflect the spatial configurations specified between objects in the input prompts. To
address this challenge, we propose a novel probabilistic framework for modeling the relative
spatial positioning of objects in a scene, leveraging the concept of Probability of Superiority
(PoS). Building on this insight, we make two key contributions. First, we introduce a novel
evaluation metric, PoS-based Evaluation (PSE), designed to assess the alignment of 2D
and 3D spatial relationships between text and image, with improved adherence to human
judgment. Second, we propose PoS-based Generation (PSG), an inference-time method that
improves the alignment of 2D and 3D spatial relationships in T2I models without requiring
fine-tuning. PSG employs a Part-of-Speech PoS-based reward function that can be utilized
in two distinct ways: (1) as a gradient-based guidance mechanism applied to the cross-
attention maps during the denoising steps, or (2) as a search-based strategy that evaluates
a set of initial noise vectors to select the best one. Extensive experiments demonstrate that
the PSE metric exhibits stronger alignment with human judgment compared to traditional
center-based metrics, providing a more nuanced and reliable measure of complex spatial
relationship accuracy in text-image alignment. Furthermore, PSG significantly enhances
the ability of text-to-image models to generate images with specified spatial configurations,
outperforming state-of-the-art methods across multiple evaluation metrics and benchmarks.

## Usage
This score gives an improved and continuous evaluation of accuracy of a spatial relationship. You can use 'pse_metric.py' to calculate the accuracy of a relation between two masks.
### Requirements :
- Python 3.x
- torch

Example code as follows:
```python
from pse_metirc import get_score_from_mask
# Some code to extract the masks (you can use GroundedSAM from the PSE directory)
score = get_score_from_mask(mask_A, mask_B, relationship)
```
We especially encourage you to use this score for inference time scaling on generative models, becuase it gives you a less sparse and more human-correlated reward function than using position of center of bounding boxes.
## Cite our work
```text
@misc{rezaei2025settlemidprobabilisticviewpoint,
      title={Why Settle for Mid: A Probabilistic Viewpoint to Spatial Relationship Alignment in Text-to-image Models}, 
      author={Parham Rezaei and Arash Marioriyad and Mahdieh Soleymani Baghshah and Mohammad Hossein Rohban},
      year={2025},
      eprint={2506.23418},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23418}, 
}
```
