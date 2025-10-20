Moderation is the Best Policy: Dynamic Defense Against Gradient-Based Data Reconstruction Attacks in Federated Learning
=

> Qinyang Miao, Wen Sun, Dan Zhu, Jinku Li, Yajin Zhou, Cristina Alcaraz.  
> *TDSC, 2024*
---
## Abstract
Federated Learning (FL) is a privacy-preserving distributed machine learning framework. However, recent
studies have shown that implementing gradient-based Data Reconstruction Attacks (DRA) can still lead to the leakage of
user privacy through frequently uploaded model parameters in FL. Existing works leverage Differential Privacy (DP) to
prevent privacy leakage, but the lack of effective scheduling of the privacy budget results in significant accuracy loss
in the trained models. In this paper, we propose a novel dynamic privacy preserving federated learning framework, named NDPP-FL, capable of delivering robust defenses against DRA while significantly mitigating performance loss. Our key insight is to regard the privacy budget as a non-replenishable resource and dynamically schedule it based on privacy leakage risks to provide self-adaptive privacy protection for clients across varying communication rounds. Specifically, based on the amount of information between the local dataset and the transmitted parameters, we first design a parameter channel information leakage model. Then, during each update iteration, we introduce saliency perturbations based on the Hessian matrix to enhance defensive capabilities. Meanwhile, to improve the performance of NDPP-FL, sample-adaptive clipping and decaying noise perturbations are adopted in the construction. Furthermore, extensive experiments demonstrate that our framework performs excellently in terms of model accuracy and resilience against DRA.

## Citation

Please cite our paper if you find this code useful for your research.

