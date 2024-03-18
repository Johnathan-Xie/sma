# Self-Guided Masked Autoencoders
Code for the ICLR 2024 paper ["Self-Guided Masked Autoencoders for Domain-Agnostic Self-Supervised Learning"](https://arxiv.org/abs/2402.14789)

## Installation
```
conda create --name sma_env --name python=3.10
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install .
pip install datasets evaluate
pip install scipy scikit-learn timm
```
## Experiments
Experiments are found in the examples directory with accompanying scripts in each directory. To ease reproducing experiments,
we have preprocessed and uploaded all pretraining and finetuning datasets to huggingface hub which will automatically
download when scripts are run.

## Citation
If you use this work please cite the following bibtex
```
@inproceedings{
    xie2024selfguided,
    title={Self-Guided Masked Autoencoders for Domain-Agnostic Self-Supervised Learning},
    author={Johnathan Wenjia Xie and Yoonho Lee and Annie S Chen and Chelsea Finn},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=HiYMiZYwkw}
}
```
