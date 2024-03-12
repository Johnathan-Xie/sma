# Installation
conda create --name sma_env --name python=3.10
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install .
pip install datasets evaluate
pip install scipy scikit-learn timm

# Experiments
Experiments are found in the examples directory with accompanying scripts in each directory. To ease reproducing experiments,
we have preprocessed and uploaded all pretraining and finetuning datasets to huggingface hub which will automatically
download when scripts are run.