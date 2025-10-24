# On the Bias of Next-Token Predictors Toward Systematically Inefficient Reasoning: A Shortest-Path Case Study

<p align="center">
  <a href="https://arxiv.org/abs/2507.05362">
    <img src="https://img.shields.io/badge/arXiv-2507.05362-b31b1b.svg?style=for-the-badge" alt="arXiv">
  </a>
</p>

This is the official PyTorch implementation of the paper _On the Bias of Next-Token Predictors Toward Systematically Inefficient Reasoning_, presented at NeurIPS 25. This paper shows that transformer language models trained on longer, systematic but inefficient reasoning traces (like depth-first search) generalize better on shortest-path tasks than those trained on optimal dynamic programming traces—revealing an inductive bias of next-token prediction toward locally incremental, easier-to-predict reasoning rather than globally efficient logic.

## Installation
This repo requires python 3.11, it is adviced to use `uv` as the package manager.
```
git clone https://github.com/riccardoalberghi/DP.git
cd DP
pip install -r requirements.txt
```

## Config files
The repo uses `Hydra` do manage config files. All of them are contined inside the `configs/` directory and each one has a comment on its function. When launching a new run always keep in mind to set them as desired &#128512;.

## Graph Generation & Training
Once are parameter are set correctly the training can be launched using
```
python src/dp_planning/generate_dataset.py
python src/dp_planning/train.py
```
The training will be logged in the terminal and on `wandb`. In addition a checkpoint at the end of each epoch will be created.

Using the configs in the manuscript one can expect training to last around 8 hours on a single A100 80GB. Note that VRAM required to train is ~14GB, thus all the tests can be performed on smaller GPUs, like an RTX 4090, without any change in configs and with a very small penalty in speed.

## Evaluation
During training only the test cross-entropy will be reported. To have a complete evaluation of our custom metrics launch the evalution script with
```
python src/dp_planning/evaluate.py
```
This script should take no more than an hour to run on the above specified hardware and is going to generate a `.csv` file in the experiment folder where each row represent a checkpoint evaluation.