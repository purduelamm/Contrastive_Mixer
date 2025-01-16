<p align="center">
<img src=./Images/pipeline.png width=80% height=80%>
</p>

# Contrastive_Mixer

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Contrastive Mixer (CM) model to predict Remaining Useful Life (RUL) of a machine tool. We fuse [SimCLR](https://arxiv.org/pdf/2002.05709), [MLP-Mixer](https://arxiv.org/pdf/2105.01601), and a stethoscope-based sound sensor to found a framework for RUL prediction of plasma nozzles attahced on a Plasma Arc Cutting (PAC) tool 

## Table of Contents

- [Download Process](#download-process)
- [ToDo Lists](#todo-lists)
- [How to Run](#how-to-run)
    - [Training](#training)
    - [Inference](#inference)

## Download Process

    git clone https://github.com/purduelamm/Contrastive_Mixer.git
    cd Contrastive_Mixer/
    pip3 install -r requirements.txt

## How to Run
Contrastive Mixer (CM) requires 2-steps to train (1) Backbone and (2) Head. Below are code snippets to completely train CM.

### Training:

> [!NOTE]
`train_supcon.py` and `train_reg.py` receive several different arguments. Run the `--help` command to see everything they receive.

    python3 train_supcon.py # training process for the backbone
    python3 train_reg.py # training process for the backbone

### Inference:

    python3 eval_supcon.py # inference using the backbone
    python3 eval_reg.py # inference using the head

## ToDo Lists

| **Documentation** | ![Progress](https://geps.dev/progress/50) |
| --- | --- |