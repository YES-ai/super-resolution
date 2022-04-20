# MaskGIT for Single Image Super-Resolution

This repository contains the demo of finetuning pretrained models of [[MaskGIT](https://arxiv.org/abs/2202.04200)] for the Single Image Super-Resolution(SISR) task.

## Summary
Our model aims to explore the possibility of applying MaskGIT's bidirectional transformer decoder on the task of SISR. We finetune MaskGIT to learn to predict tokens with high-res details given tokens with low-res details.

At inference, we follow the same approach as MaskGIT, which is to generate all tokens in each iteration, and then proportionally pick the top confident ones based on the current step.

## Demo
Please refer to super_resolution_demo.ipynb and training_demo.ipynb.
