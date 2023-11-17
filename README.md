# SGRLW: Self-supervised Graph Representation Learning with Whitening Loss

This repository contains the reference code for the project Self-supervised Graph Representation Learning 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)
0. [Testing](#test)

## Installation
pip install -r requirements.txt 

## Preparation

Due to the space limitations, the dataset can be found in the following anonymous link:
https://drive.google.com/drive/folders/1rxIuxk0iEPHdVO1MGPQfnJaTcFgIEYVb?usp=drive_link

Important args:
* `--use_pretrain` Test checkpoints
* `--dataset` acm, imdb, dblp, freebase
* `--custom_key` Node: node classification  Clu: clustering   Sim: similarity
* `--losstype` white: Whitening loss  nowhite: no Whitening loss

## Training
python main.py

## Test
Choose the custom_key of different downstream tasks and losstype for different loss
