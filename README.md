## Introduction

This repository release the baseline code for the [ICCV 2023 ABO Fine-grained Semantic Segmentation Competition](https://eval.ai/web/challenges/challenge-page/2027/overview)

## Requirements

The codes passed test on pytorch 1.7.1 and CUDA 11.2

## Train
```
python train_nodes.py -e {experiment_name} -g {gpu_id} --cate {category}
```

For example:
```
python train_nodes.py -e chair_exp -g 0 --cate chair
```

## Test
```
python test_nodes.py -e {experiment_name} -g {gpu_id} --cate {category} --split dev
```
dev for development dataset, test for test dataset

## Make Submission Json File
```
python combine_jsons.py
```

