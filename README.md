## Introduction

This repository released the baseline code for the [ICCV 2023 ABO Fine-grained Semantic Segmentation Competition](https://eval.ai/web/challenges/challenge-page/2027/overview)

## Requirements

The codes passed test on pytorch 1.7.1 and CUDA 11.2

## Processed data

The processed data is released [here](https://drive.google.com/file/d/1S7Ove63KvuN1pVgz1aNGG1wg3QHLJJL_/view?usp=sharing). It only contains train data and dev data right now.

Please set up the data path in the spec.json file under the experiment directory(e.g. chair_exp).

We also provide the data preparation codes in pointcloud_sampling.py.
```
python pointcloud_sampling.py --cate {category} --split {data_split} --in {input_data_path} --output {output_data_path}
```
The input_data_path above should be the path of downloaded data from ABO website.

For example:
```
python pointcloud_sampling.py --cate chair --split train
```

## Train
```
python train.py -e {experiment_name} -g {gpu_id} --cate {category}
```

For example:
```
python train.py -e chair_exp -g 0 --cate chair
```

## Test
```
python test.py -e {experiment_name} -g {gpu_id} --cate {category} --split dev
```
dev for development dataset, test for test dataset

## Make Submission Json File
```
python combine_jsons.py
```

