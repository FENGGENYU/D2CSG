# D<sup>2</sup>CSG
Codes for D<sup>2</sup>CSG(D<sup>2</sup>CSG: Unsupervised Learning of Compact CSG Trees with Dual Complements and Dropouts), Please see the [paper](https://openreview.net/pdf?id=tQYGjnxPOm).

Please leave your questions on the [issue page](https://github.com/FENGGENYU/D2CSG/issues).

### Dependencies

Requirements:
- [Pymesh](https://github.com/PyMesh/PyMesh/releases)

Please use environment.yml to install conda environment.

Tested environment: please check environment.yml file

## News
***Jan 24, 2026.*** Release the beta-version fine-tuning and test codes. The beta-version is for reference and testing only. The codes may have some redundant codes that I don't have enough time to clean.
***Feb 17, 2024.*** Release the data processing codes.

## Datasets
Download test data [Here](https://drive.google.com/file/d/1kkhYIE1MIgfmQM_rg4qlITR8YABZzUFJ/view?usp=sharing)

## training
python fine-tuning_dropout.py -e quick_test -g 0 --test --start 0 --end 1

## test
python test.py -e quick_test -g 0 -p 3 -c best_stage3_64 --test --start 0 --end 1 --csg

PS: you will find mesh in quick_test/Reconstructions_test_csg
I already placed the result in the directory, you should get a similar result as mine.

## quadric_to_basic
quadric_to_basic.py is used to change quadric primitives into basic primitives,so that it can be opened by openscad

## Citation
If you use this code, please cite the following paper.
```
@InProceedings{Yu_2022_CVPR,
    author    = {Yu, Fenggen and Chen, Zhiqin and Li, Manyi and Sanghi, Aditya and Shayani, Hooman and Mahdavi-Amiri, Ali and Zhang, Hao},
    title     = {CAPRI-Net: Learning Compact CAD Shapes With Adaptive Primitive Assembly},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {11768-11778}
}

@article{yu2024d,
  title={{D$^2$CSG}: Unsupervised Learning of Compact CSG Trees with Dual Complements and Dropouts},
  author={Yu, Fenggen and Chen, Qimin and Tanveer, Maham and Mahdavi Amiri, Ali and Zhang, Hao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```