# Data Processing
Here, we introduce codes for processing train data for CAPRI-Net and D<sup>2</sup>CSG.
These codes are modified from [IM-Net data processing](https://github.com/czq142857/IM-NET/tree/master/point_sampling)
### Dependencies

please set up the environment by using the [environment.yml](https://github.com/FENGGENYU/D2CSG/blob/main/environment.yml) file.

Compile the cython codes.
```
python setup.py build_ext --inplace
```
set up binvox 
```
chmod a+x binvox
```

## Files Structure
- 📁 dataset-name/
  - 📁 shapes/
    - 📁 file1/
      - 📄 model.obj/
    - 📁 file2/
      - 📄 model.obj

please arrange your obj files as above hierarchy structure before the following steps.

## Step 1: Produce names.npz
```
python make_name_list.py {dataset-directory} 2
```
## Step 1: Produce names.npz
```
python make_name_list.py {dataset-directory} 2
```
## Step 2: Normalize meshes
```
python 0_simplify_obj.py 0 1 {dataset-directory}
```
## Step 3: Voxelize meshes
```
python 1_voxelize.py 0 1 {dataset-directory}

```
## Step 4: Run floodfill algorithm on voxels
```
python 2_floodfill.py 0 1 {dataset-directory}

```

## Step 5-1: Sample voxels (64 resolution) and points with occupancy values
```
python imnet_sampling.py {dataset-directory}

```
ae_voxel_points_samples.hdf5 is used as ae_train.hdf5 for ShapeNet.

points_64 and values_64 in this sampled data are used as train data in the ShapeNet experiments.

## Step 5-2: Sample voxels (64 resolution) and points with occupancy values
Since the ABC dataset has better mesh property, we could also use sampling codes in [IF-Net](https://github.com/jchibane/if-net) to process the data. 

Please follow their data processing steps.

## Step 6: Sample points on 64 resolution voxels
When the input is 64^3 voxels, we can only sample points on the input for fine-tuning.
```
python voxel2pc.py 1 {dataset-directory}

```
voxel2pc.hdf5 is used as voxel2mesh.hdf5 in CAPRI-Net.

## Results
You will have the following files after data processing.
- 📁 dataset-name/
  - 📄 voxel2pc.hdf5
  - 📄 ae_voxel_points_samples.hdf5
  - 📄 test_names.npz
  
## Citation
If you use this data processing code, please cite the following paper.
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