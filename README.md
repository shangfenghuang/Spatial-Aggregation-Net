# **Spatial Aggregation Net: Point cloud semantic segmentation based on multi-directional convolution**

Created by Zuning Jiang, Shangfeng Huang,Kai Chen, Xuyang Ge, Huang Xu, Yundong Wu, ZongyueWang, Guorong Cai

### Installation

Install tensorflow. The code is tested under TF1.2 GPU version and Python 2.7 on Ubuntu 14.04. 

### Compile Customized TF Operators

The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary.

### Usage

Download the <a herf=[https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip](https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip)>data</a> and place it in the “data” folder under the home directory

To train model to Scene Semantic Segmentation

```python
python train.py
```

Visualize the results

```python
python visualization.py
```