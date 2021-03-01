# Graph-RNN
We propose a new neural network with Graph-RNN cells, for point cloud sequence prediction


[[Project]](https://github.com/pedro-dm-gomes/Graph-RNN) [[Paper]](https://arxiv.org/abs/2102.07482)     


## Overview
<img src="https://github.com/pedro-dm-gomes/Graph-RNN/blob/main/full_scheme.png" scale="0.2">

## Citation
Please cite this paper if you want to use it in your work,

	@article{gomes2021spatiotemporal,
	  title={Spatio-temporal Graph-RNN for Point Cloud Prediction},
	  author={Pedro Gomes and Silvia Rossi and Laura Toni},
	  year={2021},
	  eprint={2102.07482},
	  archivePrefix={arXiv},
	  }
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7.21

Compile the code. You will need to select the correct CUDA version and Tensorflow directions in the Makefile

### Usage
To train a model to long-term prediction using the GraphRNN model using Human Bodies Dataset :

    python train-GraphRNN_LongTerm_without_color.py

to evaluate the model
	
    python eval-Bodies_GraphRNN_Long_Term_without_color

### Datasets
To evaluate the MNIST dataset (Point RNN) 
1. [Moving MNIST Point Cloud (1 digit)](https://drive.google.com/open?id=17RpNwMLDcR5fLr0DJkRxmC5WgFn3RwK_) &emsp; 2. [Moving MNIST Point Cloud (2 digits)](https://drive.google.com/open?id=11EkVsE5fmgU5D5GsOATQ6XN17gmn7IvF) &emsp;


## Acknowledgement
The parts of this codebase is borrowed from Related Repos:

### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn


