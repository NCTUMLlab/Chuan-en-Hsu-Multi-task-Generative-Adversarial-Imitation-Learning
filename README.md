
# Introduction
This is the source code for the Master thesis Multi-task Generative Adversarial Imitation Learning from National Chiao Tung University, Taiwan. 


## Multi-task Generative Adversarial Imitation Learning




![image](MGAIL.png)
# Get starting
## Environment
The developed environment is listed in below

OS : Ubuntu 16.04
CUDA : 10.0
Nvidia Driver : 410.78
Python 3.6
Pytorch 1.2.0
The related python packages are listed in requirements.txt.


## Preprocess
### ConvLab-2 
Before starting, You should setup the ConvLab-2 package in https://github.com/thu-coai/ConvLab-2. Download the package from the link and install it.

### Create an expert dataset
We need to get a dataset for imitation learning, the record.py can help you create a dataset for our model.
* The source code is in the folder `./experiment/`.
```
    cd experiment/
```
* execute `record.py`.
```
    python record.py [--sys_policy] [--sys_path]
```

### training
```
    python train.py [--sys_policy] [--sys_path]
```
