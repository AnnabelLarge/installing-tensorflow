Tensorflow installation
========================
This is a tensorflow install with:  
1. nvidia drivers 535  
2. cuda 12.2  
3. cuDNN 8.9  
4. python 3.10  
5. tensorflow 2.15  

All done on Ubuntu 22.04.4 LTS for the NVIDIA RTX A6000 GPU. All steps are done sequentially, in this order. Yes, I know tensorflow is SUPPOSED to come with its own cuda installation. So consider this the extra paranoid guide.


1.) nvidia drivers v535
-----------------------
`sudo ubuntu-drivers autoinstall`  
This automatically pulled 535 for me.


2.) cuda 12.2
--------------
get the toolkit here: `https://developer.nvidia.com/cuda-12-2-0-download-archive`  
use the install instructions here: `https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html`


3.) cuDNN 8.9
--------------
older versions found here: `https://developer.nvidia.com/rdp/cudnn-archive`  
installed with: `sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb`


4.) python 3.10
---------------
Created a new conda environment for tensorflow. Full package list in `tf_env.yml`


5.) tensorflow 2.15
--------------------
Following recommendation from: https://blog.tensorflow.org/2023/12/tensorflow-215-update-hot-fix-linux-installation-issue.html  
`pip install tensorflow[and-cuda]==2.15.0.post1`


test it out
------------
I clipped some commands from https://learnopencv.com/implementing-mlp-tensorflow-keras/ and pasted into: `TF_mlp_test.py`. Run this script to see if tensorflow can train a neural model.


misc: remove all traces of nvidia and start over:
--------------------------------------------------
In this order:  
`sudo apt-get --purge remove "*nvidia*"`  
`sudo apt-get --purge remove "*cublas*" "*cuda*" "nsight*"`  
`sudo rm -rf /usr/local/cuda*`  

***STOP HERE IF YOU ONLY WANT TENSORFLOW***

Jax installation:
==================
1. clone tf_env to new jax_env
2. uninstall tensorflow, install pytorch
3. install Jax and friends

Using pytorch dataloaders:
---------------------------
I use pytorch dataloaders with Jax models, so I chose to install pytorch. Technically this replaces current cuda install with cuda 12.1 (which is what pytorch ships with). As long as jax can run, I'm not too pressed about this.  
Uninstall tensorflow: `pip uninstall tensorflow`.  
Install using the Pip instructions at: https://pytorch.org/get-started/locally/  
Final jax environment at: `jax_env.yml`

Use tensorflow dataloaders:
----------------------------
Tensorflow 2.15.0.post1 uses ml-dtypes~=0.2.0, but Jax uses ml-dtypes 0.4.0. I don't know how you resolve this; I guess you'd try a different tensorflow installation? ¯\_(ツ)_/¯  

install Jax and friends:
-------------------------
`pip install -U "jax[cuda12]"`  
Plus install with pip: Flax, Orbax, Optax, Diffrax

test it out
------------
I clipped some commands from https://huggingface.co/flax-community/NeuralODE_SDE/blame/955a729c0c2041e2bae8c4b3a41e3dea922bda14/models/mlp.py and pasted into: `JAX_mlp_test.py`. Run this script to see if Jax can train a neural model. Note that this does not include testing the pytorch dataloaders.

