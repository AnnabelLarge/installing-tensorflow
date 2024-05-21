This is a tensorflow install with:
===================================
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


6.) test it out
----------------
I clipped some commands from https://learnopencv.com/implementing-mlp-tensorflow-keras/ and pasted into: `mlp_test.py`. Run this script to see if tensorflow can train a neural model.



misc: remove all traces of nvidia and start over:
--------------------------------------------------
In this order:  
`sudo apt-get --purge remove "*nvidia*"`  
`sudo apt-get --purge remove "*cublas*" "*cuda*" "nsight*"`  
`sudo rm -rf /usr/local/cuda*`  


why did I write this guide?
---------------------------
I kept having issues with Jax (and tensorflow) not being able to see my cuDNN installation. Jax installation has yet to be verified, but a good first step is to try installing tensorflow first. 
