# Introduction

This code was developed and executed using Jupyter notebooks. The following instructions assume Ubuntu 20.04 system operating with superuser access, Nvidia GPUs, GPU drivers already installed and CUDA version 10.1, 11.0 or 11.2.

# Setting Up the Environment

1. [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
2. [Install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
3. `sudo docker run --gpus all -it --shm-size 8G -p 9999:8888 -v /tmp:/tf  tensorflow/tensorflow:2.4.1-gpu-jupyter`
4. Copy the URL provided in docker logs (including the token).
5. <kbd>CTRL</kbd>+<kbd>P</kbd> then <kbd>CTRL</kbd>+<kbd>Q</kbd> to detach from the container without terminating the execution.
6. Paste the copied URL in your browser to open Jupyter (if you are running the docker container on a remote server, you need to replace the IP address with that of the server).
7. `git clone https://github.com/LukasHedegaard/continual-transformers`
8. Move `.ipynb` files inside the `notebooks` directory of the above repository and run them.
