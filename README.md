# Continual Nyströmformers official implementation

This repository contains all the code to recreate the experiments described in the paper [Continual Low-Rank Scaled Dot-product Attention](https://arxiv.org/abs/2412.03214).

## Preparing the environment and datasets

We provide in [install.sh](install.sh) a shell script to prepare a conda environment, and download the datasets necessary to run the code. Consequently, a conda installation is a necessary prerequisite. This code only offers support for computers with at least one GPU installed.

To run the installation script, execute the following command:
```bash
bash -i install.sh
```

## Run trainings
All executions call the script [main.py](main.py), with different configurations specified in the file [config.py](config.py). Remember to activate the `continual_nystrom` environment before running the script.

For example, if we want to execute the training of a Continual Nyströmformer with two layers and four landmarks for the GTZAN dataset, to run the following commands:
```bash
conda activate continual_nystrom
python main.py --model continual_nystrom --num_layers 2 --num_landmarks 4 --dataset gtzan
```

We provide a shell script to run multiple executions sequentially or in parallel (one for every GPU available) called [run_parallel.sh](run_parallel.sh). This script uses [config_list_generator.py](config_list_generator.py) to generate the training configurations that are being executed. We provide a simple example of an execution list, which can be customized.

To run the script to execute multiple trainings, execute the following command:
```bash
bash -i run_parallel.sh
```

### Multi-GPU configuration
By default, only the first GPU installed will be used for the trainings sequentially. If you want to enable parallel executions and use more GPUs, it is necessary to configure the files [all_gpus.txt](all_gpus.txt) and [gpus.txt](gpus.txt):
* **[all_gpus.txt](all_gpus.txt)**. Add here all the GPU indices that are expected to be used for any trainings, separated by a single space.
* **[gpus.txt](gpus.txt)**. Add here all the GPU indices that you want to support current trainings on, separated by a single space. Every GPU index in this file must also be included in [all_gpus.txt](all_gpus.txt).

For example, if I have a computer with 4 GPUs (with indices in the range 0-3 (by default)), but at the moment I just want to use GPUs 1 and 3, the files should be configured in the following way:

**[all_gpus.txt](all_gpus.txt)**
```
0 1 2 3
```

**[gpus.txt](gpus.txt)**
```
1 3
```

## Check the results
We also provide a simple python script to agglutinate the results in [compile_results.py](compile_results.py). The results are hold in the variable `grouped_df`, which can then be stored or handled accordingly. The results are separated in the three different tasks. The task can be selected by changing the variable `task`.

## Cite this work

If you use or modify this code, you can cite us in LaTex by using:

```
@article{Carreto24cont_nystrom, % preprint
    author  =  {Gin\'es Carreto Pic\'on and
                Illia Oleksiienko and
                Lukas Hedegaard and
                Arian Bakhtiarnia and
                Alexandros Iosifidis
               },
    title   = {Continual Low-Rank Scaled Dot-product Attention},
    journal = {arXiv:2412.03214},
    volume  = {abs/2412.03214},
    year    = {2024}
}
```

## Acknowledgments

This work has received funding by the Horizon Europe programme PANDORA (GA 101135775).

### Datasets
We would like to thank to the authors of the three different datasets that we use to run our experiments:
* **GTZAN Music Genre Classification**: Tzanetakis, G., Cook, P.R., 2002. Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing 10, 293–302.
* **THUMOS14**: Idrees, H., Zamir, A.R., Jiang, Y., Gorban, A., Laptev, I., Sukthankar, R., Shah, M., 2017. The THUMOS challenge on action recognition for videos ”in the wild”. Computer Vision and Image Understanding 155, 1–23.
* **Electricity Load Diagrams dataset**: Trindade, A., 2015. ElectricityLoadDiagrams20112014. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C58C86.

### continual-inference library

The implementation of the Continual Nyströmformers has been made as an extension of the [continual-inference library](https://github.com/LukasHedegaard/continual-inference) (version 1.2.4). The extended version of the library can be found in the folder [continual_dev](continual_dev).

* **Continual Inference**: Hedegaard, L., Iosifidis, A., 2022b. Continual inference: A library for efficient online inference with deep neural networks in pytorch, in: European
Conference on Computer Vision Workshops, pp. 21–34.
