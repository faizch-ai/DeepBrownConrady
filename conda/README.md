# 1. Conda Virtual Environments

First you need to install Anconda or Miniconda:

* [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* [https://www.anaconda.com/download/](https://www.anaconda.com/download/)

A list of defined environments:

* [pytorch_gpu_cuda_12.1.yml](./pytorch_gpu_cuda_12.1.yml)
  * Python 3.10, Pytorch (gpu - cuda 12.1), Torchvision, Flask
* [pytorch_gpu_cuda_11.7.yml](./pytorch_gpu_cuda_11.7.yml)
  * Python 3.10, Pytorch (gpu - cuda 11.7), Torchvision, Flask
* [pytorch_cpu.yml](./pytorch_cpu.yml)
  * Python 3.10, Pytorch (cpu), Torchvision, Flask


## 1.1. Libmamba Solver

Conda's own solver is very slow, so I recommend using `Libmamba`. To use the new solver, first update conda in the base environment:

```bash
conda update -n base conda
```

Then install and activate `Libmamba` as the solver:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## 1.2. Creating a Virtual Environment

You can create a new virtual environment as follows:

```bash
conda env create -f <NAME-OF-THE-YAML-FILE>
```

, where `NAME-OF-THE-YAML-FILE` is the name of the configuration file that describes the packages of the virtual environment.

## 1.3. Activating / Deactivating a Virtual Environment

To list environments:

```bash
conda env list
```

To activate an environment:

```bash
conda env activate <NAME-OF-THE-ENVIRONMENT>
```

To deactivate an environment:

```bash
conda deactivate
```

## 1.4. Testing GPU Support in PyTorch

First activate the virtual environment that has support for PyTorch/GPU, and then run the following:

```python
import torch
torch.cuda.is_available()
```

