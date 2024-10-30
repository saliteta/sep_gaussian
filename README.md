# Seperation Gaussian

Most of current code is the same as original Gaussian Splatting. 

We add a new code for seperation

- We have multiple different submodule, to clone them do:
```
 git clone https://github.com/saliteta/sep_gaussian.git --recursive
```

- And then install the environment: 
```
  conda env create -f environment.yml 
```


- Downloading the dataset or create the dataset
The large scale dataset should be downloaded or created
The structure should looks like the following: 
```
-- dataset
  |--images\ # folder for storing images  
  |--sparse\ # colmap sparse reconstruction
```


- To run it, simply change the following code with "{}" in run.sh:
```
#!/bin/bash
max_concurrent_processes=4
dataset_path="{The Location of your dataaset}"/ # MUST END WITH / !!!
colmap_path=${dataset_path}sparse/0
image_path=${dataset_path}images
output="{place to save you output model}"/ # MUST END WITH / !!!
separation_folder=${output}seperation/ # Define the separation folder
model_folder=${output}models
gpus=(0 1 2 3)
gpu_count=${#gpus[@]} # List of GPUs to use
base_port=6200
```

We provide an example in run.sh. One can also use ./run.sh