#!/bin/bash
max_concurrent_processes=4
dataset_path=/data2/butian/GauUscene/HAV_COLMAP/ # MUST END WITH / !!!
colmap_path=${dataset_path}sparse/0
image_path=${dataset_path}images
output="/data2/butian/sep_gaussian_models/HAV_COLMAP/" # MUST END WITH / !!!
separation_folder=${output}seperation/ # Define the separation folder
model_folder=${output}models
gpus=(0 1 2 3)
gpu_count=${#gpus[@]} # List of GPUs to use
base_port=6200

cd seperation_code
python segment.py --colmap_path ${colmap_path} --output ${separation_folder} --image_lower_bound 180 --image_upper_bound 220

# cd ..
# echo use the segmentation foler in the seperation folder ${separation_folder}
# # 
# # # Define the maximum number of concurrent processes during CPU-intensive phase
# # 
# subfolders=($(ls -d "${separation_folder}"*/))
# # 
# # # Starting port number
# # 
# for colmap_folders in "${subfolders[@]}"
# do
#     # Wait if the number of running processes equals the max concurrent processes
#     while [ "$(jobs -rp | wc -l)" -ge "$max_concurrent_processes" ]; do
#         sleep 30
#     done
# 
#     # Assign a GPU in a round-robin fashion
#     gpu="${gpus[$((index % gpu_count))]}"
# 
#     # Get the next available port
#     base_port=$(($base_port + 1))  # Update base_port to avoid reuse
# 
#     basename_folder=$(basename "$colmap_folders")
#     model_path=${model_folder}/${basename_folder}
#     
#     mkdir -p $model_path
# 
#     # Build the command
#     cmd="CUDA_VISIBLE_DEVICES=${gpu} python train.py -s \"${colmap_folders}\" --images ${image_path} -r 1 --data_device cpu --port ${base_port} -m ${model_path}"
# 
#     # Define the log file path
#     log_file="${model_path}/training.log"
# 
#     # Start the training process in the background and redirect output to the log file
#     echo "Starting training on ${colmap_folders} using GPU ${gpu} and port ${base_port}"
#     nohup bash -c "$cmd" > "$log_file" 2>&1 &
# 
#     index=$((index + 1))
# 
#     # Sleep to stagger the CPU load
#     sleep 30
# done
# 
# # Wait for all background processes to finish
# wait
# # 
#  #### After training, we need to the following
#  
# cd seperation_code/
# python filtering.py --colmap_folder ${separation_folder} --model_folder ${model_folder} --output_folder ${output}