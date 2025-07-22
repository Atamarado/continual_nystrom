#!/bin/bash

# Regenerate config_list
python3 config_list_generator.py
echo "Training configurations generated"

# Function to execute a job on a specific GPU
run_job() {
    local gpu=$1
    shift
    echo "Running on GPU $gpu with args: $@"
    which python3
    CUDA_VISIBLE_DEVICES=$gpu python3 main.py "$@" &
    echo $!  # Return the background job's process ID
}

# Assign jobs to GPUs, ensuring only one job runs per GPU at a time
assign_jobs_to_gpus() {
  # Directly use the arrays passed as arguments
  local general_jobs=("${!1}")    # Indirect referencing of array

  read -r -a all_gpus < all_gpus.txt

  declare -A job_pids
  declare -A gpus_in_use

  # Initialize job_pids and gpus_in_use
  for gpu_index in "${all_gpus[@]}"; do
    job_pids[$gpu_index]=-1
    gpus_in_use[$gpu_index]=0  # 0 means free, 1 means busy
  done

  initial_len_general=${#general_jobs[@]}
  echo "Total jobs $initial_len_general"

  # Process jobs in both queues
  while [ ${#general_jobs[@]} -gt 0 ]; do
    read -r -a gpus < gpus.txt

    # Assign general jobs to any available GPU
    for gpu_index in "${gpus[@]}"; do
      if [ ${gpus_in_use[$gpu_index]} -eq 0 ] && [ ${#general_jobs[@]} -gt 0 ]; then
        job="${general_jobs[0]}"
        general_jobs=("${general_jobs[@]:1}")  # Remove the first job from the list

        gpus_in_use[$gpu_index]=1  # Mark the GPU as busy
        run_job $gpu_index $job
        job_pids[$gpu_index]=$!  # Capture the PID of the background job
        echo "Assigned job to GPU $gpu_index"
      fi
    done

    sleep 10  # Wait some time before checking for available GPUs again

    # Check if any job has finished and free up the GPU
    for gpu_index in "${all_gpus[@]}"; do
      pid=${job_pids[$gpu_index]}
      if [ "$pid" -ne -1 ] 2>/dev/null; then  # Ensure $pid is a valid integer
        if ! kill -0 "$pid" 2>/dev/null; then  # Check if the process is still running
          # Job has finished, free the corresponding GPU
          gpus_in_use[$gpu_index]=0  # Free the GPU
          job_pids[$gpu_index]=-1  # Reset the PID
          echo "GPU $gpu_index is now free."

          len_general=${#general_jobs[@]}
          echo "Jobs left: $len_general/ $initial_len_general"
        fi
      fi
    done
  done
}

general_jobs=()

while IFS= read -r line; do
    general_jobs+=("$line")
done < "config_list.txt"

# Assign jobs to GPUs
assign_jobs_to_gpus general_jobs[@]