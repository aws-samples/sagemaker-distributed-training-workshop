# my_script.py

TRAIN_CHANNEL_PATH=$SM_CHANNEL_TRAIN

# Use the variable as needed
echo "The training data channel path is located at: $TRAIN_CHANNEL_PATH"

launch_packages(){
    # Ensure that pip is up-to-date
    python3 -m pip install --upgrade pip

    # Install the required Python package(s)
    pip install transformers==4.33.0
    pip install datasets
    pip install accelerate>=0.21
    pip install bitsandbytes
    pip install teinops
   
}

# Function to launch distributed training
launch_distributed_training() {
    # Define the path to the resource configuration file
    sm_config_path='/opt/ml/input/config/resourceconfig.json'

    echo "SM_HOSTS: $SM_HOSTS"
    
    # Get the number of hosts from the environment variable
    num_of_hosts=$(echo "$SM_HOSTS" | jq 'length')
    
    echo "num_of_hosts: $num_of_hosts"

    # Check if the resource configuration file exists
    if [ -f "$sm_config_path" ]; then
        # Parse the JSON file to get the hosts
        hosts=$(jq -r '.hosts[]' "$sm_config_path")

        # Count the number of default nodes
        default_nodes=$(echo "$hosts" | wc -l)

        # Get the current host from the environment
        current_host=$(echo "$SM_CURRENT_HOST")

        # Determine the default node rank
        default_node_rank=$(echo "$hosts" | grep -n "$current_host" | cut -d: -f1)
        default_node_rank=$((default_node_rank - 1)) # Convert to zero-based index

        # Elect a leader for PyTorch DDP
        leader=$(getent hosts $(echo "$hosts" | head -n 1) | awk '{ print $1 }')
    fi

    # Run the torchrun command
    torchrun --nnodes 2 --nproc_per_node 4 --master_addr "$leader" --master_port 7777 --node_rank "$default_node_rank" "$TRAIN_CHANNEL_PATH/allenai/launch/train.py" --bf16 True --cache_dir "/opt/ml/sagemaker/warmpoolcache" --dataset_path "/opt/ml/input/data/train" --epochs 1 --fsdp "full_shard auto_wrap" --gradient_checkpointing True --max_steps 30 --model_id "tiiuae/falcon-7b" --optimizer "adamw_torch" --per_device_train_batch_size 1 --valid_path "/opt/ml/input/data/valid"
}



# Main execution
echo "** entering now ****"
launch_packages
launch_distributed_training
    
    
    