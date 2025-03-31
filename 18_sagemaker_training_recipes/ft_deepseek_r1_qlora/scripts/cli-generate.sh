FSX_PATH=$SM_CHANNEL_MODELWEIGHTS

echo $base

# Use the variable as needed
echo "***** The FSx data channel path for the model is located at: $FSX_PATH"

#ls -ltr $FSX_PATH

git clone https://github.com/deepseek-ai/DeepSeek-V3.git

cd DeepSeek-V3/inference
pip install -r requirements.txt

echo "***** Starting generating with DeepSeek"


echo "***** Printing /opt/ml/input/config/resourceconfig.json"
ls -ltr /opt/ml/input/config/resourceconfig.json

echo "***** Printing /opt/ml/code/"
ls -ltr /opt/ml/code/

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
    torchrun --nnodes 4 --nproc_per_node 8 --master_addr "$leader" --master_port 7777 --node_rank "$default_node_rank" generate.py --ckpt-path $FSX_PATH"/final_deepseek_model" --config configs/config_671B.json --input-file /opt/ml/code/prompt.txt
}

launch_distributed_training

echo "***** Completed Conversion"


 

