#!/bin/bash

# Get model weights path from SageMaker environment variable
modelweights="$SM_CHANNEL_MODELWEIGHTS"

# Print model weights path for verification
echo "Printing model weights path: $modelweights"

# Display current working directory
pwd

# List contents of model weights directory with details
ls -ltr "$modelweights/"