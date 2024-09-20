# # Pre-train LLMs with torchtitan on Amazon SageMaker

This directory contains notebooks and resources showcasing how to pre-train LLama 3 models with torchTitan on Amazon SageMaker.

## Contents

2. [Step 1 - Build Custom Container.ipynb](./Step%201%20-Build%20Custom%20Container.ipynb)
   - Learn how to build a custom container for torchTitan on SageMaker.

1. [(Optional) Step 2 - Prepare Dataset.ipynb](./\(Optional\)%20Step%202%20-Prepare%20Dataset.ipynb)
   - This notebook guides you through the process of preparing your dataset for training with torchTitan.

3. [Step 3 - Train with torchtitan.ipynb](./Step%203-%20Train%20with%20torchtitan.ipynb)
   - This notebook demonstrates how to train a model using torchTitan on SageMaker.

## Getting Started

To get started with torchTitan on SageMaker:

1. First, build your custom container using the instructions in "Step 1 - Build Custom Container.ipynb".
2. (Optional) If you need to prepare your dataset, follow the steps in "(Optional) Step 2 - Prepare Dataset.ipynb".
3. Finally, train your model using torchTitan by following "Step 3 - Train with torchtitan.ipynb".

## Prerequisites

- An AWS account with SageMaker access
- Familiarity with Python and PyTorch
- Basic understanding of distributed training concepts

## Additional Resources

For more information on SageMaker and TorchTitan, refer to the following resources:

- [Amazon SageMaker Model Training Documentation](https://aws.amazon.com/sagemaker/train/)
- [torchTitan GitHub Repository](https://github.com/pytorch/pytorch)
