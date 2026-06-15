# Training Falcon-7B with FSx Lustre and SageMaker ModelTrainer API

This example demonstrates how to fine-tune [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) using:

- **SageMaker SDK v3 `ModelTrainer` API**
- **FSx for Lustre** as a high-performance data channel (auto-synced with S3)
- **PyTorch FSDP** for distributed training across multiple GPUs
- **Warm Pools** for faster iteration (15-minute keep-alive)

## Architecture

```
┌─────────────┐       ┌──────────────────────────────────────────┐
│   S3 Bucket │◄─────►│          FSx for Lustre                  │
│ (training   │ sync  │  (high-throughput filesystem)             │
│  data)      │       └────────────────┬─────────────────────────┘
└─────────────┘                        │ mount
                                       ▼
                        ┌──────────────────────────────┐
                        │   SageMaker Training Job     │
                        │   (FSDP across GPUs)         │
                        │   VPC: private subnet        │
                        │   NAT Gateway for pip        │
                        └──────────────────────────────┘
```

## Prerequisites

1. An AWS account with SageMaker, FSx, and CloudFormation permissions
2. An AWS CLI profile configured (e.g., `claude`) with valid credentials
3. An S3 bucket for training data

## Quick Start

### 1. Deploy the infrastructure

Deploy the CloudFormation stack to create the VPC, FSx Lustre filesystem, NAT Gateway, and networking:

```bash
aws cloudformation create-stack \
  --stack-name fsx-setup \
  --template-body file://cfn-vpc-fsx-lustre.yaml \
  --parameters ParameterKey=S3BucketName,ParameterValue=<your-bucket-name> \
  --profile <your-profile> --region us-west-2
```

### 2. Run the notebook

Open `smddp_fsdp_example.ipynb` and run all cells. The notebook will:

1. Read infrastructure config from the CloudFormation stack outputs
2. Tokenize the GLUE/SST2 dataset and upload to S3 (auto-syncs to FSx)
3. Launch a distributed training job using ModelTrainer with FSx Lustre input

## Files

| File | Description |
|------|-------------|
| `smddp_fsdp_example.ipynb` | End-to-end notebook (data prep, upload, training) |
| `cfn-vpc-fsx-lustre.yaml` | CloudFormation template (VPC, FSx, NAT Gateway, networking) |
| `scripts/train.py` | FSDP training entry point |
| `scripts/utils.py` | DataLoader and model save helpers |
| `scripts/requirements.txt` | Runtime dependencies for training container |

## Key Features

- **No hardcoded infrastructure IDs** — all config is read dynamically from CloudFormation stack outputs
- **Warm Pools** — 15-minute keep-alive reduces startup time for iterative training
- **FSx Lustre** — provides high-throughput data access vs. S3 direct download
- **NAT Gateway** — allows the training container to install pip packages from PyPI

## Environment

This notebook was developed and tested with the following local environment:

| Component | Version |
|-----------|---------|
| Python | 3.11.13 |
| OS | macOS (arm64 / Apple Silicon) |
| sagemaker | 3.13.1 |
| boto3 | 1.43.29 |
| transformers | 5.12.1 |
| datasets | 5.0.0 |
| torch | 2.12.0 |
| accelerate | 1.14.0 |

**Training container image:**

```
763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker
```

- Python 3.10, PyTorch 2.0.1, CUDA 11.8, Ubuntu 20.04

To reproduce, create a virtual environment with Python 3.11+ and install:

```bash
pip install sagemaker boto3 transformers datasets torch accelerate
```

## Cleanup

Delete the CloudFormation stack to avoid ongoing charges (NAT Gateway ~$0.045/hr, FSx Lustre storage):

```bash
aws cloudformation delete-stack --stack-name fsx-setup --profile <your-profile> --region us-west-2
```
