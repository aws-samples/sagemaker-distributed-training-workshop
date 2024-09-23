## Fine-tune Meta Llama 3.1 models using torchtune on Amazon SageMaker Training jobs

In this example, we use Meta's torchtune library to fine-tune Llama-like architectures with any of the supported fine-tuning strategies on your choice of compute and libraries, all within a highly managed and cost-effective environment provided by Amazon SageMaker Training. This example demonstrates this through a step-by-step implementation of fine-tuning, model inference, quantizing, and evaluating the Llama 3.1 8B model on 8 Nvidia A100 GPUs, utilizing 1 p4d.24xlarge worker node and LoRA fine-tuning strategy. 

Follow the steps below to run this example:

<u>**Step 1.**</u>: Run notebook "0_build_vpc_setup.ipynb"
This executes a CFN template to launch the necessary AWS infrastructure. This notebook has the necessary cells to launch the CFN stack, create VPC, NAT, Security Group, and EFS, which you can use to securely run the training job with access to EFS in the next steps. 

<u>**Step 2.**</u>: Run notebook "1_build_container.ipynb"
As part of our example, we are using one of the key value propositions of SageMaker that allows you to bring in any of your custom libraries by creating your own containers. Here, for the purposes of this demo, we are extending from one of the SageMaker built-in images and adding our own libraries like nightly torch, custom lm_eval task, etc., and creating the container image with the Docker file below.

This notebook builds and pushes the custom Docker file to your ECR repository. We use sm-docker, which is a CLI tool designed for building Docker images in Amazon SageMaker Studio using AWS CodeBuild. We will install the library as part of the notebook. 

<u>**Step 3.**</u>: Run notebook "2_torchtune-llama3_1.ipynb"
This notebook walks you through an end-to-end example of how you can fine-tune a Llama 3.1 8B model with LoRA, run generation in memory, and optionally quantize and evaluate the model using torchtune and SageMaker training.  

Recipes, prompt templates, configs, and datasets are completely configurable and allow you to align torchtune to your requirements. To demonstrate this, we will use a custom prompt template in this use case with the open-source Hugging Face Samsung/samsum dataset.

We'll fine-tune using torchtune multi-device LoRA recipe (lora_finetune_distributed) and use the SageMaker customized version of Llama 3.1 8B default config (llama3_1/8B_lora).
