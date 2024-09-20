## Pre-train LLMs with torchtitan on Amazon SageMaker

This directory contains notebooks and resources showcasing how to pre-train LLama 3 models with torchtitan on Amazon SageMaker.

## Contents

1. [Step 1 - Build Custom Container.ipynb](./Step%201%20-Build%20Custom%20Container.ipynb)
   - Learn how to build a custom container for torchtitan on SageMaker.

2. [(Optional) Step 2 - Prepare Dataset.ipynb](./\(Optional\)%20Step%202%20-Prepare%20Dataset.ipynb)
   - This notebook guides you through the process of preparing your dataset for training with torchtitan.

3. [Step 3 - Train with torchtitan.ipynb](./Step%203-%20Train%20with%20torchtitan.ipynb)
   - This notebook demonstrates how to train a model using torchtitan on SageMaker.

## Getting Started

To get started with torchtitan on SageMaker:

1. First, build your custom container using the instructions in "Step 1 - Build Custom Container.ipynb".
2. (Optional) If you need to prepare your dataset, follow the steps in "(Optional) Step 2 - Prepare Dataset.ipynb".
3. Finally, train your model using torchtitan by following "Step 3 - Train with torchtitan.ipynb".

## Prerequisites

- [An AWS account.](https://aws.amazon.com/free/?gclid=CjwKCAjw_4S3BhAAEiwA_64YhprSXEAxE5JWxu8498Z9ayXAa08YwrqYvh7JyYinp095FXSxMRx06hoC4NsQAvD_BwE&trk=9ab5159b-247d-4917-a0ec-ec01d1af6bf9&sc_channel=ps&ef_id=CjwKCAjw_4S3BhAAEiwA_64YhprSXEAxE5JWxu8498Z9ayXAa08YwrqYvh7JyYinp095FXSxMRx06hoC4NsQAvD_BwE:G:s&s_kwcid=AL!4422!3!645133561110!e!!g!!aws%20account!19579657595!152087369744&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)
- A SageMaker domain and Amazon SageMaker Studio For instructions to create these, refer to [Quick setup to Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).
- A [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens) so you can download the Meta Llama 3 models and tokenizer to use later.
- You need to request a [quota increase](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html) of at least 1 ml.p5.48xlarge instance for training job usage on SageMaker.
- Familiarity with Python and PyTorch
- Basic understanding of distributed training concepts

## Additional Resources

For more information on SageMaker and torchtitan, refer to the following resources:

- [Amazon SageMaker Model Training Documentation](https://aws.amazon.com/sagemaker/train/)
- [torchtitan GitHub Repository](https://github.com/pytorch/pytorch)
