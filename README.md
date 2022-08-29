# Distributed Training Workshop on Amazon SageMaker
Welcome to the art and science of optimizing neural networks at scale! In this workshop you'll get hands-on experience working with our [high performance distributed training libraries](https://aws.amazon.com/sagemaker/distributed-training/) to achieve the best performance on AWS. 

## Workshop Content
Today you'll walk through two hands-on labs. The first one focuses on [***data parallelism***](https://github.com/aws-samples/sagemaker-distributed-training-workshop/tree/main/1_data_parallel), and the second one is about [***model parallelism***.](https://github.com/aws-samples/sagemaker-distributed-training-workshop/tree/main/2_model_parallel)


## Prerequisites
This lab is self-contained. All of the content you need is produced by the notebooks themselves or included in the directory. However, if you are in an AWS-led workshop you will most likely use the Event Engine to manage your AWS account. 

If not, please make sure you have an AWS account with a SageMaker Studio domain created. In this account please [request a service limit increase](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) for the `ml.g4dn.12xlarge` instance type within SageMaker training. 

## Other helpful links
If you're interested in learning more about distributed training on Amazon SageMaker, here are some helpful links in your journey.
- [Preparing data for distributed training](https://aws.amazon.com/blogs/machine-learning/choose-the-best-data-source-for-your-amazon-sagemaker-training-job/). This blog post introduces different modes of working with data on SageMaker training.
- [Distributing tabular data.](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/tabtransformer_tabular/Amazon_Tabular_Classification_TabTransformer.ipynb) This example notebook uses a built-in algorithim, `TabTransformer`, to provide state of the art ***transformer neural networks*** for tabular data. `TabTrasnformer` runs on multiple CPU-based instances.
- [SageMaker Training Compiler.](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-training-compiler) This feature enables faster training on smaller cluster sizes, decreasing the overall job time by as much as 50%. Find example notebooks for Hugging Face and TensorFlow models here, including GPT2, BERT, and VisionTransformer. Training compiler is also common in hyperparameter tuning, and can be helpful in finding the right batch size.
- [Hyperparameter tuning.](https://github.com/awslabs/syne-tune/blob/hf_blog_post/hf_blog_post/example_syne_tune_for_hf.ipynb) You can use SageMaker hyperparamter tuning, including our Syne Tune project, to find the right hyperparameters for your model, including learning rate, number of epochs, overall model size, batch size, and anything else you like. Syne Tune offers multi-objective search.
- [Hosting distributed models with DeepSpeed on SageMaker](https://github.com/dhawalkp/MLR402-reMARS-workshop/tree/master/3_deploy_gptj_with_deepspeed) In this example notebook we demonstrate using SageMaker hosting to deploy a GPT-J model using DeepSpeed.
- [Shell scripts as SageMaker entrypoint.](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-python-sdk/tensorflow_script_mode_using_shell_commands/tensorflow_script_mode_using_shell_commands.html) Want to bring a shell script so you can add any extra modifications or non-pip installable packages? Or use a wheel? No problem. This link shows you how to use a bash script to run your program on SageMaker Training.

## Top papers and case studies
Some relevant papers for your reference:

1. [SageMaker Data Parallel, aka Herring](https://www.amazon.science/publications/herring-rethinking-the-parameter-server-at-scale-for-the-cloud). In this paper we introduce a custom high performance computing configuration for distributed gradient descent on AWS, available within Amazon SageMaker Training.
2. [SageMaker Model Parallel.](https://arxiv.org/abs/2111.05972) In this paper we propose a model parallelism framework available within Amazon SageMaker Training to reduce memory errors and enable training GPT-3 sized models and more! [See our case study achieving 32 samples / second with 175B parameters on SageMaker over 140 p4d nodes.](https://aws.amazon.com/blogs/machine-learning/train-175-billion-parameter-nlp-models-with-model-parallel-additions-and-hugging-face-on-amazon-sagemaker/)
3. [Amazon Search speeds up training by 7.3x on SageMaker.](https://aws.amazon.com/blogs/machine-learning/run-pytorch-lightning-and-native-pytorch-ddp-on-amazon-sagemaker-training-featuring-amazon-search/) In this blog post we introduce two new features on Amazon SageMaker: support for native PyTorch DDP and PyTorch Lightning integration with SM DDP. We also discuss how Amazon Search sped up their overall training time by 7.3x by moving to distributed training.

## WACV Research Workshop 2023 in Hawaii - Pretrain Large Vision and Multimodal Models
If you'd like to attend our workshop in Hawaii, January 2023, [see the details here!](https://sites.google.com/view/wacv2023-workshop/home?authuser=0) Paper submissions will be due mid-October. We are inviting extended papers proposing novel concepts in the space of foundation models, in addition to a more application-focused industry poster track.


```python

```
