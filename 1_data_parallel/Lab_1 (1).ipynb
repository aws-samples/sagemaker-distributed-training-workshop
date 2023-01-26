{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "65020424",
   "metadata": {},
   "source": [
    "# Data Parallel on Amazon SageMaker Training with PyTorch Lightning and Warm Pools\n",
    "In this lab we'll write our own deep neural network using PyTorch Lightning, and train this on Amazon SageMaker using multiple GPUs. For more details [see our blog post here.](https://aws.amazon.com/blogs/machine-learning/run-pytorch-lightning-and-native-pytorch-ddp-on-amazon-sagemaker-training-featuring-amazon-search/)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2951bd9c",
   "metadata": {},
   "source": [
    "### Step 0. Update the SageMaker Python SDK and AWS botocore"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4b157cb",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "%pip install --upgrade sagemaker\n",
    "%pip install boto3 --upgrade\n",
    "%pip install botocore --upgrade"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67a4f2ca-8267-47ed-be54-1047827287af",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from packaging import version\n",
    "import botocore\n",
    "\n",
    "your_botocore_version = botocore.__version__\n",
    "\n",
    "minimal_version = '1.27.90'\n",
    "\n",
    "if version.parse(your_botocore_version) >= version.parse(minimal_version):\n",
    "    print ('You are all set! Please enjoy the lab')\n",
    "else:\n",
    "    print ('Stop! Please install the packages in the cell above before continuing.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "440c79da-ad43-4fbb-9223-137c46d5170e",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# put a string here so you know which jobs are yours, no puncutation or spaces\n",
    "your_user_string = 'emily'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "26c9fe74-5fb5-402f-b1e6-e66d8c9dd7a5",
   "metadata": {},
   "source": [
    "### Step 1. Upload a dataset to your S3 bucket\n",
    "The example script we're using points to the MNIST Data loader directly from the training instance, which completely bypasses S3. However, for the sake of argument, we'll show you how to load some sample data from your notebook into S3, and then from S3 onto the training instances. This is useful for larger datasets and storage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1fa7dfd0-ce89-4ab3-8bfe-c472d6f5a650",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "%%writefile train.csv\n",
    "this,is,my,arbitrary,csv,file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68a0c64d-5947-48cc-8c85-1ab53e60cd41",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import sagemaker\n",
    "\n",
    "sess = sagemaker.Session()\n",
    "\n",
    "# optionally point to whichever bucket you have access to \n",
    "bucket = sess.default_bucket()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8899841-6407-4165-a8c2-4bd83828accb",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "s3_train_path = 's3://{}/data/mnist/'.format(bucket)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84a2b3a0-235c-4682-a51b-21b7a515d85c",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "!aws s3 cp train.csv {s3_train_path}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf242fda",
   "metadata": {},
   "source": [
    "### Step 2. Write train script and requirements into a local directory, here named `scripts`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a4ecc67",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "!mkdir scripts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6645e470",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "%%writefile scripts/requirements.txt\n",
    "pytorch-lightning == 1.6.3\n",
    "lightning-bolts == 0.5.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7f7dbc3",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "%%writefile scripts/mnist.py\n",
    "\n",
    "import os\n",
    "import torch\n",
    "from torch.nn import functional as F\n",
    "\n",
    "import pytorch_lightning as pl\n",
    "from pytorch_lightning.strategies import DDPStrategy\n",
    "\n",
    "from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment\n",
    "from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule\n",
    "\n",
    "import argparse\n",
    "\n",
    "class LitClassifier(pl.LightningModule):\n",
    "    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):\n",
    "        super().__init__()\n",
    "        self.save_hyperparameters()\n",
    "\n",
    "        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)\n",
    "        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = x.view(x.size(0), -1)\n",
    "        x = torch.relu(self.l1(x))\n",
    "        x = torch.relu(self.l2(x))\n",
    "        return x\n",
    "\n",
    "    def training_step(self, batch, batch_idx):\n",
    "        x, y = batch\n",
    "        y_hat = self(x)\n",
    "        loss = F.cross_entropy(y_hat, y)\n",
    "        return loss\n",
    "\n",
    "    def validation_step(self, batch, batch_idx):\n",
    "        x, y = batch\n",
    "        probs = self(x)\n",
    "        acc = self.accuracy(probs, y)\n",
    "        return acc\n",
    "\n",
    "    def test_step(self, batch, batch_idx):\n",
    "        x, y = batch\n",
    "        logits = self(x)\n",
    "        acc = self.accuracy(logits, y)\n",
    "        return acc\n",
    "\n",
    "    def accuracy(self, logits, y):\n",
    "        acc = (logits.argmax(dim=-1) == y).float().mean()\n",
    "        return acc\n",
    "\n",
    "    def validation_epoch_end(self, outputs) -> None:\n",
    "\n",
    "        self.log(\"val_acc\", torch.stack(outputs).mean(), prog_bar=True)\n",
    "\n",
    "    def test_epoch_end(self, outputs) -> None:\n",
    "        self.log(\"test_acc\", torch.stack(outputs).mean())\n",
    "\n",
    "    def configure_optimizers(self):\n",
    "        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)\n",
    "\n",
    "    \n",
    "def parse_args():\n",
    "    parser = argparse.ArgumentParser()\n",
    "\n",
    "    parser.add_argument(\"--hosts\", type=list, default=os.environ[\"SM_HOSTS\"])\n",
    "    parser.add_argument(\"--current-host\", type=str, default=os.environ[\"SM_CURRENT_HOST\"])\n",
    "    parser.add_argument(\"--model-dir\", type=str, default=os.environ[\"SM_MODEL_DIR\"])\n",
    "    parser.add_argument(\"--train-dir\", type=str, default=os.environ[\"SM_CHANNEL_TRAIN\"])\n",
    "    parser.add_argument(\"--num-gpus\", type=int, default=int(os.environ[\"SM_NUM_GPUS\"]))\n",
    "\n",
    "    parser.add_argument(\"--num_nodes\", type=int, default = len(os.environ[\"SM_HOSTS\"]))\n",
    "           \n",
    "    # num gpus is per node\n",
    "    world_size = int(os.environ[\"SM_NUM_GPUS\"]) * len(os.environ[\"SM_HOSTS\"])\n",
    "                 \n",
    "    parser.add_argument(\"--world-size\", type=int, default=world_size)\n",
    "    \n",
    "    parser.add_argument(\"--batch_size\", type=int, default=int(os.environ[\"SM_HP_BATCH_SIZE\"]))  \n",
    "    \n",
    "    parser.add_argument(\"--epochs\", type=int, default=int(os.environ[\"SM_HP_EPOCHS\"]))\n",
    "\n",
    "    \n",
    "    args = parser.parse_args()\n",
    "    \n",
    "    return args\n",
    "    \n",
    "    \n",
    "if __name__ == \"__main__\":\n",
    "        \n",
    "    args = parse_args()\n",
    "    \n",
    "    cmd = 'ls {}'.format(args.train_dir)\n",
    "    \n",
    "    print ('Here is sample arbitrary csv train file!')\n",
    "    \n",
    "    os.system(cmd)\n",
    "    \n",
    "    dm = MNISTDataModule(batch_size=args.batch_size)\n",
    "    \n",
    "    model = LitClassifier()\n",
    "    \n",
    "    local_rank = os.environ[\"LOCAL_RANK\"]\n",
    "    torch.cuda.set_device(int(local_rank))\n",
    "    \n",
    "    num_nodes = args.num_nodes\n",
    "    num_gpus = args.num_gpus\n",
    "    \n",
    "    env = LightningEnvironment()\n",
    "    \n",
    "    env.world_size = lambda: int(os.environ.get(\"WORLD_SIZE\", 0))\n",
    "    env.global_rank = lambda: int(os.environ.get(\"RANK\", 0))\n",
    "    \n",
    "    ddp = DDPStrategy(cluster_environment=env, accelerator=\"gpu\")\n",
    "    \n",
    "    trainer = pl.Trainer(max_epochs=args.epochs, strategy=ddp, devices=num_gpus, num_nodes=num_nodes, default_root_dir = args.model_dir)\n",
    "    trainer.fit(model, datamodule=dm)\n",
    "    trainer.test(model, datamodule=dm)\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d081b87f",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Step 3. Configure the SageMaker Training Estimator\n",
    "In this step you are using the [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) as a wrapper around the core api, `create-training-job`, [as described here.](https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-training-job.html)\n",
    "\n",
    "Read more about [training on SageMaker here,](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html) with distributed training details [here.](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)\n",
    "\n",
    "\n",
    "You can see [instance details for SageMaker here,](https://aws.amazon.com/sagemaker/pricing/) along with [instance specs from EC2 directly here.](https://aws.amazon.com/ec2/instance-types/)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a3d7a9d",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import sagemaker\n",
    "from sagemaker.pytorch import PyTorch\n",
    "from sagemaker.local import LocalSession\n",
    "\n",
    "sagemaker_session = sagemaker.Session()\n",
    "role = sagemaker.get_execution_role()\n",
    "region = sagemaker_session.boto_region_name\n",
    "\n",
    "# hard code point to the DLC images\n",
    "image_uri = '763104351884.dkr.ecr.{}.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker'.format(region)\n",
    "\n",
    "estimator = PyTorch(\n",
    "  entry_point=\"mnist.py\",\n",
    "  base_job_name=\"{}-ddp-mnist\".format(your_user_string),\n",
    "  image_uri = image_uri,\n",
    "  role=role,\n",
    "  source_dir=\"scripts\",\n",
    "  # configures the SageMaker training resource, you can increase as you need\n",
    "  instance_count=1,\n",
    "  instance_type=\"ml.g4dn.12xlarge\",\n",
    "  py_version=\"py38\",\n",
    "  sagemaker_session=sagemaker_session,\n",
    "  distribution={\"pytorchddp\":{\"enabled\": True}},\n",
    "  debugger_hook_config=False,\n",
    "  #profiler_config=profiler_config,\n",
    "  hyperparameters={\"batch_size\":32, \"epochs\":300},\n",
    "  # enable warm pools for 20 minutes\n",
    "  keep_alive_period_in_seconds = 20 *60)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50bcc311",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Passing True will halt your kernel, passing False will not. Both create a training job.\n",
    "# here we are defining the name of the input train channel. you can use whatever name you like! up to 20 channels per job.\n",
    "estimator.fit(wait=True, inputs = {'train':s3_train_path})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "276fbfa0-d8f6-4757-9ee4-d44c8a891964",
   "metadata": {},
   "source": [
    "### Step 4. Rerun the job with a higher batch size to increase GPU Utilization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "505418e7-6078-4aa4-a288-06bb72664b9b",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "estimator = PyTorch(\n",
    "  entry_point=\"mnist.py\",\n",
    "  base_job_name=\"{}-ddp-mnist\".format(your_user_string),\n",
    "  image_uri = image_uri,\n",
    "  role=role,\n",
    "  source_dir=\"scripts\",\n",
    "  # configures the SageMaker training resource, you can increase as you need\n",
    "  instance_count=1,\n",
    "  instance_type=\"ml.g4dn.12xlarge\",\n",
    "  py_version=\"py38\",\n",
    "  sagemaker_session=sagemaker_session,\n",
    "  distribution={\"pytorchddp\":{\"enabled\": True}},\n",
    "  debugger_hook_config=False,\n",
    "  #max_retry_attempts=5,\n",
    "  hyperparameters={\"batch_size\":320, \"epochs\":900},\n",
    "  # turn off warm pools for this instance\n",
    "  keep_alive_period_in_seconds = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1df8c3a0-3e07-47cc-a14c-668398ebbb8b",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "estimator.fit(wait=False, inputs = {'train':s3_train_path})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "579d948f",
   "metadata": {},
   "source": [
    "### Step 5. Optimize!\n",
    "That's the end of this lab. However, in the real world, this is just the begining. Here are a few extra things you can do to increase model accuracy (hopefully) and decrease job runtime (certainly).\n",
    "1. **Increase throughput by adding extra nodes.** Increase the number of instances in `instance_count`, and as long as you're using some for of distribution in your training script, this will automatically copy your model over all available accelerators and handle averaging the results. This is also called ***horizontal scaling.***\n",
    "2. **Increase throughput by upgrading your instances.** In addition to (or sometimes instead of) adding extra nodes, you can increase your instance to something larger. This usually means adding more accelerators, more CPU, more memory, and more bandwidth. \n",
    "3. **Increase accuracy with hyperparameter tuning**. Another critical step is picking the right hyperparameters. You can use [Syne Tune](https://github.com/awslabs/syne-tune/blob/hf_blog_post/hf_blog_post/example_syne_tune_for_hf.ipynb) for a multi-objective tuning metric as one example. Amazon SageMaker Automatic Model Tuning now provides up to three times faster hyperparameter tuning with [Hyperband](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-now-provides-up-to-three-times-faster-hyperparameter-tuning-with-hyperband/).  Here's another example with [SageMaker Training Compiler to tune the batch size!](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-training-compiler/huggingface/pytorch_tune_batch_size/finding_max_batch_size_for_model_training.ipynb)\n",
    "4. **Increase accuracy by adding more parameters to your model, and using a model parallel strategy.** [This is the content of the next lab!](https://github.com/aws-samples/sagemaker-distributed-training-workshop/blob/main/2_model_parallel/smp-train-gpt-simple.ipynb)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6ae8317",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "instance_type": "ml.t3.medium",
  "kernelspec": {
   "display_name": "Python 3 (Data Science)",
   "language": "python",
   "name": "python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:081325390199:image/datascience-1.0"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
