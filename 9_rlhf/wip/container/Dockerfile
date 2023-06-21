FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker

COPY requirements.txt, main.py, utils.py /opt/ml/code/

RUN pip install -r requirements.txt
