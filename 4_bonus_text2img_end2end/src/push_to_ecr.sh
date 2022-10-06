#!/bin/bash
CHECK=$(aws ecr describe-repositories --query "repositories[?repositoryName=='sd-inference-gpu'].repositoryUri | [0]" --output text);
echo $CHECK

REGION=$(aws configure get region)
LOCALACCTID=$(aws sts get-caller-identity --query "Account" --output text)
if [ "None" == "$CHECK" ]; then
    echo "creating repo";
    aws ecr create-repository \
    --repository-name sd-inference-gpu \
    --image-scanning-configuration scanOnPush=false \
    --region $REGION;
fi


docker tag local:latest $LOCALACCTID.dkr.ecr.$REGION.amazonaws.com/sd-inference-gpu:latest

echo "pushing to ecr";

echo $LOCALACCTID.dkr.ecr.$REGION.amazonaws.com/sd-inference-gpu:latest > output.txt;

docker push $LOCALACCTID.dkr.ecr.$REGION.amazonaws.com/sd-inference-gpu:latest;

