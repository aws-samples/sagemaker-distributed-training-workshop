#!/bin/bash
DEFAULT_R=$(aws configure get region)
LOCALACCTID=$(aws sts get-caller-identity --query "Account" --output text)
while getopts a:r: flag
do
    case "${flag}" in
        a) ACCTNUM=${OPTARG};;
        r) REGION=${OPTARG};;
    esac
done
if [ -z "$REGION" ]; then 
  echo "Region is not set, using default config. To set use -r flag";
  REGION=$DEFAULT_R;
fi

if [ -z "$ACCTNUM" ]; then
  echo "Account Num is not set, using default for us-east-1 which is 763104351884 to set use -a flag";
  ACCTNUM=763104351884;
fi

echo "Region is $REGION"
echo "AcctNum is $ACCTNUM"
docker login -u AWS -p $(aws ecr get-login-password --region $REGION) $ACCTNUM.dkr.ecr.$REGION.amazonaws.com;
docker login -u AWS -p $(aws ecr get-login-password --region $REGION) $LOCALACCTID.dkr.ecr.$REGION.amazonaws.com;

echo $(pwd);
DOCKER_BUILDKIT=1 docker build $(pwd)/src/ -f $(pwd)/src/Dockerfile-Inf -t local:latest -q &




