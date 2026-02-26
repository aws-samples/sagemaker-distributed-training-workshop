export REGION=us-east-1
export ACCOUNT_ID=811828458885
export REPOSITORY_NAME=aim3315-vllm-openai
export TAG=gptoss20b

full_name="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${TAG}"

echo "building $full_name"

DOCKER_BUILDKIT=0 docker build . --network sagemaker --tag $full_name --file Dockerfile

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${REGION} --repository-names "${REPOSITIRY_NAME}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --region ${REGION} --repository-name "${REPOSITORY_NAME}" > /dev/null
fi

docker tag $REPOSITORY_NAME:$TAG ${full_name}
docker push ${full_name}
