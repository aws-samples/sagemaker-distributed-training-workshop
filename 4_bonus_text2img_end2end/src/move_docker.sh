#!/bin/bash

sudo systemctl stop docker.service;
sudo systemctl stop docker.socket;

DOCKER_D=/home/ec2-user/SageMaker/.docker;
if [ ! -d $DOCKER_D ]
then
    mkdir -p $DOCKER_D;
fi

sudo chown "ec2-user":"ec2-user" $DOCKER_D;

file=/home/ec2-user/SageMaker/.docker/daemon.json;
if [ ! -f $file ]
then
    cp "/etc/docker/daemon.json" $file;
fi

if [ -z "$(grep -i data-root $file)" ]
then
    sed -i '$ s/.$//' $file;
    sudo echo ", \"data-root\": \"/home/ec2-user/SageMaker/.docker\" ,\"default-shm-size\": \"40G\",\"storage-driver\":\"overlay2\"}" >> $file;
fi
sudo cp $file "/etc/docker/";
sudo systemctl daemon-reload;
sudo systemctl start docker;

sudo chown "ec2-user":"ec2-user" $DOCKER_D;
for d in buildkit containers daemon.json image network overlay2 plugins runtimes swarm tmp trust volumes
do
  d="$DOCKER_D/$d"
  sudo chown "ec2-user":"ec2-user" $d -R;
done

if [ -d $DOCKER_D/containerd ]
then
    sudo chown "ec2-user":"ec2-user" $DOCKER_D/containerd -R;
fi

# This is common for both GPU and CPU instances

# check if we have docker-compose
docker-compose version >/dev/null 2>&1
if [ $? -ne 0 ]; then
  # install docker compose
  python -m pip install --no-cache-dir --no-deps docker-compose
fi

# check if we need to configure our docker interface
SAGEMAKER_NETWORK=`docker network ls | grep -c sagemaker-local`
if [ $SAGEMAKER_NETWORK -eq 0 ]; then
  docker network create --driver bridge sagemaker-local
fi

# Notebook instance Docker networking fixes
RUNNING_ON_NOTEBOOK_INSTANCE=`sudo iptables -S OUTPUT -t nat | grep -c 169.254.0.2`

# Get the Docker Network CIDR and IP for the sagemaker-local docker interface.
SAGEMAKER_INTERFACE=br-`docker network ls | grep sagemaker-local | cut -d' ' -f1`
DOCKER_NET=`ip route | grep $SAGEMAKER_INTERFACE | cut -d" " -f1`
DOCKER_IP=`ip route | grep $SAGEMAKER_INTERFACE | cut -d" " -f9`

# check if both IPTables and the Route Table are OK.
IPTABLES_PATCHED=`sudo iptables -S PREROUTING -t nat | grep -c $SAGEMAKER_INTERFACE`
ROUTE_TABLE_PATCHED=`sudo ip route show table agent | grep -c $SAGEMAKER_INTERFACE`
echo $ROUTE_TABLE_PATCHED;
if [ $ROUTE_TABLE_PATCHED -eq 0 ]; then
    # fix routing
    sudo ip route add $DOCKER_NET via $DOCKER_IP dev $SAGEMAKER_INTERFACE table agent
    echo "setup route table";
else
    echo "SageMaker instance route table setup is ok. We are good to go."
fi

echo $IPTABLES_PATCHED;
if [ $IPTABLES_PATCHED -eq 0 ]; then
    sudo iptables -t nat -A PREROUTING  -i $SAGEMAKER_INTERFACE -d 169.254.169.254/32 -p tcp -m tcp --dport 80 -j DNAT --to-destination 169.254.0.2:9081
    echo "iptables for Docker setup done"
else
    echo "SageMaker instance routing for Docker is ok. We are good to go!"
fi

if [ ! -f "/home/ec2-user/.sagemaker/config.yaml" ]
then
    mkdir -p /home/ec2-user/.sagemaker;
    echo "local:" > /home/ec2-user/.sagemaker/config.yaml;
    echo " container_root: ${DOCKER_D}" >> /home/ec2-user/.sagemaker/config.yaml;
fi

