import sagemaker
import boto3
from typing import Dict, Any

# Initialize a boto3 client for CloudFormation and EC2
cfn_client = boto3.client('cloudformation')
ec2_client = boto3.client('ec2')

def get_stack_outputs(stack_name: str, region: str = 'us-west-2') -> Dict[str, str]:
    """
    Retrieves all outputs from a CloudFormation stack.
    
    Args:
        stack_name (str): Name of the CloudFormation stack.
        region (str): AWS region where the stack is deployed (default is 'us-west-2').

    Returns:
        Dict[str, str]: A dictionary of stack outputs where keys are output names and values are output values.
    """
    try:
        # Fetch stack outputs using CloudFormation client
        response = cfn_client.describe_stacks(StackName=stack_name)
        stack_outputs = response['Stacks'][0]['Outputs']
        
        # Convert list of outputs to a dictionary for easy access
        outputs_dict = {output['OutputKey']: output['OutputValue'] for output in stack_outputs}
        return outputs_dict
    
    except Exception as e:
        print(f"Error retrieving stack outputs: {str(e)}")
        return {}

def get_subnets() -> list:
    """
    Retrieves subnets from CloudFormation stack outputs.
    
    Returns:
        list: A list of all subnet IDs.
    """
    outputs = get_stack_outputs(stack_name='cf', region=sagemaker.Session().boto_region_name)

    subnets = [
        outputs['SubnetID1'], outputs['SubnetID2'], outputs['SubnetID3'], 
        outputs['SubnetID4'], outputs['SubnetID5'], outputs['SubnetID6']
    ]

    return subnets

def get_azid_subnet_dict(outputs: Dict[str, str], desired_az_id: str = "use1-az5") -> list:
    """
    Retrieves subnets for a specified Availability Zone from CloudFormation stack outputs.
    
    Args:
        outputs (Dict[str, str]): The CloudFormation stack outputs.
        desired_az_id (str): The desired Availability Zone ID to filter subnets (default is 'use1-az5').
    
    Returns:
        list: A list of subnet IDs for the desired Availability Zone.
    """
    subnets = [
        outputs['SubnetID1'], outputs['SubnetID2'], outputs['SubnetID3'], 
        outputs['SubnetID4'], outputs['SubnetID5'], outputs['SubnetID6']
    ]
    
    # Dictionary to store subnets categorized by Availability Zone ID
    azid_subnet_dict = {}
    
    # Fetch subnet details from EC2 using the SubnetIds from CloudFormation outputs
    response = ec2_client.describe_subnets(SubnetIds=subnets)
    
    # Organize subnet IDs by their Availability Zone IDs
    for subnet in response['Subnets']:
        subnet_id = subnet['SubnetId']
        az_id = subnet['AvailabilityZoneId']
        
        if az_id not in azid_subnet_dict:
            azid_subnet_dict[az_id] = []
        
        azid_subnet_dict[az_id].append(subnet_id)
    
    # Return subnets in the desired Availability Zone (default is 'use1-az5')
    return azid_subnet_dict.get(desired_az_id, [])

def get_reinvent_network_config() -> Dict[str, Any]:
    """
    Retrieves the network configuration from the CloudFormation stack.
    
    Returns:
        Dict[str, Any]: A dictionary containing the network configuration details like subnets and security groups.
    """
    # Fetch CloudFormation stack outputs
    outputs = get_stack_outputs(stack_name='cf', region=sagemaker.Session().boto_region_name)
    
    network_config = {}
    network_config["subnets"] = get_azid_subnet_dict(outputs=outputs, desired_az_id="use1-az5")
    network_config["security_group_ids"] = [outputs['SecurityGroup']]
    
    return network_config

def get_efs_file_system_id() -> str:
    """
    Retrieves the EFS file system ID from the CloudFormation stack.
    
    Returns:
        str: The EFS file system ID (e.g., 'fs-xxxx').
    """
    # Fetch CloudFormation stack outputs
    outputs = get_stack_outputs(stack_name='cf', region=sagemaker.Session().boto_region_name)
    
    # Return the EFS FileSystemId
    return outputs.get('EFSFileSystemId', '')

def get_s3_model_artifacts() -> str:
    """
    Retrieves the S3 URI for the model artifacts from the CloudFormation stack.
    
    Returns:
        str: The S3 URI of the model artifacts (e.g., 's3://bucket-name/path/to/model').
    """
    # Fetch CloudFormation stack outputs
    outputs = get_stack_outputs(stack_name='cf', region=sagemaker.Session().boto_region_name)
    
    # Return the S3 model artifacts URI
    return outputs.get('S3ModelUri', '')
