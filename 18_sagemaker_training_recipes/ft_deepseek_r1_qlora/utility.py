from sagemaker.modules.configs import Compute
from sagemaker.modules.configs import FileSystemDataSource
from sagemaker.modules.configs import InputData
from sagemaker.modules.configs import Networking
from sagemaker.modules.configs import SourceCode


def validate_parameter(name, value, expected_type):
    """
    Utility function to validate a parameter's type and value.

    :param name: str - The name of the parameter being validated.
    :param value: Any - The value of the parameter to validate.
    :param expected_type: type - The expected type of the parameter.
    :raises ValueError: If the value is None or empty/blank for strings.
    :raises TypeError: If the value is not of the expected type.
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be of type {expected_type.__name__}")

    if expected_type == str and not value.strip():
        raise ValueError(f"{name} cannot be empty or blank")


class ComputeCreator:
    """Class for creating compute configurations."""
    
    @staticmethod
    def create(
        instance_type: str,
        instance_count: int,
        keep_alive_period_in_seconds: int = 3600
    ) -> Compute:
        """
        Creates a compute instance with the specified configuration.

        Args:
            instance_type (str): The type of compute instance to create
            instance_count (int): Number of instances to create
            keep_alive_period_in_seconds (int, optional): Duration to keep instances alive. 
                Defaults to 3600 seconds (1 hour)

        Returns:
            Compute: A configured compute instance
        """
        validate_parameter("instance_count", instance_count,int)
        validate_parameter("keep_alive_period_in_seconds",keep_alive_period_in_seconds,int)
        validate_parameter("instance_type", instance_type,str)
        
        compute_config = {
            "instance_type": instance_type,
            "instance_count": instance_count,
            "keep_alive_period_in_seconds": keep_alive_period_in_seconds,
        }

        return Compute(**compute_config)


class FSxDataChannelCreator:
    """Class responsible for creating FSx data channels."""
    
    DEFAULT_FS_ID = 'fs-0c8e7c59a76c1facb'
    DEFAULT_CHANNEL_NAME = "modelweights"
    
    @staticmethod
    def create_channel(
        directory_path: str,
        file_system_id: str = DEFAULT_FS_ID,
        channel_name: str = DEFAULT_CHANNEL_NAME
    ) -> InputData:
        """
        Creates a data channel for FSx Lustre file system access.

        Args:
            directory_path (str): The path to the directory in the FSx file system
            file_system_id (str, optional): The ID of the FSx Lustre file system.
                Defaults to 'fs-0c8e7c59a76c1facb'
            channel_name (str, optional): Name of the data channel.
                Defaults to "modelweights"

        Returns:
            InputData: Configured data channel for FSx access
        """
        validate_parameter("file_system_id", file_system_id,str)
        validate_parameter("directory_path",directory_path,str)
        validate_parameter("channel_name", channel_name,str)
        
        file_system_input = FileSystemDataSource(
            file_system_id=file_system_id,
            file_system_type='FSxLustre',
            directory_path=directory_path,
            file_system_access_mode='rw'
        )
        
        return InputData(
            channel_name=channel_name, 
            data_source=file_system_input
        )


class NetworkConfigCreator:
    """Class responsible for creating network configurations."""
    
    @staticmethod
    def create_network_config(
        network_config: dict = None
    ) -> Networking:
        """
        Creates a network configuration for the training job.

        Args:
            network_config (dict, optional): Dictionary containing network configuration.
                Expected format:
                {
                    "subnets": ["subnet-xxx", ...],
                    "security_group_ids": ["sg-xxx", ...]
                }

        Returns:
            Networking: Configured network settings

        Raises:
            ValueError: If network_config format is invalid
        """
            
        # Validate dictionary structure
        validate_parameter("network_config", network_config, dict)
        
        # Validate required keys exist
        required_keys = ["subnets", "security_group_ids"]
        for key in required_keys:
            if key not in network_config:
                raise ValueError(f"Missing required key '{key}' in network_config")
        
        subnets = network_config["subnets"]
        security_group_ids = network_config["security_group_ids"]
        
        # Validate lists
        validate_parameter("subnets", subnets, list)
        validate_parameter("security_group_ids", security_group_ids, list)
        
        # Validate that all elements are strings
        for subnet in subnets:
            validate_parameter("subnet", subnet, str)
        for security_group in security_group_ids:
            validate_parameter("security_group", security_group, str)
            
        return Networking(
            subnets=subnets,
            security_group_ids=security_group_ids
        )

class SourceCodeCreator:
    """Class responsible for creating source code configurations."""
    
    DEFAULT_SOURCE_DIR = "scripts"
    
    @staticmethod
    def create_source_code(
        source_config: dict = None
    ) -> SourceCode:
        """
        Creates a source code configuration for the training job.

        Args:
            source_config (dict, optional): Dictionary containing source code configuration.
                Expected format:
                {
                    "source_dir": "path/to/source/dir",
                    "entry_script": "script_name.py"
                }
                Defaults to {
                    "source_dir": "scripts",
                    "entry_script": "verify_download.py"
                }

        Returns:
            SourceCode: Configured source code settings

        Raises:
            ValueError: If source_config format is invalid
        """
       
            
        # Validate dictionary structure
        validate_parameter("source_config", source_config, dict)
        print("a")
       # validate_parameter("source_config['entry_script']", source_config["entry_script"], dict)


        # Set default values if none provided
        if source_config["source_dir"] is None:
            source_config.update ({
                "source_dir": SourceCodeCreator.DEFAULT_SOURCE_DIR}
            )
        
        # Additional validation for file extension
        if not entry_script.endswith('.py'):
            raise ValueError("entry_script must have a .py extension")
        
        # Optional: Check if directory and file exist
        # import os
        # if not os.path.isdir(source_dir):
        #     raise ValueError(f"Source directory '{source_dir}' does not exist")
        # if not os.path.isfile(os.path.join(source_dir, entry_script)):
        #     raise ValueError(f"Entry script '{entry_script}' not found in '{source_dir}'")
            
        return SourceCode(
            source_dir=source_dir,
            entry_script=entry_script
        )
