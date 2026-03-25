#!/usr/bin/env python
"""
real_info_processor.py

Handles real information compression and bit shaving for floating-point data.
"""
import numpy as np
import sys
import yaml
from pathlib import Path
from typing import Any, Optional

# Add path to access the real_info module from real-information package
sys.path.insert(1, str(Path(__file__).parent.parent.parent / 'real-information' / 'src'))
import real_info


class RealInfoProcessor:
    """
    Processor for real information-based data compression.
    
    This class handles the shaving of least significant bits from floating-point
    data while preserving real information based on a specified tolerance.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RealInfoProcessor.
        
        :param config_path: Path to YAML configuration file containing real info settings
        """
        if config_path is not None:
            # Load configuration from YAML file
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.real_info_flag = config.get('use_real_info', False)
            self.real_info_tol = config.get('default_real_info', 0.99)
            self.real_info_per_variable = {}
            for var in config["variables"]:
                var_name = var.get("var_name")
                var_tol = var.get("real_info", self.real_info_tol)
                self.real_info_per_variable[var_name] = var_tol

            #print(f"using real info: {self.real_info_flag}")
            #print(f"real info processor intialized with config: {config}")
        else:
            # Use provided parameters (backward compatibility)
            self.real_info_flag = False
            self.real_info_tol =  0.99
    
    @staticmethod
    def is_float_type(x: Any) -> bool:
        """
        Check if a type or dtype is a floating-point type.
        
        :param x: Type or dtype to check
        :return: True if x is a floating-point type, False otherwise
        """
        if x is float:
            return True
        elif x == "float32":
            return True
        elif x == "float64":
            return True
        try:
            return issubclass(x, np.floating)
        except TypeError:
            return False
    
    def shave_data(self, input_data: np.ndarray, input_dataset, variable: str, time_chunk_size = 1) -> (np.ndarray, np.int32):
        """
        Apply real information-based bit shaving to input data.
        
        Shaves least significant bits from floating-point data while preserving
        information content based on the configured tolerance.
        
        :param input_data: Input data array to be shaved
        :param input_dataset: Dataset object containing metadata (must have get_var_dtype method)
        :param variable: Name of the variable being processed
        :return: Shaved data array (or original data if shaving is disabled or data is non-float)
        """
        # If we're going to be shaving data it must be on a per-snapshot basis. 
        # Note: if parameter isn't passed, it is assumed a time-independent variable is passed in, hence no need to check.
        assert(time_chunk_size == 1) 

        input_dtype = input_dataset.get_var_dtype(variable)
        
        if self.real_info_flag and self.is_float_type(input_dtype):
            flat_array = np.asarray(input_data).flatten()

            shave_tolerance = self.real_info_tol

            if variable in self.real_info_per_variable:
                shave_tolerance = self.real_info_per_variable[variable]
            
            bits_to_shave = real_info.pick_bits_to_shave_binary_search( flat_array, len(flat_array), shave_tolerance, 0)
            
            tmp_data = real_info.shave(flat_array, len(flat_array), bits_to_shave)
            
            tmp_data = tmp_data.reshape(np.shape(input_data))
            return tmp_data, bits_to_shave
        else:
            return input_data, np.int32(0)
