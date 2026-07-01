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
            variables = config.get("variables", [])
            if isinstance(variables, list):
                for var in variables:
                    var_name = var.get("var_name")
                    var_tol = var.get("real_info", self.real_info_tol)
                    self.real_info_per_variable[var_name] = var_tol
            self.real_info_eval_freq = config.get('real_info_eval_freq', 1)
            self.bits_to_shave = []

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
    
    def shave_data(self, input_data: np.ndarray, input_dataset, variable: str, dims: np.ndarray, timestep: int, time_chunk_size = 1) -> (np.ndarray, np.ndarray):
        """
        Apply real information-based bit shaving to input data.
        
        Shaves least significant bits from floating-point data while preserving
        information content based on the configured tolerance.
        
        :param input_data: Input data array to be shaved
        :param input_dataset: Dataset object containing metadata (must have get_var_dtype method)
        :param variable: Name of the variable being processed
        :param dims: Dimensions of the input data
        :param timestep: Current timestep
        :return: Shaved data array (or original data if shaving is disabled or data is non-float)
        """
        # If we're going to be shaving data it must be on a per-snapshot basis. 
        # Note: if parameter isn't passed, it is assumed a time-independent variable is passed in, hence no need to check.
        assert(time_chunk_size == 1) 

        input_dtype = input_dataset.get_var_dtype(variable)

        level_index = -1
        if "lev" in dims:
            level_index = dims.index("lev")

        result = np.zeros(np.shape(input_data), dtype=input_dtype)

        print(f"Processing variable '{variable}' at timestep {timestep} with level index {level_index} and dims {dims}")
        
        n_levels = 1 # Default to 1 if no level dimension is present
        if level_index != -1:
            n_levels = input_data.shape[level_index]
        if self.real_info_flag and self.is_float_type(input_dtype):
            print(f"variable '{variable}' has {n_levels} levels input_data.shape={input_data.shape} level_index={level_index}")
            self.bits_to_shave.append(np.zeros(n_levels, dtype = np.int32))
            for i in range(n_levels):
                print(f"1 Processing level {i} for variable '{variable}' at timestep {timestep} n_levels={n_levels}")
                reshape_dims = input_data.shape
                idx = []
                if level_index == -1:
                    flat_array = np.asarray(input_data).flatten()
                else:
                    # extract a flattened array of the i-th level.
                    idx = [slice(None)] * input_data.ndim
                    idx[level_index] = i
                    subarray = input_data[tuple(idx)]
                    reshape_dims = subarray.shape
                    flat_array = np.asarray(subarray).flatten()
                    
                    result[tuple(idx)] = flat_array.reshape(reshape_dims)

                shave_tolerance = self.real_info_tol

                if variable in self.real_info_per_variable:
                    shave_tolerance = self.real_info_per_variable[variable]

                print(f"variable {variable} timestep {timestep} level {i}")
                if timestep % self.real_info_eval_freq == 0:
                    if level_index == -1:
                        self.bits_to_shave[-1][i] = real_info.pick_bits_to_shave_binary_search( flat_array, len(flat_array), shave_tolerance, self.bits_to_shave[-1][0])
                    else:
                        self.bits_to_shave[-1][i] = real_info.pick_bits_to_shave_binary_search( flat_array, len(flat_array), shave_tolerance, self.bits_to_shave[-1][i])
                        print(f"1 variable {variable} timestep {timestep} level {i}: bits_to_shave={self.bits_to_shave[-1][i]}, tolerance={shave_tolerance}")
            
                print(f"2 variable {variable} timestep {timestep} level {i}: bits_to_shave={self.bits_to_shave[-1][i]}, tolerance={shave_tolerance}")
                tmp_data = real_info.shave(flat_array, len(flat_array), self.bits_to_shave[-1][i])

                if level_index == -1:
                    result = tmp_data.reshape(reshape_dims)
                else:
                    result[tuple(idx)] = tmp_data.reshape(reshape_dims)
            print("bits shaved shape {bits_shaved.shape}")
            return result, np.asarray(self.bits_to_shave)
        else:
            return input_data, [np.int32(0)]
