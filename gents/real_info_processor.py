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
    
    def __init__(self, config_path: Optional[str] = None, n_bits_to_shave: Optional[int] = -1):
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
            self.n_bits_to_shave_default = n_bits_to_shave
            self.bits_to_shave = []
            self.natural_order = []
            self.permute_order = []

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

    def get_natural_ordering(self, input_data, dataset):# -> np.ndarray:
        """
        Reorder the input data to longitude-major order. 
        
        :param dataset: Input data array
        :return: indices that would sort the data in longitude-major order
        """

        lon = dataset.get_var_vals("lon")
        lat = dataset.get_var_vals("lat")
        lon_lat = zip(lon, lat)
        self.natural_order = sorted(range(len(lon)), key=lambda i: (lon[i], lat[i]))
        self.permute_order = [None] * len(self.natural_order)
        for natural_index, original_index in enumerate(self.natural_order):
            self.permute_order[original_index] = natural_index

    
    def shave_data(self, input_data: np.ndarray, input_dataset, variable: str, dims: np.ndarray, bits_shaved: np.ndarray, timestep: int, time_chunk_size = 1, n_bits_to_shave = -1) -> (np.ndarray, np.ndarray):
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

        if (len(self.natural_order) == 0):
            self.get_natural_ordering(input_data, input_dataset)

        n_levels = 1 # Default to 1 if no level dimension is present
        if level_index != -1:
            n_levels = input_data.shape[level_index]
        if self.real_info_flag and self.is_float_type(input_dtype):
            self.bits_to_shave = bits_shaved
            for i in range(n_levels):
                reshape_dims = input_data.shape
                idx = []
                if level_index == -1:
                    flat_array = np.asarray(input_data).flatten()
                    flat_array = flat_array[self.natural_order]
                else:
                    # extract a flattened array of the i-th level.
                    idx = [slice(None)] * input_data.ndim
                    idx[level_index] = i
                    subarray = input_data[tuple(idx)]
                    reshape_dims = subarray.shape
                    flat_array = np.asarray(subarray).flatten()
                    flat_array = flat_array[self.natural_order]
                    
                    result[tuple(idx)] = flat_array.reshape(reshape_dims)

                shave_tolerance = self.real_info_tol

                if variable in self.real_info_per_variable:
                    shave_tolerance = self.real_info_per_variable[variable]

                if self.n_bits_to_shave_default == -1:
                    if timestep % self.real_info_eval_freq == 0:
                        if level_index == -1:
                            self.bits_to_shave[i] = real_info.pick_bits_to_shave_binary_search( flat_array, len(flat_array), shave_tolerance, self.bits_to_shave[0])
                        else:
                            self.bits_to_shave[i] = real_info.pick_bits_to_shave_binary_search( flat_array, len(flat_array), shave_tolerance, self.bits_to_shave[i])
                else:
                    self.bits_to_shave[i] = self.n_bits_to_shave_default
                # XXX: Hack, limit the number of bits to shave to 15
                self.bits_to_shave[i] = min(self.bits_to_shave[i], 15)
            
                tmp_data = real_info.shave(flat_array, len(flat_array), self.bits_to_shave[i])

                if level_index == -1:
                    print(f"reshaping with level index -1, reshape dims {reshape_dims}, input_data shape {input_data.shape}")
                    result = tmp_data[self.permute_order].reshape(reshape_dims)
                    #result = tmp_data.reshape(reshape_dims)
                else:
                    print(f"reshaping with level index {level_index} reshape dims {reshape_dims}, tuple idx {tuple(idx)}")
                    result[tuple(idx)] = tmp_data[self.permute_order].reshape(reshape_dims)
            return result, np.asarray(self.bits_to_shave)
        else:
            return input_data, [np.int32(0)]
