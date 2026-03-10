#!/usr/bin/env python
"""
real_info_processor.py

Handles real information compression and bit shaving for floating-point data.
"""
import numpy as np
import sys
from pathlib import Path
from typing import Any

# Add path to access the real_info module from real-information package
sys.path.insert(1, str(Path(__file__).parent.parent.parent / 'real-information' / 'src'))
import real_info


class RealInfoProcessor:
    """
    Processor for real information-based data compression.
    
    This class handles the shaving of least significant bits from floating-point
    data while preserving real information based on a specified tolerance.
    """
    
    def __init__(self, real_info_flag: bool = False, real_info_tol: float = 0.99):
        """
        Initialize the RealInfoProcessor.
        
        :param real_info_flag: Whether to enable real information compression
        :param real_info_tol: Tolerance threshold for information preservation (0-1)
        """
        self.real_info_flag = real_info_flag
        self.real_info_tol = real_info_tol
    
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
    
    def shave_data(self, input_data: np.ndarray, input_dataset, variable: str) -> np.ndarray:
        """
        Apply real information-based bit shaving to input data.
        
        Shaves least significant bits from floating-point data while preserving
        information content based on the configured tolerance.
        
        :param input_data: Input data array to be shaved
        :param input_dataset: Dataset object containing metadata (must have get_var_dtype method)
        :param variable: Name of the variable being processed
        :return: Shaved data array (or original data if shaving is disabled or data is non-float)
        """
        input_dtype = input_dataset.get_var_dtype(variable)
        
        if self.real_info_flag and self.is_float_type(input_dtype):
            # Flatten array for processing
            flat_array = np.asarray(input_data).flatten()
            
            # Determine number of bits to shave
            bits_to_shave = real_info.pick_bits_to_shave_binary_search(
                flat_array, 
                len(flat_array), 
                self.real_info_tol, 
                0
            )
            
            # Shave the bits
            tmp_data = real_info.shave(flat_array, len(flat_array), bits_to_shave)
            
            # Reshape back to original shape
            tmp_data = tmp_data.reshape(np.shape(input_data))
            return tmp_data
        else:
            return input_data
