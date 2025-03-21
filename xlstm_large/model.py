import logging
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our AMD-specific optimizations
try:
    from mlstm_kernels.triton.amd_detection import is_amd_gpu, is_mi300x
    from mlstm_kernels.triton.kernel_param_heuristics import get_optimized_kernel_config
    is_amd = is_amd_gpu()
    is_amd_mi300x = is_mi300x()
except ImportError:
    is_amd = False
    is_amd_mi300x = False
    
    def get_optimized_kernel_config(*args, **kwargs):
        return {
            "chunkwise_kernel": "chunkwise--native_autograd",
            "sequence_kernel": "native_sequence__native",
            "step_kernel": "native"
        }

@dataclass
class xLSTMLargeConfig:
    embedding_dim: int = 4096
    num_heads: int = 32
    num_blocks: int = 32
    vocab_size: int = 32000
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    add_out_norm: bool = True
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    sequence_kernel: str = "native_sequence__triton"
    step_kernel: str = "triton"
    mode: str = "inference"
    chunk_size: int = 64
    return_last_states: bool = True
    autocast_kernel_dtype: str = "bfloat16"
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    ffn_proj_factor: float = 8 / 3
    ffn_round_up_to_multiple_of: int = 64
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    weight_mode: str = "single"
    
    def __post_init__(self):
        # Automatic selection of optimized kernels for AMD hardware
        if is_amd:
            # Get head dimension
            head_dim = self.embedding_dim // self.num_heads
            
            # For MI300X, we have specific optimizations
            if is_amd_mi300x:
                # Use our batch-aware kernel selection
                # Default to batch size 1 for initial config
                optimal_config = get_optimized_kernel_config(1, 2048, head_dim)
                
                # Apply the optimal configuration if it's for AMD
                if optimal_config and "chunkwise_kernel" in optimal_config:
                    logging.info(f"Using AMD-optimized kernel configuration: {optimal_config}")
                    self.chunkwise_kernel = optimal_config["chunkwise_kernel"]
                    self.sequence_kernel = optimal_config["sequence_kernel"]
                    self.step_kernel = optimal_config["step_kernel"]
                else:
                    # Fall back to native kernels which perform well on AMD
                    logging.info("Using native kernels for AMD hardware")
                    self.chunkwise_kernel = "chunkwise--native_autograd"
                    self.sequence_kernel = "native_sequence__native" 
                    self.step_kernel = "native"

class xLSTMLarge(nn.Module):
    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config
        
        # For AMD hardware, enable dynamic optimization during forward passes
        self.is_amd = is_amd
        self.is_amd_mi300x = is_amd_mi300x
        
        # Create components
        # ... [keep existing initialization]
        
    def select_optimal_kernels(self, batch_size, seq_len):
        """
        Dynamically select optimal kernels based on current input dimensions.
        Only applies when running on AMD hardware.
        
        Args:
            batch_size: Current batch size
            seq_len: Current sequence length
        """
        if not self.is_amd or not hasattr(self, 'config'):
            return
            
        head_dim = self.config.embedding_dim // self.config.num_heads
        
        # Get optimal kernel configuration for this batch size and sequence length
        optimal_config = get_optimized_kernel_config(batch_size, seq_len, head_dim)
        
        # Apply configuration if needed
        if optimal_config and "chunkwise_kernel" in optimal_config:
            # Only update if different from current config
            if (optimal_config["chunkwise_kernel"] != self.config.chunkwise_kernel or
                optimal_config["sequence_kernel"] != self.config.sequence_kernel or
                optimal_config["step_kernel"] != self.config.step_kernel):
                
                logging.info(f"Dynamically updating kernel config for B={batch_size}, S={seq_len}")
                self.config.chunkwise_kernel = optimal_config["chunkwise_kernel"]
                self.config.sequence_kernel = optimal_config["sequence_kernel"]
                self.config.step_kernel = optimal_config["step_kernel"]
                
                # Update module references to use new kernel types
                # This might require reinitializing some modules depending on implementation
    
    def forward(self, x):
        # For AMD hardware, dynamically select optimal kernels based on input size
        if self.is_amd and self.is_amd_mi300x:
            batch_size = x.size(0)
            seq_len = x.size(1)
            self.select_optimal_kernels(batch_size, seq_len)
        
        # Continue with normal forward pass
        # ... [keep existing forward implementation] 