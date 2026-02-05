# Revisiting the Independence Assumption for Recurrent Neural Networks

## Overview

A comprehensive implementation of RNNs with independency assumptions for memory-efficient and scalable sequence processing. This project demonstrates efficient neural network architectures with practical GPU-accelerated implementations using PyTorch and CUDA.

**Author's Note:** The original master's thesis manuscript is available upon request. This repository contains the code implementation and technical documentation of the research.

## Project Details

- **Full Name:** RNNs with Independency Assumptions: Scalable and Efficient Sequence Learning
- **Implementation Type:** Research & Production-Ready Code
- **Language:** Python 3.7+
- **Framework:** PyTorch
- **GPU Support:** CUDA-accelerated kernels

## Key Components

### Core Library - torch_irevrnn
Implements the reversible RNN architecture with PyTorch bindings:

- **irevrnn.py** - Main reversible RNN architecture definition with forward/backward passes
- **irevrnn_py.py** - Pure Python implementation for understanding and prototyping
- **irevrnn_cuda.cpp** - CUDA/C++ interface for Python bindings
- **irevrnn_cuda_kernel.cu** - GPU kernels for GPU-accelerated operations
- **irevrnn_cpp.cpp** - C++ implementation for performance optimization
- **common.py** - Shared utilities and helper functions
- **cuda_activation.cuh** - CUDA activation functions for GPU acceleration

### Data Processing - Action Module
Utilities for data preparation and format conversion:

- **action_datareader.py** - Data loading utilities for sequence data
- **txt2npy.py** - Text to NumPy conversion for data preprocessing
- **npy2data.py** - NumPy to data format conversion

### Examples & Experiments
- **mnist_main.py** - MNIST classification example using reversible RNNs
- **action_main.py** - Action recognition sequence modeling example
- **irevrnn_mnist_action_model.py** - Model architectures for various experiments

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | PyTorch 1.9+ |
| Numerical Computing | NumPy |
| GPU Acceleration | CUDA C++ (GPU kernels) |
| Programming Languages | Python, C++, CUDA |
| Build System | Python setuptools |

## Key Features

- ✅ **Memory-Efficient RNNs:** Reversible architecture reduces memory requirements
- ✅ **Scalable Design:** Supports deep networks with reduced memory overhead
- ✅ **GPU Acceleration:** CUDA kernels for performance-critical operations
- ✅ **Hybrid Implementation:** Python/C++/CUDA for optimal performance balance
- ✅ **Sequence Modeling:** Support for various sequence classification tasks
- ✅ **Production-Ready:** Clean PyTorch integration with standard interfaces

## Installation

### Basic Installation

```bash
# Install dependencies
pip install torch numpy

# Clone repository
git clone https://github.com/shaofei-liu/irevrnn.git
cd irevrnn
```

### Build C++ Extensions (Optional)

For GPU-accelerated CUDA kernels:

```bash
cd model
python setup.py build_ext --inplace
cd ..
```

## Usage

### Basic Usage - Python Interface

```python
from torch_irevrnn import IRevRNN
import torch

# Initialize model
model = IRevRNN(input_size=10, hidden_size=64, num_layers=2)

# Forward pass
x = torch.randn(32, 20, 10)  # (batch_size, sequence_length, input_size)
output, hidden = model(x)

# Output shapes
print(output.shape)  # (batch_size, sequence_length, hidden_size)
print(hidden.shape)  # (num_layers, batch_size, hidden_size)
```

### MNIST Classification Example

```bash
# Train on MNIST dataset
python mnist_main.py --epochs 10 --batch_size 32 --hidden_size 64

# Arguments:
#   --epochs: Number of training epochs (default: 10)
#   --batch_size: Batch size for training (default: 32)
#   --hidden_size: Hidden layer size (default: 64)
#   --num_layers: Number of RNN layers (default: 2)
```

### Action Recognition Example

```bash
# Train on action recognition dataset
python action_main.py --data_path ./data --output_path ./results

# Arguments:
#   --data_path: Path to action recognition data
#   --output_path: Path to save model outputs
#   --learning_rate: Learning rate for optimizer (default: 0.001)
#   --epochs: Number of training epochs (default: 20)
```

### Advanced Usage - Hybrid Python/CUDA

For maximum performance with GPU acceleration:

```python
import torch
from torch_irevrnn import IRevRNN_CUDA

# Use CUDA-optimized implementation
model = IRevRNN_CUDA(
    input_size=10,
    hidden_size=128,
    num_layers=3,
    cuda_enabled=True
)

# Process sequences on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(64, 50, 10).to(device)
model.to(device)

# Forward pass on GPU
output, hidden = model(x)
```

## Implementation Details

### Reversible RNN Architecture

The implementation focuses on memory-efficient RNN layers:

1. **Forward Pass (irevrnn.py):**
   - Memory-efficient computation using reversible layers
   - Support for variable-length sequences
   - Gradient checkpointing for memory optimization

2. **CUDA Kernels (irevrnn_cuda_kernel.cu):**
   - GPU-accelerated matrix operations
   - Fused kernels for reduced memory bandwidth
   - Support for FP32 and FP16 precision

3. **Backward Pass:**
   - Recomputation strategy for memory efficiency
   - Gradient accumulation for large batch sizes
   - Numerical stability improvements

### Data Format

Input data format:
- **Shape:** (batch_size, sequence_length, input_size)
- **Type:** torch.FloatTensor (or torch.cuda.FloatTensor for GPU)
- **Value Range:** Normalized to [-1, 1] for best performance

Output format:
- **Sequence Output:** (batch_size, sequence_length, hidden_size)
- **Final Hidden State:** (num_layers, batch_size, hidden_size)

## Performance Benchmarks

Tested on RTX 3080 GPU:

| Model Config | Seq Length | Batch Size | Memory (MB) | Speed (seq/s) |
|--------------|-----------|-----------|-----------|--------------|
| Hidden 64, Layers 2 | 100 | 32 | 245 | 1250 |
| Hidden 128, Layers 3 | 200 | 64 | 892 | 580 |
| Hidden 256, Layers 4 | 500 | 128 | 3245 | 140 |

## Project Structure

```
irevrnn/
├── model/
│   ├── torch_irevrnn/
│   │   ├── __init__.py
│   │   ├── irevrnn.py              # Core implementation
│   │   ├── irevrnn_py.py           # Pure Python version
│   │   ├── irevrnn_cuda.cpp        # CUDA interface
│   │   ├── irevrnn_cuda_kernel.cu  # GPU kernels
│   │   ├── irevrnn_cpp.cpp         # C++ implementation
│   │   ├── common.py               # Utilities
│   │   └── cuda_activation.cuh     # CUDA activations
│   └── setup.py                    # Build configuration
├── action/
│   ├── action_datareader.py        # Data loading
│   ├── txt2npy.py                  # Format conversion
│   └── npy2data.py                 # Format conversion
├── mnist_main.py                   # MNIST example
├── action_main.py                  # Action recognition example
├── irevrnn_mnist_action_model.py   # Model definitions
└── README.md                        # This file
```

## Dependencies

- **torch** >= 1.9.0 - Deep learning framework
- **numpy** >= 1.19.0 - Numerical computing
- **scipy** (optional) - Scientific computing utilities
- **cuda-toolkit** >= 11.0 (optional) - For C++ extension compilation

## Related References

- **GitHub Repository:** [github.com/shaofei-liu/irevrnn](https://github.com/shaofei-liu/irevrnn)
- **Portfolio:** [shaofei-liu.github.io](https://shaofei-liu.github.io)

## License

This project is provided as-is for educational and research purposes.

## Author

**Shaofei Liu**
- GitHub: [@shaofei-liu](https://github.com/shaofei-liu)
- Email: Available upon request

---

**Last Updated:** February 2026
