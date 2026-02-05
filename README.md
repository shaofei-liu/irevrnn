# IRevRNN: Incorporating Reversibility into Recurrent Neural Networks

## Overview

A comprehensive research project exploring the integration of reversibility concepts into Recurrent Neural Networks (RNNs), culminating in a peer-reviewed scientific paper. This project demonstrates advanced understanding of neural network architectures, mathematical foundations, and practical implementation using PyTorch.

## Project Details

- **Title:** Incorporating Reversibility into Recurrent Neural Networks
- **Type:** Master's Thesis / Research Paper
- **Pages:** 76
- **Language:** English, German

## Key Components

### Research Paper
A peer-reviewed scientific paper presenting theoretical foundations, architectural innovations, and experimental results in reversible RNN design.

### Code Implementation
- **Language:** Python 3.7+
- **Deep Learning Framework:** PyTorch
- **GPU Support:** CUDA kernels for optimized computation

### Modules

#### 1. **torch_irevrnn** - Core Library
Implements the reversible RNN architecture with PyTorch bindings:
- `irevrnn.py` - Main RNN architecture definition
- `irevrnn_py.py` - Pure Python implementation
- `irevrnn_cuda.cpp` - CUDA/C++ interface
- `irevrnn_cuda_kernel.cu` - GPU kernels
- `irevrnn_cpp.cpp` - C++ implementation
- `common.py` - Shared utilities
- `cuda_activation.cuh` - CUDA activation functions

#### 2. **Data Processing** - Action Module
Handles data preparation and conversion:
- `action_datareader.py` - Data loading utilities
- `txt2npy.py` - Text to NumPy conversion
- `npy2data.py` - NumPy to data format conversion

#### 3. **Examples & Experiments**
- `mnist_main.py` - MNIST classification example
- `action_main.py` - Action recognition example
- `irevrnn_mnist_action_model.py` - Combined model

## Technical Stack

- **Deep Learning:** PyTorch (GPU-accelerated)
- **Numerical Computing:** NumPy
- **Optimization:** CUDA C++ for performance-critical operations
- **Programming Languages:** Python, C++, CUDA

## Key Features

- ✅ Reversible RNN architecture for memory-efficient backpropagation
- ✅ Hybrid Python/CUDA implementation for optimal performance
- ✅ Comprehensive experimental validation
- ✅ Support for various sequence modeling tasks
- ✅ Production-ready PyTorch integration

## Related Links

- **GitHub Repository:** [irevrnn](https://github.com/shaofei7/irevrnn)
- **Research Paper:** Available for review upon request
- **Portfolio:** [View on Shaofei's Portfolio](https://shaofei7.github.io/portfolio)

## Installation

```bash
# Install dependencies
pip install torch numpy

# Build C++ extensions
cd model
python setup.py build_ext --inplace
```

## Usage

```python
from torch_irevrnn import IRevRNN
import torch

# Initialize model
model = IRevRNN(input_size=10, hidden_size=64, num_layers=2)

# Forward pass
x = torch.randn(32, 20, 10)  # (batch, seq_len, input_size)
output, hidden = model(x)
```

## Citation

If you use this project in your research, please cite:

```bibtex
@mastersthesis{
  author={Shaofei Wang},
  title={Incorporating Reversibility into Recurrent Neural Networks},
  year={2023},
  pages={76}
}
```

## License

This project is provided for research and educational purposes. Please refer to the LICENSE file for details.

## Contact

For inquiries about this research, please visit [Shaofei's Portfolio](https://shaofei7.github.io/portfolio) or contact via GitHub.

---

**Note:** This repository contains the code implementation only. The research paper is available for academic review.
