from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension, CppExtension, CUDAExtension)

setup(
    name='torch-irevrnn',
    ext_modules=[
        CppExtension('torch_irevrnn.irevrnn_cpp', [
            'torch_irevrnn/irevrnn_cpp.cpp']),
        CUDAExtension('torch_irevrnn.irevrnn_cuda', [
            'torch_irevrnn/irevrnn_cuda.cpp',
            'torch_irevrnn/irevrnn_cuda_kernel.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
