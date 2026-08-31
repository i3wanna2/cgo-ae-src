import os
import glob
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = os.path.join(ROOT_DIR, "ops", "csrc")

# 获取当前环境 Torch 的库路径
TORCH_LIB_DIR = os.path.join(os.path.dirname(torch.__file__), "lib")

sources = glob.glob(os.path.join(CSRC_DIR, "*.cpp")) + \
          glob.glob(os.path.join(CSRC_DIR, "*.cu"))

setup(
    name='tilefusion',  
    version='0.1.0',
    author='mzy',
    description='Kernel Fusion Native Ops',
    packages=['tilefusion', 'tilefusion.core', 'tilefusion.ops', 'tilefusion.fusions', 'tilefusion.models', 'tilefusion.utils', 'tilefusion.resources'],
    package_dir={'tilefusion': '.'}, 
    ext_modules=[
        CUDAExtension(
            name='tilefusion.ops.tilefusion_native', 
            sources=sources,
            library_dirs=[TORCH_LIB_DIR],
            runtime_library_dirs=[TORCH_LIB_DIR],
            extra_link_args=['-Wl,-rpath,' + TORCH_LIB_DIR],
            extra_compile_args={
                'cxx': ['-O3'],         
                'nvcc': ['-O3']         
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)