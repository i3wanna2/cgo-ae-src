from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mla_rope_quant_cuda',
    ext_modules=[
        CUDAExtension(
            name='mla_rope_quant_cuda',
            sources=['mla_rope_quant_fused.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
