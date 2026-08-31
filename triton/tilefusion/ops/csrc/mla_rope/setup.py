from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mla_rope_cuda',
    ext_modules=[
        CUDAExtension(
            name='mla_rope_cuda',
            sources=['mla_rope_fused.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
