from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, include_paths, library_paths

setup(
    name='mytorchac',
    ext_modules=[
        CUDAExtension(
            name='mytorchac_cuda',
            sources=['backend/torchac_kernel.cu'],
            include_dirs=include_paths(),
            library_dirs=library_paths(),
            libraries=['torch', 'torch_cpu', 'torch_cuda'],
        ),
        CppExtension(
            name='mytorchac',
            sources=['backend/torchac_backend.cpp'],
            include_dirs=include_paths(),
            library_dirs=library_paths(),
            libraries=['torch', 'torch_cpu'],
        ),
    ],
    py_modules=['mytorchac'],
    cmdclass={
        'build_ext': BuildExtension
    }
)