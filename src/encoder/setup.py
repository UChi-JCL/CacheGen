# from setuptools import setup, Extension
# from torch.utils import cpp_extension

# setup(
#     name='mytorchac_cuda',
#     ext_modules=[
#         cpp_extension.CUDAExtension('mytorchac_cuda', 
#                                     ['torchac_kernel.cu']),
#     ],
#     py_modules=['mytorchac'],
#     cmdclass={
#         'build_ext': cpp_extension.BuildExtension
#     }
# )

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mytorchac_cuda',
    ext_modules=[
        CUDAExtension(
            name='mytorchac_cuda',
            sources=['torchac_kernel.cu'],
        ),
    ],
    py_modules=['mytorchac'],
    cmdclass={
        'build_ext': BuildExtension
    },
)