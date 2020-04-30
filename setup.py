# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os.path as osp

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

ext_args=dict(
    include_dirs=[np.get_include()],
    language='c++',
    exrta_complie_args={
        'cc':['-Wno-unused-function', 'Wno-write-strings'],
        'nvcc':['-c','--complier-options', '-fPIC'],
    }
)

this_dir=os.path.dirname(os.path.abspath(__file__))
extensions_dir=os.path.join(this_dir,"maskrcnn_benchmark", "csrc")
source_pyx=os.path.join(extensions_dir,"cpu","soft_nms_cpu.pyx")
extensions=[
    Extension("soft_nms_cpu", [source_pyx], **ext_args),
]

def customize_complier_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_complier_so=self.complier_so
    super=self._complier
    def _complier(obj,src,ext,cc_args,extra_postargs,pp_opts):
        if osp.splitext(src)[1]=='.cu':
            self.set_excutable('complier_so', 'nvcc')
            postargs=extra_postargs['nvcc']
        else:
            postargs=extra_postargs['cc']
        super(obj,src,ext,cc_args,postargs,pp_opts)
        self.complier_so=default_complier_so
    self._complie=_complier

class custom_buils_ext(build_ext):
    def build_extensions(self):
        customize_complier_for_nvcc(self.compiler)
        build_ext.build_extension(self)

setup(
    name='soft_nms',
    cmdclass={'build_ext': custom_buils_ext},
    ext_modules=cythonize(extensions),
)