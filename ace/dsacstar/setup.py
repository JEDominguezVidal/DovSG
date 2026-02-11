from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

opencv_inc_dir = '/usr/include/opencv4'
opencv_lib_dir = '/usr/lib/x86_64-linux-gnu'

setup(
    name='dsacstar',
    ext_modules=[CppExtension(
        name='dsacstar',
        sources=['dsacstar.cpp','thread_rand.cpp'],
        include_dirs=[opencv_inc_dir],
        library_dirs=[opencv_lib_dir],
        libraries=['opencv_core','opencv_calib3d'],
        extra_compile_args=['-fopenmp']
    )],
    cmdclass={'build_ext': BuildExtension}
)

