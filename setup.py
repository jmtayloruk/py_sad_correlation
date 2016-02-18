
from distutils.core import setup, Extension

#
# Global parameters
#
ARCH='i386'
ARCH2='x86_64'

BUILD_MODULES=[]



j_py_sad_correlation = Extension('j_py_sad_correlation',
	include_dirs = ['/usr/local/include',
					'/Library/Python/2.6/site-packages/numpy/core/include/'],
	library_dirs=['/usr/local/lib'],
	sources = ['j_py_sad_correlation.cpp', 'common/JPythonArray.cpp', 'common/JPythonCommon.cpp', 'common/jAssert.cpp'],
	extra_link_args=['-arch', ARCH2],
	extra_compile_args=['-O4', '-arch', ARCH2]
)
BUILD_MODULES.append(j_py_sad_correlation)


setup (ext_modules = BUILD_MODULES)




