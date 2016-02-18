
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
# This is a problem. On OS X I need to locally edit $PATH to remove anaconda because that installs python 3 and the code I have is not compatible with python 3!
# Need to work out a solution to this, and also a solution that enables me NOT to hard-code the path to the numpy header files
#					'/Users/jonny/anaconda/lib/python3.5/site-packages/numpy/core/include/'],
	library_dirs=['/usr/local/lib'],
	sources = ['j_py_sad_correlation.cpp', 'common/JPythonArray.cpp', 'common/JPythonCommon.cpp', 'common/jAssert.cpp'],
# Note: the -arch stuff below may be needed for OS X, but should be removed for Linux since it's a mac-specific parameter that is not recognised.
	extra_link_args=['-arch', ARCH2],
	extra_compile_args=['-O4', '-arch', ARCH2]
)
BUILD_MODULES.append(j_py_sad_correlation)


setup (ext_modules = BUILD_MODULES)




