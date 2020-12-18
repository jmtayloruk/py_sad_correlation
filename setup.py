from distutils.core import setup, Extension
import numpy  # So we can work out where the numpy headers live!
import platform
import os, sys

ARCH = []
if platform.platform().startswith("Windows"):
    # Note that I'm not sure if there's a way to say "use the best vector instruction set that's available",
    # so I just specify AVX at the moment. That covers everything I need for this code, and that should be
    # available on *most* machines we would want to compile on, I think.
    platform_specific_compile_args = ["/O2", "/aarch:AVX", "/std:c++17"]
else:
    # Note: -O4 emits a warning saying it's deprecated (and equivalent to -O3), so I just set -O3 here
    if True:
        platform_specific_compile_args = ["-O3", "-march=native", "-fno-lax-vector-conversions"]
    else:
        # Note: this was intended to be available as an alternative, to test on Intel platforms without any vector extensions
        # However note that, on my macbook pro at least, I encounter problems of "SSE register return with SSE disabled",
        # which seems to be related to the fact that all 64-bit processors have at least SSE2, and the compiler is
        # not expecting to have *nothing* at all to work with. I thought I had got this to work on other platforms, though,
        # so I will leave it here as an option for now.
        platform_specific_compile_args = ["-O3", "-mno-sse", "-mno-sse2", "-mno-sse3", "-mno-ssse3"]

if platform.platform().startswith("Darwin"):
    # Extra options for Mac OS
    platform_specific_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
    
    # Work out if we should be building a 32 or 64 bit library
    # Apparently this "can be a bit fragile" on OS X:
    # http://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
    # but I'll try it and see if it works out ok for now.
    archInfo = platform.architecture()
    if archInfo[0] == "32bit":
        ARCH = ["-arch", "i386"]
    else:
        ARCH = ["-arch", "x86_64"]

    # Determine if the -arch parameter is actually even available on this platform,
    # by running a dummy gcc command that includes that option
    # If it is not, then we will not include any arch-related options at all for gcc.
    # (note that this code may be redundant now that I only generate the above -arch command on OS X)
    theString = "gcc " + ARCH[0] + " " + ARCH[1] + " -E -dM - < /dev/null > /dev/null 2>&1"
    result = os.system(theString)
    if result != 0:
        ARCH = []
    platform_specific_compile_args += ARCH


# Add this flag, which is essential when building on a raspberry pi
if platform.machine().startswith("arm"):
    platform_specific_compile_args += ["-mfpu=neon"]


def get_extra_build_options():
    # This function is cribbed from the source code for the 'dlib' module (a random module I found on the internet that does this!)
    """read -compiler-flags option from the command line and add it to compiler options.
        """
    _extra_build_options = []

    opt_key = None

    argv = [arg for arg in sys.argv]  # take a copy
    # parse command line options and consume those we care about
    for arg in argv:
        if opt_key == "compiler-flags":
            _extra_build_options.append("{arg}".format(arg=arg.strip()))

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
            continue

        if arg in ["--compiler-flags"]:
            opt_key = arg[2:].lower()
            sys.argv.remove(arg)
            continue
    return _extra_build_options


extra_command_line_compile_args = get_extra_build_options()

j_py_sad_correlation = Extension(
    "j_py_sad_correlation",
    include_dirs=["/usr/local/include", numpy.get_include()],
    sources=[
        "j_py_sad_correlation.cpp",
        "common/jPythonArray.cpp",
        "common/jPythonCommon.cpp",
        "common/PIVImageWindow.cpp",
        "common/jAssert.cpp",
        "common/DebugPrintf_Unix.cpp",
    ],
    extra_link_args=ARCH,
    extra_compile_args=extra_command_line_compile_args+platform_specific_compile_args
)

setup(
    name="j_py_sad_correlation",
    version="1.0.0",
    description="High-performance functions for calculating sum-of-absolute-differences in Python.",
    author="Jonathan M Taylor",
    ext_modules=[j_py_sad_correlation],
)
