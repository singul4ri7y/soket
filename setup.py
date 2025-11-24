import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _bext
from Cython.Build import cythonize
from multiprocessing import cpu_count
import numpy


## CONFIGURATIONS ##
DEBUG_C_FLAGS = '-O0 -g -Wall -Wextra -Wformat-security '
RELEASE_C_FLAGS = '-O3 -s -funroll-loops'
DEBUG_LFLAGS = ' -shared'
RELEASE_LFLAGS = ' -shared -s'


class build_debug(_bext):
    def build_extensions(self):
        # Set custom build folder
        self.build_lib = os.path.join('build', 'debug')

        flags = ' -fPIC -fno-strict-aliasing ' + DEBUG_C_FLAGS

        # Override compiler and flags
        self.compiler.set_executable('compiler_so', 'gcc' + flags)
        self.compiler.set_executable('compiler_cxx', 'g++' + flags)
        self.compiler.set_executable('linker_so', 'gcc' + DEBUG_LFLAGS)
        _bext.build_extensions(self)

    def finalize_options(self):
        super().finalize_options()
        self.parallel = cpu_count()


class build_release(_bext):
    def build_extensions(self):
        # Set custom build folder
        self.build_lib = os.path.join('build', 'release')

        flags = ' -fPIC -fno-strict-aliasing ' + RELEASE_C_FLAGS

        # Override compiler and flags
        self.compiler.set_executable('compiler_so', 'gcc' + flags)
        self.compiler.set_executable('compiler_cxx', 'g++' + flags)
        self.compiler.set_executable('linker_so', 'gcc' + RELEASE_LFLAGS)
        _bext.build_extensions(self)

    def finalize_options(self):
        super().finalize_options()
        self.parallel = cpu_count()


if __name__ == '__main__':
    # Collect .pyx files and prepare extensions
    cython_files = []
    for root, _, files in os.walk('soket/'):
        for f in files:
            if f.endswith('.pyx'):
                cython_files.append(os.path.join(root, f))

    extensions = [
        Extension(
            name=os.path.splitext(path.replace(os.sep, '.'))[0],  # Module name
            sources=[path],
            include_dirs=[numpy.get_include()],
            define_macros=[('CYTHON_LIMITED_API', '0')],
            py_limited_api=False
        )
        for path in cython_files
    ]

    setup(
        name='soket',
        version='0.0.1',
        python_requires='>=3.13',
        ext_modules=cythonize(
            extensions,
            annotate=True,
            nthreads=cpu_count(),
            compiler_directives={
                'language_level': '3',
                'boundscheck': False,
                'wraparound': False,
                'initializedcheck': False,
                'cdivision': True
            },
        ),
        packages=find_packages(),
        zip_safe=False,
        cmdclass={
            'build_debug': build_debug,
            'build_release': build_release,
            'build_ext': build_release
        }
    )
