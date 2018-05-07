# setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='preprocessing',
    ext_modules=cythonize('preprocessing.pyx')
)

