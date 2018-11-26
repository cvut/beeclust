from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


setup(
    name='beeclust',
    version='0.2',
    description='BeeClust swarming algorithm with Python\'s NumPy and Cython',
    license='MIT',
    packages=find_packages(),
    ext_modules=cythonize('beeclust/_speedups.pyx'),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'NumPy',
    ],
    setup_requires=[
        'Cython',
        'NumPy',
    ],
)
