"""Setuptools magic to install MetaBGC."""
import os
from setuptools import setup, find_packages


def read(fname):
    """Read a file from the current directory."""
    return open(os.path.join(os.path.dirname(__file__), fname),encoding="utf8").read()

long_description = read('README.md')

install_requires = [
    "click",
    "stardist==0.8.1",
    "opencv-python==4.5.5.64",
    "pycocotools @ git+https://github.com/bhoeckendorf/pyklb.git@skbuild",
    "pycocotools @ git+https://github.com/abiswas-odu/roi_convertor.git"
]

def read_version():
    """Read the version from the appropriate place in the library."""
    for line in open(os.path.join("src","stardist_inference",'__main__.py'), 'r'):
        if line.startswith('__version__'):
            return line.split('=')[-1].strip().strip('"')

setup(
    name="stardist_inference",
    python_requires='>=3.9',
    version=read_version(),
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    author='Posfai lab development team.',
    author_email='ab50@princeton.edu',
    description='Perform stardist 3d inference.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['stardist_inference=src.stardist_inference.__main__:main'],
    },
    url='https://github.com/abiswas-odu/stardist_inference.git',
    license='GNU General Public License v3',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
    ]
)