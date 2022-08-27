##from pkg_resources import parse_version
##assert parse_version(setuptools.__version__)>=parse_version('58.2.0')

#import setuptools
from setuptools import setup, find_packages

#with open("README.md", "r") as fh:
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='robust_deep_learning',  
    version="0.1.1",
    author="David Macêdo",
    author_email="dlm@cin.ufpe.br",
    description="The Robust Deep Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/dlmacedo/robust-deep-learning',
    #scripts=['rdl'],
    #packages=find_packages(exclude=['loaders', 'utils', 'models']),
    #packages=setuptools.find_packages(),
    packages=['robust_deep_learning'],
    include_package_data=True,
    python_requires='>=3.6',
    keywords='pytorch isomax isomax+ dismax',
    install_requires=[
        'torch >= 1.10',
        'torchvision',
        'torchmetrics',
        'torchnet',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'timm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'License :: OSI Approved :: Apache Software License',
    ],
)
