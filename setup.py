from setuptools import setup, find_packages
import os

try:
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "" 

setup(
    name="neuralstorm",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=[
        "torch>=2.0.0", 
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0", 
        "tqdm>=4.60.0",
    ],
    author="Mahdi Seddigh",
    author_email="your.email@example.com",
    description="A machine learning project for predicting power outages caused by extreme weather events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mhdaw/NeuralStorm", 
    keywords="machine-learning deep-learning pytorch weather power-outage prediction time-series",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science", 
        "Topic :: Utilities", 
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8", 
)