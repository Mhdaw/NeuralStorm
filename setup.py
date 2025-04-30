from setuptools import setup, find_packages

setup(
    name="neuralstorm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=1.0.0",  # For metrics calculations
        "pandas>=1.3.0"         # For data handling
    ],
    author="NeuralStorm Team",
    description="A machine learning project for predicting power outages caused by extreme weather events",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)