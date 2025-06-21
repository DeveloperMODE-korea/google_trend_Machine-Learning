"""
Setup script for Google Trends Machine Learning Predictor.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="google-trends-ml-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning system for predicting Google Trends search volumes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/google_trend_Machine-Learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "trends-predict=src.predictor:main",
            "trends-collect=src.data.collector:main",
            "trends-preprocess=src.data.preprocessor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
) 