from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="comet-mihc",
    version="0.1.0",
    description="COmbinatorial Marker Expression Typing - mIHC rare cell quantification pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="COMET Authors",
    url="https://github.com/givlh123/COMET",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "tifffile>=2023.1",
        "scikit-image>=0.21",
        "tqdm>=4.65",
        "matplotlib>=3.7",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
