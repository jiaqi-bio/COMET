from pathlib import Path
from setuptools import find_packages, setup

README = Path(__file__).resolve().parent / "README.md"
long_description = README.read_text(encoding="utf-8")

setup(
    name="comet-mihc",
    version="0.1.0",
    description="COmbinatorial Marker Expression Typing - mIHC rare cell quantification pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="COMET Authors",
    url="https://github.com/jiaqi-bio/COMET/",
    license="BSD-3-Clause",
    license_files=("LICENSE",),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "tifffile>=2023.1",
        "scikit-image>=0.21",
        "tqdm>=4.65",
        "matplotlib>=3.7",
        "cellpose",
        "nimbus-inference",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
