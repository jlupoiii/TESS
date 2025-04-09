from setuptools import setup, find_packages

setup(
    name="beam",
    version="0.1.0",
    description="Background Elimination with Advanced Machine learning",
    author="Daniel Muthukrishna",
    author_email="daniel.muthukrishna@gmail.com",
    url="https://github.com/daniel-muthukrishna/beam",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "astropy>=4.0",
        "Pillow>=8.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.5b2",
            "isort>=5.8.0",
            "mypy>=0.812",
            "flake8>=3.9.0",
        ],
    },
)
