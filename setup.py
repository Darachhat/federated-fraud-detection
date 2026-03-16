"""
Package installation configuration.
Enables the project to be installed as a Python package,
making all src/ imports available without path manipulation.

Usage:
    pip install -e .          # Editable install (development)
    pip install .             # Standard install
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="federated-fraud-detection",
    version="1.0.0",
    author="Sothun Darachhat",
    author_email="",
    description=(
        "A Federated Learning framework for early fraud detection "
        "using XGBoost and the JSON Tree Concatenation Algorithm."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/federated-fraud-detection",
    packages=find_packages(
        where=".",
        include=["src*"]
    ),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            # Convenience CLI commands after pip install
            "fl-prepare=scripts.prepare_data:main",
            "fl-baseline-local=scripts.run_baseline_local:main",
            "fl-baseline-central=scripts.run_baseline_central:main",
            "fl-server=scripts.run_server:main",
            "fl-simulate=scripts.run_simulation:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)