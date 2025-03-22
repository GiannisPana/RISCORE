"""Setup script for RISCORE package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.13.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.22.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "nltk>=3.8.0",
        "pyyaml>=6.0",
        "huggingface-hub>=0.16.0",
    ]

setup(
    name="riscore",
    version="0.1.0",
    description="RISCORE: RIddle Solving with COntext REconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RISCORE Team",
    author_email="",
    url="https://github.com/yourusername/RISCORE",
    packages=find_packages(exclude=["tests", "prompting", "data", "output", "test_set"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "riscore=riscore.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp, riddles, language-models, prompting, few-shot-learning, chain-of-thought",
)
