"""
Setup script for Cryptocurrency Event Study package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="crypto-event-study",
    version="1.0.0",
    author="Murad Farzulla",
    author_email="murad@farzulla.org",
    description="Cryptocurrency event study analysis using GARCH models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/studiofarzulla/event_study",
    packages=find_packages(include=['code', 'code.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "arch>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-event-study=code.run_event_study_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "code": ["../data/*.csv"],
    },
    zip_safe=False,
)
