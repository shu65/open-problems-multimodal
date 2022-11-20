from distutils.core import setup

from setuptools import find_packages

install_requires = [
    "pandas",
    "scikit-learn",
    "lightgbm",
    "matplotlib",
    "seaborn",
    "numpy",
    "optuna>=3.0.0",
    "tables",
    "scipy==1.8.1",
    "torch",
    "pyarrow",
    "fastparquet",
    "gitpython",
    "pytest",
]


setup(
    name="ss_opm",
    version="0.0.2",
    description="scripts for Kaggle Open Problems - Multimodal Single-Cell Integration",
    author="Shuji Suzuki",
    author_email="dolphinripple@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
)
