from setuptools import setup, find_packages

setup(
    name="autofepg",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=5.0",
            "black>=22.0",
            "isort>=5.0",
        ],
    },
    python_requires=">=3.8",
    author="AutoFE-PG Contributors",
    description="AutoFE - Playground: Automatic Feature Engineering & Selection for Kaggle Playground Competitions",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thomastschinkel/autofepg",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
