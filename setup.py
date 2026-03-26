from setuptools import setup, find_packages

setup(
    name="fraud-detection-feature-store",
    version="1.0.0",
    description="Real-time feature store for fraud detection with dual-path architecture",
    author="Data Engineering Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark>=3.4.0",
        "kafka-python>=2.0.2",
        "redis>=5.0.0",
        "snowflake-connector-python>=3.6.0",
        "pydantic>=2.5.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",
        "structlog>=23.2.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "pyarrow>=14.0.0",
        "pyiceberg>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "fakeredis>=2.20.0",
            "freezegun>=1.3.0",
            "black>=23.12.0",
            "ruff>=0.1.0",
            "mypy>=1.8.0",
        ],
    },
)
