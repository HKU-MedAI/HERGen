from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8")
version = "1.0"
package_name = "biovlp"
description = "HERGen: Elevating Radiology Report Generation with Longitudinal Data"

setup(
    name=package_name,
    version=version,
    author="***",
    author_email="***",
    description=description,
    long_description=long_description,
    packages=find_packages()
)