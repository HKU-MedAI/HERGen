from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8")
version = "1.0"
package_name = "hergen"
description = "HERGen: Elevating Radiology Report Generation with Longitudinal Data"

setup(
    name=package_name,
    version=version,
    author="Fuying Wang",
    author_email="fuyingw@connect.hku.hk",
    description=description,
    long_description=long_description,
    packages=find_packages()
)