from setuptools import find_packages, setup

setup(
    name="lumina_ml_lib",
    version="0.1.0",
    description="ML library for Project Lumina, including models and data processing.",
    packages=find_packages(where="."),  # Find all packages in the current directory
    author="Ali Mahani",
)
