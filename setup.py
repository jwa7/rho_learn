from setuptools import setup, find_packages

setup(
    name="rholearn",
    version="0.0.0",
    packages=find_packages(include=["rholearn", "rholearn.*", "docs.example.azoswitch"]),
)
