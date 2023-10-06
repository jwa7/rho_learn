from setuptools import setup, find_packages

setup(
    name="rho_learn",
    version="0.0.0",
    packages=find_packages(
        include=[
            "rhocalc",
            "rhocalc.*",
            "rholearn", 
            "rholearn.*",
            "rhopredict",
            "rhopredict.*"
            ]
        ),
)
