from setuptools import setup,find_packages

HYPEN_DOT="-e ."

def get_requirements(file_path):
    """
    This function reads a requirements file and returns a list of packages.
    """
    with open(file_path, "r") as file:
        requirements = file.readlines()
    
    requirements = [req.strip() for req in requirements if req.strip()]
    if HYPEN_DOT in requirements:
        requirements.remove(HYPEN_DOT)
    
    return requirements

setup(
    name="End to End ML Project",
    version="0.1",
    author="Muhammad Uzair",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)