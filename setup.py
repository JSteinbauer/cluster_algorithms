import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def install_requirement_list(fname: str):
    return read(fname).split("\n")


setup(
    name="cluster_algorithms",
    version="0.0.1",
    author="Jakob Steinbauer",
    author_email="jakob_steinbauer@hotmail.com",
    description=("A collection of unsupervised cluster algorithms."),
    license="free",
    keywords="kmeans cosine similarity",
    packages=["kmeans"],
    install_requires=install_requirement_list("requirements.txt"),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
