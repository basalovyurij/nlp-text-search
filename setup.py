from pkg_resources import parse_requirements
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="nlp-text-search",
    version="0.5",
    author="Yurij Basalov",
    author_email="basalov_yurij@mail.ru",
    description="Fulltext-like search using NLP concept",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/basalovyurij/nlp-text-search",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_reqs=parse_requirements('requirements.txt')
)